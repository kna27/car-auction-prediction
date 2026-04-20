import json
import os
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor,
                              VotingRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error, r2_score)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sqlalchemy import create_engine
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from src.model.visualizer import _model_slug

load_dotenv()

MODEL_DIR = os.path.join("src", "model", "saved_models")
RESULTS_DIR = os.path.join("src", "model", "results")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Database connection parameters
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "car_auctions")

def get_engine():
    if DB_PASSWORD:
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    else:
        connection_string = f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)

def load_data_from_db():
    engine = get_engine()
    query = """
        SELECT 
            a.year,
            a.mileage,
            a.title_status,
            a.state,
            a.engine,
            a.drivetrain,
            a.body_style,
            a.num_modifications,
            a.has_forced_induction,
            a.sale_price,
            mk.name as make,
            md.name as model,
            t.name as transmission,
            ec.name as exterior_color,
            ic.name as interior_color
        FROM auctions a
        LEFT JOIN models md ON a.model_id = md.id
        LEFT JOIN makes mk ON md.make_id = mk.id
        LEFT JOIN transmissions t ON a.transmission_id = t.id
        LEFT JOIN colors ec ON a.exterior_color_id = ec.id
        LEFT JOIN colors ic ON a.interior_color_id = ic.id;
    """
    df = pd.read_sql(query, engine)
    return df

def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    # Small threshold filters out noise like rare colors/states for small datasets
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='Unknown')),
        ('onehot', OneHotEncoder(handle_unknown='infrequent_if_exist', min_frequency=0.05))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    rf = RandomForestRegressor(random_state=42, criterion='absolute_error')
    gb = GradientBoostingRegressor(random_state=42, loss='huber')
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 20))
    lasso = LassoCV(alphas=np.logspace(-3, 3, 20), random_state=42)
    
    ensemble = VotingRegressor([
        ('rf', rf), 
        ('gb', gb),
        ('ridge', ridge),
        ('lasso', lasso)
    ])
    
    log_target_model = TransformedTargetRegressor(
        regressor=ensemble,
        func=np.log1p,
        inverse_func=np.expm1
    )
    
    return Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', log_target_model)
    ])

def extract_feature_importances(pipeline, numeric_features, categorical_features):
    transformed_model = pipeline.named_steps['model']
    ensemble_model = transformed_model.regressor_
    rf_model = ensemble_model.named_estimators_['rf']
    
    preprocessor = pipeline.named_steps['preprocessor']
    cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
    cat_feature_names = cat_encoder.get_feature_names_out(categorical_features)
    
    feature_names = numeric_features + list(cat_feature_names)
    importances = rf_model.feature_importances_
    
    df_imp = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    return df_imp

def train_and_evaluate(only_models: Optional[List[str]] = None):
    print("Loading data from PostgreSQL database...")
    df = load_data_from_db()
    df = df.dropna(subset=['sale_price']).copy()
    
    print("\n=============================================")
    print("TRAINING INDIVIDUAL CAR MODELS")
    print("=============================================")
    
    X = df.drop(columns=['sale_price'])
    y = df['sale_price']
    
    numeric_features = ['year', 'mileage', 'num_modifications', 'has_forced_induction']
    cat_features = ['title_status', 'state', 'engine', 'drivetrain', 'transmission', 'body_style', 'exterior_color', 'interior_color']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    trained_models = {}
    test_results_list = []
    per_model_metrics: dict = {}
    
    train_indices = X_train.index
    test_indices = X_test.index
    
    candidate_models = list(df['model'].dropna().unique())
    if only_models:
        only = set(map(str, only_models))
        candidate_models = [m for m in candidate_models if str(m) in only]

    for car_model in candidate_models:
        print(f"\n--- Model: {car_model} ---")
        
        car_train_df = df.loc[train_indices]
        car_train_df = car_train_df[car_train_df['model'] == car_model]
        
        car_test_df = df.loc[test_indices]
        car_test_df = car_test_df[car_test_df['model'] == car_model]
            
        X_train_car = car_train_df.drop(columns=['sale_price', 'make', 'model'])
        y_train_car = car_train_df['sale_price']
        
        X_test_car = car_test_df.drop(columns=['sale_price', 'make', 'model'])
        y_test_car = car_test_df['sale_price']
        
        # Filter out constant categorical features to reduce noise
        valid_cat_features = [col for col in cat_features if X_train_car[col].nunique() > 1]
        
        pipeline = build_pipeline(numeric_features, valid_cat_features)
        
        # Hyperparameter search space
        param_dist = {
            'model__regressor__rf__n_estimators': [50, 100, 200],
            'model__regressor__rf__max_depth': [3, 5, 7, 10],
            'model__regressor__rf__min_samples_leaf': [1, 2, 4],
            'model__regressor__gb__n_estimators': [50, 100, 150],
            'model__regressor__gb__learning_rate': [0.01, 0.05, 0.1],
            'model__regressor__gb__max_depth': [2, 3, 5],
            'model__regressor__gb__min_samples_leaf': [1, 2, 4]
        }
        
        print(f"Tuning ensemble on {len(car_train_df)} samples...")
        random_search = RandomizedSearchCV(
            pipeline, 
            param_dist, 
            n_iter=50, 
            cv=5, 
            scoring='neg_mean_absolute_error', 
            n_jobs=-1, 
            random_state=42
        )
        random_search.fit(X_train_car, y_train_car)
        best_model = random_search.best_estimator_
        
        trained_models[car_model] = best_model
        
        preds = best_model.predict(X_test_car)
        mae = mean_absolute_error(y_test_car, preds)
        r2 = r2_score(y_test_car, preds)
        mape = mean_absolute_percentage_error(y_test_car, preds)
        print(f"Test MAE : ${mae:.2f}")
        print(f"Test MAPE: {mape:.2%}")
        print(f"Test R2  : {r2:.3f}")

        per_model_metrics[_model_slug(car_model)] = {
            "MAE": float(mae),
            "MAPE": float(mape),
            "R2": float(r2),
        }

        car_test_results = car_test_df.copy()
        car_test_results['Predicted'] = preds
        car_test_results['Actual'] = y_test_car
        test_results_list.append(car_test_results)

        # Save per-model predictions (used to draw per-model charts on incremental runs)
        car_test_results.to_csv(
            os.path.join(RESULTS_DIR, f"test_predictions_{_model_slug(car_model)}.csv"),
            index=False,
        )
        
        feature_imp = extract_feature_importances(best_model, numeric_features, valid_cat_features)
        feature_imp.to_csv(
            os.path.join(RESULTS_DIR, f"feature_importances_{_model_slug(car_model)}.csv"),
            index=False,
        )

    print("\n=============================================")
    print("FINAL AGGREGATE PERFORMANCE")
    print("=============================================")
    
    if test_results_list:
        test_results = pd.concat(test_results_list)
        # Update the aggregate file (merge on incremental runs, overwrite on full runs).
        agg_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
        if only_models and os.path.exists(agg_path):
            try:
                existing_agg = pd.read_csv(agg_path)
                only_set = set(map(str, only_models))
                # Remove existing rows for models we just trained to avoid duplicates
                existing_agg = existing_agg[~existing_agg["model"].astype(str).isin(only_set)]
                test_results = pd.concat([existing_agg, test_results], ignore_index=True)
            except Exception as e:
                print(f"Warning: Could not merge with existing aggregate results: {e}")

        test_results.to_csv(agg_path, index=False)

        # Recalculate aggregate performance across ALL models now in the CSV
        aggregate_mae = mean_absolute_error(test_results['Actual'], test_results['Predicted'])
        aggregate_r2 = r2_score(test_results['Actual'], test_results['Predicted'])
        aggregate_mape = mean_absolute_percentage_error(test_results['Actual'], test_results['Predicted'])
        
        print(f"Aggregate MAE : ${aggregate_mae:.2f}")
        print(f"Aggregate MAPE: {aggregate_mape:.2%}")
        print(f"Aggregate R2  : {aggregate_r2:.3f}")
        
        # Save summary (merge per_model into existing on incremental runs)
        summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
        existing = {}
        if os.path.exists(summary_path):
            with open(summary_path, "r") as f:
                existing = json.load(f) or {}

        merged_per_model = dict(existing.get("per_model") or {})
        merged_per_model.update(per_model_metrics)

        results_summary = {
            "MAE": float(aggregate_mae),
            "R2": float(aggregate_r2),
            "MAPE": float(aggregate_mape),
            "per_model": merged_per_model,
        }
        with open(summary_path, "w") as f:
            json.dump(results_summary, f, indent=4)

        from src.model.visualizer import generate_all_visualizations as _gen_all
        from src.model.visualizer import \
            plot_per_model_accuracy_lines as _gen_per_model

        # Always regenerate all visualizations to ensure the aggregate dashboard is up to date.
        _gen_all()
            
        # Clean up old local/global files if they exist
        old_files = ["best_global_rf.joblib", "global_feature_importances.csv", "test_predictions_global.csv", "test_predictions_local.csv"]
        for gf in old_files:
            gf_path = os.path.join(MODEL_DIR if "joblib" in gf else RESULTS_DIR, gf)
            if os.path.exists(gf_path):
                os.remove(gf_path)
    else:
        print("Not enough data to evaluate models.")
        
    for car_model, model in trained_models.items():
        joblib.dump(model, os.path.join(MODEL_DIR, f"model_{car_model.replace(' ', '_')}.joblib"))
        
    print(f"\nAll models saved to {MODEL_DIR}")

if __name__ == "__main__":
    train_and_evaluate()
