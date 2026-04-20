import json
import os
import re
import sys
from urllib.parse import urlparse
from typing import Any, Dict, List

import joblib
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlalchemy import text

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _ROOT)

from src.model.predict_auction import MODEL_DIR, RESULTS_DIR, scrape_url
from src.data_cleaning.clean_data import clean_dataset
from src.pipeline.training import get_training_state, run_full_retrain, run_training_pipeline
from src.database.load_data import get_engine
from src.eda.visualize import _model_slug, generate_all_visualizations

VIS_DIR = os.path.join(_ROOT, "visualizations")
PROCESSED_PATH = os.path.join(_ROOT, "data", "processed", "all_vehicles_cleaned.csv")

app = FastAPI(title="Car Auction Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if os.path.isdir(VIS_DIR):
    app.mount("/visualizations", StaticFiles(directory=VIS_DIR), name="visualizations")


class TrainRequest(BaseModel):
    search_path: str


class PredictRequest(BaseModel):
    url: str


def _load_json(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _viz_exists(name: str) -> bool:
    p = os.path.join(VIS_DIR, name)
    return os.path.isfile(p)


def _normalize_model_key(s: str) -> str:
    return str(s or "").strip().lower().replace(" ", "_")


def _is_valid_search_path(path: str) -> bool:
    # Expected: make/model-slug (letters, numbers, hyphens)
    return bool(re.fullmatch(r"[a-z0-9\-]+/[a-z0-9\-]+", (path or "").strip().lower()))


def _is_valid_cnb_url(url: str) -> bool:
    try:
        p = urlparse((url or "").strip())
    except Exception:
        return False
    if p.scheme not in {"http", "https"}:
        return False
    if (p.netloc or "").lower() not in {"carsandbids.com", "www.carsandbids.com"}:
        return False
    return (p.path or "").startswith("/auctions/")

@app.get("/api/models")
def get_models():
    models: List[Dict[str, Any]] = []
    if os.path.exists(MODEL_DIR):
        for f in os.listdir(MODEL_DIR):
            if f.endswith(".joblib") and f.startswith("model_"):
                model_id = f.replace("model_", "").replace(".joblib", "")
                models.append(
                    {"id": model_id, "name": model_id.replace("_", " ").title()}
                )

    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    summary = _load_json(summary_path)
    per_model = summary.get("per_model") or {}
    normalized_per_model = {
        _normalize_model_key(k): v for k, v in per_model.items()
    }

    for m in models:
        mid = m["id"]
        mid_key = _normalize_model_key(mid)
        m["metrics"] = normalized_per_model.get(mid_key)

    general_charts = [
        {"id": "accuracy_line", "title": "Actual vs predicted (all models)", "url": "/visualizations/accuracy_line_graph.png"},
        {"id": "actual_vs_pred", "title": "Scatter: actual vs predicted", "url": "/visualizations/actual_vs_predicted.png"},
        {"id": "residuals", "title": "Residuals", "url": "/visualizations/residual_plot.png"},
        {"id": "price_box", "title": "Price distribution by model", "url": "/visualizations/price_distribution_by_model.png"},
    ]
    charts = [c for c in general_charts if _viz_exists(os.path.basename(c["url"]))]

    return {"models": models, "summary": summary, "charts": charts}


@app.get("/api/models/{model_id}")
def get_model_details(model_id: str):
    safe_id = model_id.replace(" ", "_")
    feature_imp_path = os.path.join(
        RESULTS_DIR, f"feature_importances_{safe_id}.csv"
    )
    importances: List[Dict[str, Any]] = []
    if os.path.exists(feature_imp_path):
        df = pd.read_csv(feature_imp_path)
        df = df[df["Importance"] >= 0.001]
        importances = df.to_dict(orient="records")

    chart_file = f"accuracy_line_{safe_id}.png"
    chart_url = (
        f"/visualizations/{chart_file}" if _viz_exists(chart_file) else None
    )

    return {
        "model_id": safe_id,
        "importances": importances[:50],
        "accuracy_chart_url": chart_url,
    }


@app.get("/api/train/status")
def train_status():
    return get_training_state()


@app.post("/api/train")
def trigger_train(req: TrainRequest, background_tasks: BackgroundTasks):
    if not _is_valid_search_path(req.search_path):
        raise HTTPException(
            status_code=422,
            detail="Invalid search path format. Use make/model-slug, e.g. bmw/e46-m3",
        )
    st = get_training_state()
    if st.get("status") == "running":
        raise HTTPException(status_code=409, detail="A training job is already running.")
    background_tasks.add_task(run_training_pipeline, req.search_path)
    return {"message": "Training started.", "status": "running"}

@app.post("/api/train/full")
def trigger_full_retrain(background_tasks: BackgroundTasks):
    st = get_training_state()
    if st.get("status") == "running":
        raise HTTPException(status_code=409, detail="A training job is already running.")
    background_tasks.add_task(run_full_retrain)
    return {"message": "Full retrain started.", "status": "running"}

@app.delete("/api/models/{model_id}")
def delete_model(model_id: str):
    safe_id = model_id.replace(" ", "_")
    safe_key = _normalize_model_key(safe_id)

    removed: List[str] = []
    def _try_remove(path: str) -> None:
        if os.path.exists(path):
            os.remove(path)
            removed.append(path)

    _try_remove(os.path.join(MODEL_DIR, f"model_{safe_id}.joblib"))
    _try_remove(os.path.join(RESULTS_DIR, f"feature_importances_{safe_id}.csv"))
    _try_remove(os.path.join(RESULTS_DIR, f"test_predictions_{safe_id}.csv"))
    _try_remove(os.path.join(VIS_DIR, f"accuracy_line_{safe_id}.png"))

    # Remove from DB so full retrain cannot re-create this model.
    deleted_auctions = 0
    deleted_models = 0
    try:
        engine = get_engine()
        with engine.begin() as conn:
            rows = conn.execute(
                text(
                    """
                    SELECT id
                    FROM models
                    WHERE lower(replace(name, ' ', '_')) = :k
                    """
                ),
                {"k": safe_key},
            ).fetchall()
            model_ids = [r[0] for r in rows]
            if model_ids:
                for mid in model_ids:
                    deleted_auctions += conn.execute(
                        text("DELETE FROM auctions WHERE model_id = :mid"),
                        {"mid": mid},
                    ).rowcount or 0
                    deleted_models += conn.execute(
                        text("DELETE FROM models WHERE id = :mid"),
                        {"mid": mid},
                    ).rowcount or 0
    except Exception:
        # Keep endpoint resilient even if DB is unavailable.
        pass

    # Remove model rows from processed dataset so local data aligns with DB deletion.
    if os.path.exists(PROCESSED_PATH):
        try:
            df = pd.read_csv(PROCESSED_PATH)
            if "model" in df.columns:
                keep = df["model"].apply(lambda x: _normalize_model_key(_model_slug(x)) != safe_key)
                df_filtered = df[keep].copy()
                if len(df_filtered) != len(df):
                    df_filtered.to_csv(PROCESSED_PATH, index=False)
        except Exception:
            pass

    # Remove from summary per_model
    summary_path = os.path.join(RESULTS_DIR, "experiment_summary.json")
    if os.path.exists(summary_path):
        summary = _load_json(summary_path)
        pm = dict(summary.get("per_model") or {})
        removed_keys = []
        for k in list(pm.keys()):
            if _normalize_model_key(k) == safe_key:
                removed_keys.append(k)
        for k in removed_keys:
            pm.pop(k, None)
        if removed_keys:
            summary["per_model"] = pm
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=4)

    # Remove from test_predictions.csv so aggregate charts don't show the model
    agg_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if os.path.exists(agg_path):
        try:
            df_agg = pd.read_csv(agg_path)
            if "model" in df_agg.columns:
                keep_agg = df_agg["model"].apply(lambda x: _normalize_model_key(_model_slug(x)) != safe_key)
                df_agg_filtered = df_agg[keep_agg].copy()
                if len(df_agg_filtered) != len(df_agg):
                    df_agg_filtered.to_csv(agg_path, index=False)
                    # Sync visualizations immediately so the deleted model disappears from charts
                    generate_all_visualizations()
        except Exception:
            pass

    return {
        "deleted": safe_id,
        "removed_files": removed,
        "deleted_auctions": deleted_auctions,
        "deleted_models": deleted_models,
    }


def _format_prediction_details(raw: Dict[str, Any]) -> Dict[str, Any]:
    if not raw:
        return {}
    keys = [
        ("year", "Year"),
        ("make", "Make"),
        ("model", "Model"),
        ("mileage", "Mileage"),
        ("title_status", "Title status"),
        ("location", "Location"),
        ("engine", "Engine"),
        ("drivetrain", "Drivetrain"),
        ("transmission", "Transmission"),
        ("body_style", "Body style"),
        ("exterior_color", "Exterior color"),
        ("interior_color", "Interior color"),
        ("num_modifications", "Modifications count"),
        ("url", "Listing URL"),
    ]
    out = []
    for k, label in keys:
        v = raw.get(k)
        if v is None or v == "":
            continue
        out.append({"key": k, "label": label, "value": str(v)})
    return {"fields": out}


@app.post("/api/predict")
def predict_auction(req: PredictRequest):
    if not _is_valid_cnb_url(req.url):
        raise HTTPException(
            status_code=422,
            detail="Invalid URL. Use a Cars & Bids auction URL like https://carsandbids.com/auctions/...",
        )
    details = scrape_url(req.url)
    if not details or not details.get("make"):
        raise HTTPException(
            status_code=400,
            detail="Failed to scrape details or invalid auction page.",
        )

    df_raw = pd.DataFrame([details])
    df_raw["sale_price"] = np.nan
    df_clean = clean_dataset(df_raw, drop_rebuilt=False)

    model_name_extracted = df_clean.iloc[0]["model"]
    model_path = os.path.join(
        MODEL_DIR, f"model_{model_name_extracted.replace(' ', '_')}.joblib"
    )

    if not os.path.exists(model_path):
        raise HTTPException(
            status_code=404,
            detail=f"No model found for '{model_name_extracted}'. Please train it first.",
        )

    model_to_use = joblib.load(model_path)
    X = df_clean.drop(
        columns=["sale_price", "url", "date", "location", "make", "model"],
        errors="ignore",
    )

    prediction = float(model_to_use.predict(X)[0])

    return {
        "prediction": prediction,
        "details": details,
        "details_display": _format_prediction_details(details),
        "clean_features": df_clean.drop(
            columns=["sale_price", "url", "date", "location", "make", "model"],
            errors="ignore",
        ).to_dict(orient="records")[0],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
