import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob

RESULTS_DIR = os.path.join("src", "model", "results")
VIS_DIR = os.path.join("visualizations")

os.makedirs(VIS_DIR, exist_ok=True)

plt.style.use('dark_background')
sns.set_palette("viridis")

def plot_feature_importances():
    importance_files = glob.glob(os.path.join(RESULTS_DIR, "*feature_importances*.csv"))
    
    for file in importance_files:
        df = pd.read_csv(file)
        # Take top 10 features for readability
        df = df.head(10)
        
        # Clean up feature names (e.g. cat__model_ND Miata -> model_ND Miata)
        df['Feature'] = df['Feature'].str.replace('cat__', '').str.replace('num__', '')
        
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Importance', y='Feature', data=df, hue='Feature', palette='magma', legend=False)
        
        title_name = os.path.basename(file).replace('.csv', '').replace('_', ' ').title()
        plt.title(f"{title_name} - Top 10 Features", fontsize=16)
        plt.xlabel("Random Forest Feature Importance", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.tight_layout()
        
        out_name = os.path.basename(file).replace('.csv', '.png')
        plt.savefig(os.path.join(VIS_DIR, out_name), dpi=300, bbox_inches='tight')
        plt.close()

def plot_actual_vs_predicted():
    pred_files = glob.glob(os.path.join(RESULTS_DIR, "test_predictions_*.csv"))
    
    for file in pred_files:
        df = pd.read_csv(file)
        
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x='Actual', y='Predicted', hue='model', data=df, s=100, alpha=0.8, palette='Set2')
        
        # Add perfect prediction line
        min_val = min(df['Actual'].min(), df['Predicted'].min())
        max_val = max(df['Actual'].max(), df['Predicted'].max())
        plt.plot([min_val, max_val], [min_val, max_val], color='white', linestyle='--', alpha=0.5)
        
        plt.title("Actual vs Predicted Auction Prices", fontsize=16)
        plt.xlabel("Actual Sale Price ($)", fontsize=12)
        plt.ylabel("Predicted Sale Price ($)", fontsize=12)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        plt.savefig(os.path.join(VIS_DIR, "actual_vs_predicted.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_residuals():
    pred_files = glob.glob(os.path.join(RESULTS_DIR, "test_predictions_*.csv"))
    
    for file in pred_files:
        df = pd.read_csv(file)
        df['Residual'] = df['Actual'] - df['Predicted']
        
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Predicted', y='Residual', hue='model', data=df, s=100, alpha=0.8, palette='Set2')
        plt.axhline(0, color='white', linestyle='--', alpha=0.5)
        
        plt.title("Residual Analysis", fontsize=16)
        plt.xlabel("Predicted Sale Price ($)", fontsize=12)
        plt.ylabel("Residual Error (Actual - Predicted) ($)", fontsize=12)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        
        plt.savefig(os.path.join(VIS_DIR, "residual_plot.png"), dpi=300, bbox_inches='tight')
        plt.close()

def plot_accuracy_line_graph():
    pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if not os.path.exists(pred_path):
        return
        
    df = pd.read_csv(pred_path)
    
    # Sort by actual price to make a smooth line graph
    df_sorted = df.sort_values(by='Actual').reset_index(drop=True)
    
    plt.figure(figsize=(14, 7))
    plt.plot(df_sorted.index, df_sorted['Actual'], label='Actual Sale Price', color='white', linewidth=2, zorder=1)
    
    # Scatter predicted prices as dots colored by car model
    sns.scatterplot(x=df_sorted.index, y=df_sorted['Predicted'], hue=df_sorted['model'], palette='Set2', s=60, alpha=0.9, zorder=2)
    
    plt.title("Model Accuracy: Actual vs Predicted Prices (Sorted)", fontsize=16)
    plt.xlabel("Test Samples (Sorted by Actual Price)", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    
    plt.savefig(os.path.join(VIS_DIR, "accuracy_line_graph.png"), dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Visualizations...")
    plot_feature_importances()
    plot_actual_vs_predicted()
    plot_residuals()
    plot_accuracy_line_graph()
    print(f"Visualizations saved to {VIS_DIR}")
