import os
import re

import matplotlib

matplotlib.use("Agg")  # required for FastAPI background threads (macOS)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

RESULTS_DIR = os.path.join("src", "model", "results")
VIS_DIR = os.path.join("visualizations")

os.makedirs(VIS_DIR, exist_ok=True)

plt.style.use("dark_background")
sns.set_palette("viridis")


def _model_slug(name: str) -> str:
    s = str(name).replace(" ", "_").replace("/", "_")
    return re.sub(r"[^a-zA-Z0-9_\-]", "", s)


def _load_predictions_df(path=None) -> pd.DataFrame | None:
    """Loads predictions CSV"""
    p = path or os.path.join(RESULTS_DIR, "test_predictions.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    return None if df.empty else df

def _get_model_colors():
    """Returns dictionary mapping model names to colors"""
    df = _load_predictions_df()
    if df is None or "model" not in df.columns:
        return {}
        
    models = sorted(df["model"].dropna().unique())
    # Use Set2 for up to 8 colors, then fallback to husl for more
    if len(models) <= 8:
        palette = sns.color_palette("Set2", n_colors=len(models))
    else:
        palette = sns.color_palette("husl", n_colors=len(models))
    
    return dict(zip(models, palette))


def plot_accuracy_line_graph():
    """Plots actual vs predicted prices sorted by actual price"""
    # Load predictions dataset and exit if empty
    df = _load_predictions_df()
    if df is None:
        return

    # Sort test samples by actual price to create a smooth baseline curve
    df_sorted = df.sort_values(by="Actual").reset_index(drop=True)

    plt.figure(figsize=(14, 7))
    # Plot the actual prices as a solid white line
    plt.plot(
        df_sorted.index,
        df_sorted["Actual"],
        label="Actual Sale Price",
        color="white",
        linewidth=2,
        zorder=1,
    )
    # Overlay predicted prices as colored scatter points
    sns.scatterplot(
        x=df_sorted.index,
        y=df_sorted["Predicted"],
        hue=df_sorted["model"],
        palette=_get_model_colors(),
        s=60,
        alpha=0.9,
        zorder=2,
    )

    plt.title("Model Accuracy: Actual vs Predicted Prices (Sorted)", fontsize=16)
    plt.xlabel("Test Samples (Sorted by Actual Price)", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        os.path.join(VIS_DIR, "accuracy_line_graph.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_aggregate_actual_vs_predicted():
    """Creates scatter plot of actual vs predicted prices"""
    # Load predictions dataset and exit if empty
    df = _load_predictions_df()
    if df is None:
        return

    plt.figure(figsize=(9, 9))
    # Scatter predicted vs actual, colored by model
    sns.scatterplot(
        x="Actual",
        y="Predicted",
        hue="model",
        data=df,
        s=80,
        alpha=0.85,
        palette=_get_model_colors(),
    )

    # Draw a diagonal dashed line representing perfect predictions
    min_val = min(df["Actual"].min(), df["Predicted"].min())
    max_val = max(df["Actual"].max(), df["Predicted"].max())
    plt.plot([min_val, max_val], [min_val, max_val], color="white", linestyle="--", alpha=0.5)

    plt.title("Actual vs Predicted Auction Prices (All Models)", fontsize=16)
    plt.xlabel("Actual Sale Price ($)", fontsize=12)
    plt.ylabel("Predicted Sale Price ($)", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        os.path.join(VIS_DIR, "actual_vs_predicted.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_aggregate_residuals():
    """Plots residuals against predicted sale price"""
    # Load predictions dataset and exit if empty
    df = _load_predictions_df()
    if df is None:
        return

    # Calculate residual (error) for each prediction
    df = df.copy()
    df["Residual"] = df["Actual"] - df["Predicted"]

    plt.figure(figsize=(11, 6))
    sns.scatterplot(
        x="Predicted",
        y="Residual",
        hue="model",
        data=df,
        s=80,
        alpha=0.85,
        palette=_get_model_colors(),
    )
    plt.axhline(0, color="white", linestyle="--", alpha=0.5)

    plt.title("Residual Analysis (All Models)", fontsize=16)
    plt.xlabel("Predicted Sale Price ($)", fontsize=12)
    plt.ylabel("Residual (Actual − Predicted) ($)", fontsize=12)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        os.path.join(VIS_DIR, "residual_plot.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_price_distribution_by_model():
    """Creates boxplot of actual sale prices by model"""
    # Load predictions dataset and exit if empty
    df = _load_predictions_df()
    if df is None or "model" not in df.columns:
        return

    plt.figure(figsize=(12, 6))
    # Sort models alphabetically and plot price distribution
    order = sorted(df["model"].dropna().unique(), key=str)
    sns.boxplot(data=df, x="model", y="Actual", order=order, palette=_get_model_colors(), hue="model", legend=False)
    plt.xticks(rotation=25, ha="right")
    plt.title("Test-Set Actual Sale Prices by Model", fontsize=16)
    plt.xlabel("Model", fontsize=12)
    plt.ylabel("Actual Sale Price ($)", fontsize=12)
    plt.grid(True, alpha=0.2, axis="y")
    plt.tight_layout()

    plt.savefig(
        os.path.join(VIS_DIR, "price_distribution_by_model.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_feature_heatmap():
    """Creates a correlation heatmap of numeric features"""
    df = _load_predictions_df()
    if df is None:
        return

    # Select only numeric columns for correlation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude target and prediction columns to focus only on feature interactions
    exclude_cols = ['Actual', 'Predicted', 'sale_price']
    features_only = [c for c in numeric_cols if c not in exclude_cols]
    
    if not features_only:
        return

    corr = df[features_only].corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr, 
        annot=True, 
        fmt=".2f", 
        cmap="RdBu_r", 
        center=0,
        linewidths=.5,
        cbar_kws={"shrink": .8}
    )
    
    plt.title("Feature Correlation Heatmap", fontsize=16)
    plt.tight_layout()

    plt.savefig(
        os.path.join(VIS_DIR, "feature_heatmap.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_model_accuracy_line(model_slug: str) -> None:
    """Plots model accuracy line from per-model or aggregate predictions"""
    # Determine correct file path: use per-model CSV if available, otherwise fallback to aggregate
    model_slug = _model_slug(model_slug)
    per_path = os.path.join(RESULTS_DIR, f"test_predictions_{model_slug}.csv")
    agg_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    pred_path = per_path if os.path.exists(per_path) else agg_path
    
    # Load the selected predictions file
    df = _load_predictions_df(pred_path)
    if df is None:
        return

    # Filter to specific model if we fell back to the aggregate file
    if "model" in df.columns and pred_path == agg_path:
        df = df[df["model"].apply(_model_slug) == model_slug]
        if df.empty:
            return

    # Sort actual prices to form a smooth curve
    df_sorted = df.sort_values(by="Actual").reset_index(drop=True)
    if len(df_sorted) < 2:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(
        df_sorted.index,
        df_sorted["Actual"],
        label="Actual",
        color="white",
        linewidth=2,
        zorder=1,
    )
    # Determine model color from global mapping
    model_name = df_sorted["model"].iloc[0] if "model" in df_sorted.columns else None
    model_colors = _get_model_colors()
    pred_color = model_colors.get(model_name, "#38bdf8")

    plt.scatter(
        df_sorted.index,
        df_sorted["Predicted"],
        label="Predicted",
        c=[pred_color],
        s=55,
        alpha=0.9,
        zorder=2,
    )

    title_name = df_sorted["model"].iloc[0] if "model" in df_sorted.columns else model_slug
    plt.title(f"Accuracy: {title_name} (Test Set, Sorted by Actual)", fontsize=15)
    plt.xlabel("Test Samples (Sorted by Actual Price)", fontsize=12)
    plt.ylabel("Price ($)", fontsize=12)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.2)
    plt.tight_layout()

    plt.savefig(
        os.path.join(VIS_DIR, f"accuracy_line_{model_slug}.png"),
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()


def plot_per_model_accuracy_lines(only_slugs: list[str] | None = None):
    """Generates accuracy line plots for each model"""
    if only_slugs:
        for s in only_slugs:
            plot_model_accuracy_line(s)
        return

    df = _load_predictions_df()
    if df is None or "model" not in df.columns:
        return

    for model_name in df["model"].dropna().unique():
        plot_model_accuracy_line(_model_slug(model_name))


def generate_all_visualizations():
    """Regenerates all visualizations"""
    plot_accuracy_line_graph()
    plot_aggregate_actual_vs_predicted()
    plot_aggregate_residuals()
    plot_price_distribution_by_model()
    plot_feature_heatmap()
    plot_per_model_accuracy_lines()


if __name__ == "__main__":
    print("Generating visualizations...")
    generate_all_visualizations()
    print(f"Visualizations saved to {VIS_DIR}")
