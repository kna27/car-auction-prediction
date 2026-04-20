import os
import re

import matplotlib

matplotlib.use("Agg")  # required for FastAPI background threads (macOS)
import matplotlib.pyplot as plt
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


def plot_accuracy_line_graph():
    pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if not os.path.exists(pred_path):
        return

    df = pd.read_csv(pred_path)
    df_sorted = df.sort_values(by="Actual").reset_index(drop=True)

    plt.figure(figsize=(14, 7))
    plt.plot(
        df_sorted.index,
        df_sorted["Actual"],
        label="Actual Sale Price",
        color="white",
        linewidth=2,
        zorder=1,
    )
    sns.scatterplot(
        x=df_sorted.index,
        y=df_sorted["Predicted"],
        hue=df_sorted["model"],
        palette="Set2",
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
    pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if not os.path.exists(pred_path):
        return

    df = pd.read_csv(pred_path)
    if df.empty:
        return

    plt.figure(figsize=(9, 9))
    sns.scatterplot(
        x="Actual",
        y="Predicted",
        hue="model",
        data=df,
        s=80,
        alpha=0.85,
        palette="Set2",
    )

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
    pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if not os.path.exists(pred_path):
        return

    df = pd.read_csv(pred_path)
    if df.empty:
        return

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
        palette="Set2",
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
    pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if not os.path.exists(pred_path):
        return

    df = pd.read_csv(pred_path)
    if df.empty or "model" not in df.columns:
        return

    plt.figure(figsize=(12, 6))
    order = sorted(df["model"].dropna().unique(), key=str)
    sns.boxplot(data=df, x="model", y="Actual", order=order, palette="Set2", hue="model", legend=False)
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


def plot_model_accuracy_line(model_slug: str) -> None:
    """
    Prefer per-model predictions file if present:
    - src/model/results/test_predictions_<slug>.csv
    Fallback to the aggregate test_predictions.csv.
    """
    model_slug = _model_slug(model_slug)
    per_path = os.path.join(RESULTS_DIR, f"test_predictions_{model_slug}.csv")
    agg_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    pred_path = per_path if os.path.exists(per_path) else agg_path
    if not os.path.exists(pred_path):
        return

    df = pd.read_csv(pred_path)
    if df.empty:
        return

    if "model" in df.columns and pred_path == agg_path:
        # Filter to the specific model when using the aggregate file.
        df = df[df["model"].apply(_model_slug) == model_slug]
        if df.empty:
            return

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
    plt.scatter(
        df_sorted.index,
        df_sorted["Predicted"],
        label="Predicted",
        c="#38bdf8",
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
    if only_slugs:
        for s in only_slugs:
            plot_model_accuracy_line(s)
        return

    pred_path = os.path.join(RESULTS_DIR, "test_predictions.csv")
    if not os.path.exists(pred_path):
        return

    df = pd.read_csv(pred_path)
    if df.empty or "model" not in df.columns:
        return

    for model_name in df["model"].dropna().unique():
        plot_model_accuracy_line(_model_slug(model_name))


def generate_all_visualizations():
    """Regenerate charts from the latest training run (no feature-importance PNGs)."""
    plot_accuracy_line_graph()
    plot_aggregate_actual_vs_predicted()
    plot_aggregate_residuals()
    plot_price_distribution_by_model()
    plot_per_model_accuracy_lines()


if __name__ == "__main__":
    print("Generating visualizations...")
    generate_all_visualizations()
    print(f"Visualizations saved to {VIS_DIR}")
