import copy
import json
import os
import sys
import traceback
from typing import Any, Dict, Optional

import pandas as pd

_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
sys.path.insert(0, _ROOT)

from src.data_cleaning.clean_data import clean_dataset, merge_into_all_vehicles_cleaned
from src.data_collection.scraper import (
    HEADERS,
    ChromeDriverManager,
    ChromeOptions,
    ChromeService,
    scrape_model,
    webdriver,
    write_csv,
)
from src.database.load_data import get_engine, load_data, setup_database
from src.eda.visualize import _model_slug
from src.model.train_model import RESULTS_DIR, train_and_evaluate

_TRAINING_STATE: Dict[str, Any] = {
    "status": "idle",
    "phase": "",
    "fetched": 0,
    "total": 0,
    "listing_page": 0,
    "message": "",
    "error": None,
}


def get_training_state() -> Dict[str, Any]:
    return copy.deepcopy(_TRAINING_STATE)


def _set_state(**kwargs: Any) -> None:
    _TRAINING_STATE.update(kwargs)


def run_training_pipeline(search_path: str) -> None:
    _set_state(
        status="running",
        phase="starting",
        message="Initializing…",
        error=None,
        fetched=0,
        total=0,
        listing_page=0,
    )

    driver: Optional[Any] = None

    try:
        chrome_options = ChromeOptions()
        chrome_options.add_argument("--headless=new")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        chrome_options.add_argument(f"user-agent={HEADERS['User-Agent']}")

        driver = webdriver.Chrome(
            service=ChromeService(ChromeDriverManager().install()),
            options=chrome_options,
        )

        full_path = f"/search/{search_path}" if not search_path.startswith("/") else search_path
        if not full_path.startswith("/search/"):
            full_path = f"/search/{search_path}"

        def on_progress(payload: Dict[str, Any]) -> None:
            ph = payload.get("phase")
            if ph == "listing":
                page = int(payload.get("page") or 0)
                _set_state(
                    phase="listing",
                    listing_page=page,
                    message=f"Scanning search results (page {page})…",
                )
            elif ph == "fetching":
                fetched = int(payload.get("fetched") or 0)
                total = int(payload.get("total") or 0)
                _set_state(
                    phase="fetching",
                    fetched=fetched,
                    total=total,
                    message=f"Fetched {fetched}/{total} auctions…",
                )

        print(f"Scraping path: {full_path}")
        rows = scrape_model(driver, full_path, progress_callback=on_progress)

        if len(rows) == 0:
            _set_state(
                status="complete",
                phase="done",
                message="No auctions found for this search.",
            )
            return

        slug = search_path.replace("/", "_").replace("-", "_")
        output_file = os.path.join(_ROOT, "data", "raw", f"{slug}_raw.csv")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        write_csv(rows, output_file)

        _set_state(phase="cleaning", message="Cleaning data…", fetched=len(rows), total=len(rows))
        df_raw = pd.DataFrame(rows)
        df_clean = clean_dataset(df_raw, drop_rebuilt=True)
        merge_into_all_vehicles_cleaned(df_clean)

        _set_state(phase="loading_db", message="Loading new rows into database…")
        engine = get_engine()
        try:
            with engine.connect():
                pass
        except Exception as conn_err:
            _set_state(
                status="error",
                phase="error",
                error=str(conn_err),
                message="Database connection failed.",
            )
            print(conn_err)
            return

        setup_database(engine)
        load_data(engine, df=df_clean)

        _set_state(
            phase="training",
            message="Training new model(s) and generating charts (this may take several minutes)…",
        )
        models_to_train = sorted(set(map(str, df_clean["model"].dropna().unique().tolist())))
        train_and_evaluate(only_models=models_to_train)

        _set_state(
            status="complete",
            phase="done",
            message="Training and visualizations complete.",
        )
    except Exception as e:
        traceback.print_exc()
        _set_state(
            status="error",
            phase="error",
            error=str(e),
            message=str(e),
        )
    finally:
        if driver is not None:
            driver.quit()


def run_full_retrain() -> None:
    """
    Full retrain using existing DB rows (no scraping). Regenerates all charts.
    """
    _set_state(
        status="running",
        phase="training",
        message="Full retrain: training all models and regenerating charts…",
        error=None,
        fetched=0,
        total=0,
        listing_page=0,
    )
    try:
        train_and_evaluate(only_models=None)
        _set_state(
            status="complete",
            phase="done",
            message="Full retrain complete.",
        )
    except Exception as e:
        traceback.print_exc()
        _set_state(
            status="error",
            phase="error",
            error=str(e),
            message=str(e),
        )
