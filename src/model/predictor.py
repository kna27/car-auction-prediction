import os
import sys

import joblib
import numpy as np
import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.data.cleaner import clean_dataset
from src.data.scraper import parse_auction_page
from src.model.trainer import engineer_features

MODEL_DIR = os.path.join("src", "model", "saved_models")
RESULTS_DIR = os.path.join("src", "model", "results")
SUMMARY_FILE = os.path.join(RESULTS_DIR, "experiment_summary.json")

def scrape_url(url: str) -> dict:
    """Scrapes individual auction URL using headless Selenium"""
    # Configure Chrome options for headless execution
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option("useAutomationExtension", False)

    driver = webdriver.Chrome(options=chrome_options)
    
    try:
        print(f"Scraping {url}...")
        driver.get(url)
        
        try:
            # Wait for auction details section to be loaded in the DOM
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located(
                    (By.CSS_SELECTOR, "div.cnb-details-quick-facts")
                )
            )
        except TimeoutException:
            print("Warning: Timed out waiting for auction details to load. Attempting to parse anyway...")
            
        # Parse the loaded page source
        html = driver.page_source
        details = parse_auction_page(html, url)
        return details
    finally:
        driver.quit()

def predict_price(url: str):
    """Predicts final sale price for live auction URL"""
    details = scrape_url(url)
    if not details or not details.get("make"):
        print("Failed to scrape details or invalid auction page.")
        return

    print("\n--- Auction Details ---")
    for k, v in details.items():
        print(f"{k.capitalize()}: {v}")

    # Convert to DataFrame
    df_raw = pd.DataFrame([details])
    df_raw['sale_price'] = np.nan # Mock target column so clean_dataset doesn't fail
    
    # Clean it without dropping rebuilt cars
    df_clean = clean_dataset(df_raw, drop_rebuilt=False)
    df_clean = engineer_features(df_clean)
    
    # Determine the correct model to load based on the car's model type
    model_name_extracted = df_clean.iloc[0]['model']
    model_path = os.path.join(MODEL_DIR, f"model_{model_name_extracted.replace(' ', '_')}.joblib")
    
    if os.path.exists(model_path):
        print(f"Found Model for '{model_name_extracted}'! Loading...")
        model_to_use = joblib.load(model_path)
    else:
        print(f"Error: No Model found for '{model_name_extracted}'.")
        print(f"Please train a model for this car by running `python src/model/trainer.py`")
        return

    # Remove metadata features before passing to the model
    X = df_clean.drop(columns=['sale_price', 'url', 'image_url', 'date', 'make', 'model'], errors='ignore')
    
    try:
        # Run prediction and format output
        prediction = model_to_use.predict(X)[0]
        print("\n=========================================")
        print(f"PREDICTED SALE PRICE: ${prediction:,.2f}")
        print("=========================================")
        
        if df_clean.iloc[0]['title_status'] and "rebuilt" in str(df_clean.iloc[0]['title_status']).lower():
            print("This car has a Rebuilt/Salvage title. The model was primarily trained on Clean titles, so this prediction may be an overestimate.")
            
    except Exception as e:
        print(f"Error during prediction: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python src/model/predictor.py <auction_url>")
    else:
        predict_price(sys.argv[1])
