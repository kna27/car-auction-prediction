import glob
import os

import numpy as np
import pandas as pd

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")
ALL_VEHICLES_CLEANED = os.path.join(PROCESSED_DATA_DIR, "all_vehicles_cleaned.csv")

def clean_mileage(mileage_str):
    """
    Extracts numerical mileage from raw string ("54,000 miles" -> 54000)
    """
    if pd.isna(mileage_str):
        return np.nan
    # Extract only digits
    digits = ''.join(filter(str.isdigit, str(mileage_str)))
    return int(digits) if digits else np.nan

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw auction dataset by parsing dates, imputing missing values, 
    and extracting features
    """
    # Drop Rebuilt or Salvage titles as they skew prices
    df = df[~df['title_status'].str.contains("Rebuilt|Salvage", case=False, na=False)].copy()
    # Clean the " Save" artifact from model name
    df['model'] = df['model'].str.replace(" Save", "", regex=False)

    # Clean Mileage
    df['mileage'] = df['mileage'].apply(clean_mileage)
    df['mileage'] = df['mileage'].fillna(df['mileage'].median())

    # Clean Year
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['year'] = df['year'].fillna(df['year'].median()).astype(int)

    # Clean Sale Price
    df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
    df['sale_price'] = df['sale_price'].fillna(df['sale_price'].median())

    # Clean Num Modifications
    df['num_modifications'] = pd.to_numeric(df['num_modifications'], errors='coerce')
    df['num_modifications'] = df['num_modifications'].fillna(0) # Default to 0

    # Handle Categorical Columns
    categorical_cols = [
        'make', 'model', 'title_status', 'location', 'engine', 
        'drivetrain', 'transmission', 'body_style', 'exterior_color', 'interior_color'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            # Impute missing with mode
            mode_val = df[col].mode()[0] if not df[col].mode().empty else "Unknown"
            df[col] = df[col].fillna(mode_val)
            df[col] = df[col].astype(str).str.strip()
            
    if 'image_url' in df.columns:
        df['image_url'] = df['image_url'].astype(str).str.strip()

    # Apply specific casing and spacing fixes
    if 'transmission' in df.columns:
        df['transmission'] = df['transmission'].str.title()
        df['transmission'] = df['transmission'].str.replace(" )", ")", regex=False)
        df['transmission'] = df['transmission'].str.replace(" Manual)", ")", regex=False)

    # Feature Engineering: Forced Induction
    if 'engine' in df.columns:
        df['has_forced_induction'] = df['engine'].str.contains("Supercharged|Turbocharged|Twin Turbo|Turbo", case=False).astype(int)
        # Normalize engine name while keeping forced induction details for specific trim categorization
        df['engine'] = df['engine'].str.replace("I-4", "I4", regex=False)
        df['engine'] = df['engine'].str.replace("Inline-4", "I4", regex=False)
        df['engine'] = df['engine'].str.replace(" Flat ", " Flat-", regex=False)
        df['engine'] = df['engine'].str.replace(" H6", " Flat-6", regex=False)
        df['engine'] = df['engine'].str.replace("Flat-Six", "Flat-6", case=False, regex=False)
        df['engine'] = df['engine'].str.replace(r"(\d\.\d) Flat-6", r"\1L Flat-6", regex=True)

    # Clean Title Status (remove state info)
    if 'title_status' in df.columns:
        df['title_status'] = df['title_status'].str.split('(').str[0].str.strip()

    # Extract State from Location
    def extract_state(loc):
        parts = loc.split(',')
        if len(parts) >= 2:
            # State is the first part after city
            state_part = parts[1].strip().split(' ')[0]
            return state_part
        return loc
        
    if 'location' in df.columns:
        df['state'] = df['location'].apply(extract_state).str.upper()

    return df


def merge_into_all_vehicles_cleaned(df_new: pd.DataFrame) -> pd.DataFrame:
    """Append new cleaned rows to main CSV file, removing duplicates by auction URL"""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    if os.path.exists(ALL_VEHICLES_CLEANED):
        df_existing = pd.read_csv(ALL_VEHICLES_CLEANED)
        combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        combined = df_new
    if "url" in combined.columns:
        combined = combined.drop_duplicates(subset=["url"], keep="last")
    combined.to_csv(ALL_VEHICLES_CLEANED, index=False)
    return combined


def main():
    """Iterates through all raw CSV files, applies dataset cleaning, and combines them"""
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    raw_files = glob.glob(os.path.join(RAW_DATA_DIR, "*_raw.csv"))
    
    if not raw_files:
        print(f"No raw data files found in {RAW_DATA_DIR}")
        return

    all_data = []
    for file in raw_files:
        print(f"Processing {file}...")
        df = pd.read_csv(file)
        cleaned_df = clean_dataset(df)
        
        # Save individually
        base_name = os.path.basename(file).replace("_raw.csv", "_cleaned.csv")
        out_path = os.path.join(PROCESSED_DATA_DIR, base_name)
        cleaned_df.to_csv(out_path, index=False)
        print(f"Saved {out_path}")
        
        all_data.append(cleaned_df)
        
    # Combine into one CSV file
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        master_df.to_csv(ALL_VEHICLES_CLEANED, index=False)
        print(f"Saved combined dataset to {ALL_VEHICLES_CLEANED}")

if __name__ == "__main__":
    main()
