import os
import glob
import pandas as pd
import numpy as np

RAW_DATA_DIR = os.path.join("data", "raw")
PROCESSED_DATA_DIR = os.path.join("data", "processed")

def clean_mileage(mileage_str):
    if pd.isna(mileage_str):
        return np.nan
    # Extract only digits
    digits = ''.join(filter(str.isdigit, str(mileage_str)))
    return int(digits) if digits else np.nan

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # 0. Initial Filtering (Outliers / Rebuilt Titles)
    # Drop Rebuilt or Salvage titles as they skew prices
    df = df[~df['title_status'].str.contains("Rebuilt|Salvage", case=False, na=False)].copy()

    # Clean the " Save" artifact from model name
    df['model'] = df['model'].str.replace(" Save", "", regex=False)

    # 1. Clean Mileage
    df['mileage'] = df['mileage'].apply(clean_mileage)
    df['mileage'] = df['mileage'].fillna(df['mileage'].median())

    # 1b. Clean Year
    if 'year' in df.columns:
        df['year'] = pd.to_numeric(df['year'], errors='coerce')
        df['year'] = df['year'].fillna(df['year'].median()).astype(int)

    # 2. Clean Sale Price
    df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
    df['sale_price'] = df['sale_price'].fillna(df['sale_price'].median())

    # 3. Clean Num Modifications
    df['num_modifications'] = pd.to_numeric(df['num_modifications'], errors='coerce')
    df['num_modifications'] = df['num_modifications'].fillna(0) # Default to 0

    # 4. Handle Categorical Columns
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

    # Apply specific casing fixes
    if 'transmission' in df.columns:
        df['transmission'] = df['transmission'].str.title()

    # 4b. Feature Engineering: Forced Induction
    if 'engine' in df.columns:
        df['has_forced_induction'] = df['engine'].str.contains("Supercharged|Turbocharged", case=False).astype(int)
        # Normalize engine name
        df['engine'] = df['engine'].str.replace("Supercharged ", "", case=False, regex=False)
        df['engine'] = df['engine'].str.replace("Turbocharged ", "", case=False, regex=False)
        df['engine'] = df['engine'].str.replace("I-4", "I4", regex=False)

    # 4c. Clean Title Status (remove state info)
    if 'title_status' in df.columns:
        df['title_status'] = df['title_status'].str.split('(').str[0].str.strip()

    # 5. Extract State from Location (assuming format "City, State, Zip" or "City, State")
    def extract_state(loc):
        parts = loc.split(',')
        if len(parts) >= 2:
            # State is usually the first part after city
            state_part = parts[1].strip().split(' ')[0]
            return state_part
        return loc
        
    if 'location' in df.columns:
        df['state'] = df['location'].apply(extract_state).str.upper()

    return df

def main():
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
        
    # Combine into one master dataset
    if all_data:
        master_df = pd.concat(all_data, ignore_index=True)
        master_out = os.path.join(PROCESSED_DATA_DIR, "all_vehicles_cleaned.csv")
        master_df.to_csv(master_out, index=False)
        print(f"Saved combined dataset to {master_out}")

if __name__ == "__main__":
    main()
