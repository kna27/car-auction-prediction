import os
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

load_dotenv()

# Database connection parameters
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "car_auctions")

PROCESSED_DATA_PATH = os.path.join("data", "processed", "all_vehicles_cleaned.csv")
SCHEMA_PATH = os.path.join("src", "data", "schema.sql")

def get_engine():
    """
    Creates and returns a SQLAlchemy engine using environment variables.
    Handles connections both with and without a password.
    """
    if DB_PASSWORD:
        connection_string = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    else:
        connection_string = f"postgresql+psycopg2://{DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_string)

def setup_database(engine):
    """
    Initializes the PostgreSQL database schema by running the schema.sql script.
    Creates necessary tables (e.g., auctions, models, makes) if they do not exist.
    """
    with engine.connect() as conn:
        with open(SCHEMA_PATH, 'r') as f:
            sql_script = f.read()
        conn.execute(text(sql_script))
        conn.commit()
        print("Database schema initialized.")

def get_or_create(conn, table, column, value, extra_cols=None, extra_vals=None):
    """
    Helper function to efficiently query for a normalized relational table ID.
    If the value does not exist, it inserts it and returns the newly generated ID.
    """
    if not extra_cols:
        extra_cols = []
    if not extra_vals:
        extra_vals = []
        
    query = text(f"SELECT id FROM {table} WHERE {column} = :val")
    result = conn.execute(query, {"val": value}).fetchone()
    
    if result:
        return result[0]
    else:
        cols = [column] + extra_cols
        vals = [value] + extra_vals
        placeholders = [":val"] + [f":val{i}" for i in range(len(extra_vals))]
        
        insert_query = text(f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({', '.join(placeholders)}) RETURNING id")
        
        params = {"val": value}
        for i, v in enumerate(extra_vals):
            params[f"val{i}"] = v
            
        new_id = conn.execute(insert_query, params).fetchone()[0]
        return new_id

def get_existing_auction_urls(conn) -> set:
    """
    Retrieves a set of all URLs currently in the 'auctions' table.
    Used for rapid deduplication before inserting new rows.
    """
    r = conn.execute(text("SELECT url FROM auctions WHERE url IS NOT NULL"))
    return {row[0] for row in r.fetchall()}


def load_data(engine, df: Optional[pd.DataFrame] = None):
    if df is None:
        if not os.path.exists(PROCESSED_DATA_PATH):
            print(f"Processed data file not found: {PROCESSED_DATA_PATH}")
            return
        df = pd.read_csv(PROCESSED_DATA_PATH)

    if df.empty:
        print("No rows to load.")
        return

    with engine.connect() as conn:
        existing = get_existing_auction_urls(conn)
        inserted = 0
        skipped = 0
        for _, row in df.iterrows():
            row_url = row['url'] if pd.notna(row.get('url')) else None
            if not row_url:
                skipped += 1
                continue
            if row_url in existing:
                skipped += 1
                continue

            # 1. Make
            make_id = get_or_create(conn, "makes", "name", row['make'])
            
            # 2. Model
            model_id = get_or_create(conn, "models", "name", row['model'], extra_cols=["make_id"], extra_vals=[make_id])
            
            # 3. Transmission
            trans_id = get_or_create(conn, "transmissions", "name", row['transmission'])
            
            # 4. Colors
            ext_color_id = get_or_create(conn, "colors", "name", row['exterior_color'])
            int_color_id = get_or_create(conn, "colors", "name", row['interior_color'])
            
            # 5. Auction Insert
            # Handle potential NaN in date
            auction_date = row['date'] if pd.notna(row['date']) else None

            auction_query = text("""
                INSERT INTO auctions 
                (year, model_id, transmission_id, exterior_color_id, interior_color_id, 
                 mileage, title_status, location, state, engine, drivetrain, body_style, 
                 num_modifications, sale_price, auction_date, has_forced_induction, url)
                VALUES 
                (:year, :model_id, :trans_id, :ext_col_id, :int_col_id, 
                 :mileage, :title_status, :location, :state, :engine, :drivetrain, :body_style, 
                 :num_modifications, :sale_price, :auction_date, :has_forced_induction, :url)
                ON CONFLICT (url) DO NOTHING
            """)

            result = conn.execute(auction_query, {
                "year": row['year'] if pd.notna(row.get('year')) else None,
                "model_id": model_id,
                "trans_id": trans_id,
                "ext_col_id": ext_color_id,
                "int_col_id": int_color_id,
                "mileage": row['mileage'] if pd.notna(row['mileage']) else None,
                "title_status": row['title_status'] if pd.notna(row['title_status']) else None,
                "location": row['location'] if pd.notna(row['location']) else None,
                "state": row.get('state', None),
                "engine": row['engine'] if pd.notna(row['engine']) else None,
                "drivetrain": row['drivetrain'] if pd.notna(row['drivetrain']) else None,
                "body_style": row['body_style'] if pd.notna(row['body_style']) else None,
                "num_modifications": row['num_modifications'] if pd.notna(row['num_modifications']) else 0,
                "sale_price": row['sale_price'] if pd.notna(row['sale_price']) else None,
                "auction_date": auction_date,
                "has_forced_induction": row.get('has_forced_induction', 0),
                "url": row_url
            })
            if result.rowcount:
                inserted += 1
                if row_url:
                    existing.add(row_url)

        conn.commit()
        print(f"Data loaded: {inserted} new row(s), {skipped} already in database.")

def main():
    engine = get_engine()
    
    # Check if we can connect
    try:
        with engine.connect() as conn:
            pass
    except Exception as e:
        print(f"Could not connect to database. Please check your PostgreSQL server and credentials.\n{e}")
        return

    setup_database(engine)
    load_data(engine)

if __name__ == "__main__":
    main()
