# Predicting Enthusiast Car Auction Final Sale Prices

CS210 Final Project

This project focuses on predicting the final sale price of enthusiast vehicles listed on Cars & Bids using web scraping, data cleaning, relational databases, and machine learning.

## 1. Problem Definition and Relevance

Unlike standard commuter cars where depreciation follows a relatively predictable curve, enthusiast car values are influenced by subjective factors such as transmission type, rare exterior/interior colors, and aftermarket modifications.

### Course Connection
This project connects to CS210 course concepts through:
- **Web Scraping:** Extracting unstructured data from web pages using BeautifulSoup and Selenium.
- **Data Cleaning:** Handling real-world datasets with inconsistent entries.
- **Data Management:** Designing a relational schema to ensure data integrity and efficient querying.
- **Data Provenance:** Maintaining a documented history of data transformations for reproducibility.

### Use Cases
By predicting sale prices within a narrow margin of error, this project can help
- **Buyers:** To determine if a current bid is undervalued or if they are overpaying.
- **Sellers:** To estimate the ROI on their vehicle and decide on reserve prices.

---

## 2. Novelty and Importance

### Gap in Current Tools
Traditional valuation tools like Kelley Blue Book (KBB) fail in the enthusiast market due to not supporting older cars (more than 21 years old), not weighing enthusiast-specific factors such as manual transmissions, and lack of real-time integration with actual auction results. 

---

## 3. Data Description

### Data Source
Primary data is completed auction results scraped from [Cars & Bids](https://carsandbids.com/).

### Features
| Category | Features |
| :--- | :--- |
| **Categorical** | Make, Model, Title Status, Location, Engine, Drivetrain, Transmission, Body Style, Exterior Color, Interior Color |
| **Numerical** | Mileage, Number of Modifications, Vehicle Age |
| **Target** | Final Sale Price (USD) |

### Data Accessibility & Format
Raw data is stored in CSV format before being cleaned and migrated to a PostgreSQL relational database.

## 4. Data Provenance

- **Source Tracking:** Raw data is extracted directly from Cars & Bids and stored in `data/raw/` in its original, unedited format. It can be re-scraped by running `src/data/scraper.py` or using the web UI and entering which vehicle to scrape.
- **Transformation Pipeline:** All cleaning and normalization (e.g., stripping "Miles Shown", handling missing values) are performed through `src/data/cleaner.py`. No edits are done manually, and the transition from raw to processed data is entirely script-driven and reproducible.
- **Versioned Artifacts:** The original scraped datasets are stored in `data/raw/` and the cleaned datasets are stored in `data/processed/`
- **Environment Locking:** All library versions and environment requirements are documented in `requirements.txt` to ensure consistent execution across different systems.

---

## 5. Methodology

### Data Management (Extraction & Transformation)
- **Scraping (`src/data/scraper.py`):** Selenium-based scraper navigates completed auctions (either "Sold for" or "Sold After" listings, excluding listings where reserve was not met) to extract auction details.
- **Cleaning (`src/data/cleaner.py`):** Standardizing features (removing "Miles Shown" for mileage, commas for mileage and price, etc), handling missing values via median/mode imputation, and stripping non-numeric characters from price strings.
- **Database Schema (`src/data/loader.py`):** Cleaned data is loaded into a PostgreSQL database with a structured schema (`src/data/schema.sql`). Categorical variables are handled via one-hot encoding during the ML phase.

### Machine Learning (`src/model/trainer.py`)
- **Model Selection (Stacked Ensemble):** 
    - **RidgeCV:** Captures global linear trends in vehicle depreciation.
    - **Gradient Boosting Regressor:** Handles complex, non-linear interactions (e.g., rarity of specific color/transmission combos).
    - **Bayesian Ridge (Meta-Learner):** Learns the optimal weighting between the linear and boosting layers.
    - **Target Transformation:** Prices are modeled in log-space (`np.log1p`) to normalize variance and improve accuracy on high-value outliers.
- **Evaluation:** Models are evaluated using **Mean Absolute Error (MAE)**, **Mean Absolute Percentage Error (MAPE)**, and **R2 Score**.

---

## 6. Results

### Key Insights
The ensemble approach demonstrates that features such as **manual transmissions**, **mileage**, and **number of modifications** are the strongest predictors. The inclusion of `GradientBoosting` allowed the model to capture enthusiast-specific premiums that standard linear models miss.

### Visualizations (`src/model/visualizer.py`)
- **Accuracy Line Graph:** Actual vs. predicted prices sorted by value.
- **Actual vs. Predicted:** Scatter plot with diagonal reference line.
- **Residual Plots:** Error distribution analysis.
- **Price Distributions:** Boxplots of actual sale prices per model.

### Predictions (`src/model/predictor.py`)
- You are able to enter a link to an ongoing auction and the model will predict the final sale price of the vehicle.

---

## 7. Implementation Details

### Directory Structure
```text
.
├── data/
│   ├── raw/            # Original scraped CSVs
│   └── processed/      # Cleaned CSVs ready for DB load
├── frontend/           # React + Vite dashboard
├── src/
│   ├── api/            # FastAPI backend layer
│   ├── data/           # Ingestion: Scraper, Cleaner, and SQL Loader
│   └── model/          # ML: Trainer, Predictor, and Pipeline
├── visualizations/     # Generated plots and car images
├── .env                # Environment variables (DB credentials)
├── LICENSE
├── requirements.txt
```

---

## 8. Reproducibility and Execution Guide

### Prerequisites
- Python 3.10+
- Node.js & npm
- PostgreSQL

### Setup
1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd car-auction-prediction
   ```

2. **Python Environment:**
   Use the `requirements.txt` file to replicate the exact environment dependencies.
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Database Setup:**
   Create a `.env` file in the root directory with your PostgreSQL credentials:
   ```env
   DB_NAME=car_auctions
   DB_USER=your_user
   DB_PASSWORD=your_password
   DB_HOST=localhost
   DB_PORT=5432
   ```

### Executing the Technical Implementation
1. **Scrape Data** 
   ```bash
   python src/data/scraper.py
   ```

2. **Clean & Load Data:** 
   ```bash
   python src/data/cleaner.py
   python src/data/loader.py
   ```

3. **Train Models:** 
   ```bash
   python src/model/trainer.py
   ```

4. **Create Visualizations:** 
   ```bash
   python src/model/visualizer.py
   ```

5. **Predict Ongoing Auction:** 
   ```bash
   python src/model/predictor.py <auction_url>
   ```
   
### Run Web UI
1. **Frontend**
   ```bash
   cd frontend
   npm install
   npm run dev
   ```

2. **Backend**
   ```bash
   python src/api/main.py
   ```
