import csv
import os
import re
import time
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.common.exceptions import TimeoutException
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


BASE_URL = "https://carsandbids.com"

MODELS_TO_SCRAPE = [
    {"slug": "nd-miata", "path": "/search/mazda/nd-miata"},
    {"slug": "s2000", "path": "/search/honda/s2000"},
    {"slug": "f8x-m4", "path": "/search/bmw/f8x-m4"},
    {"slug": "e46-m3", "path": "/search/bmw/e46-m3"},
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) "
        "Gecko/20100101 Firefox/123.0"
    )
}

def parse_search_page(html: str) -> List[Dict[str, str]]:
    """
    From a search results page, return sold auctions with:
    - full auction URL
    - sale_price (numbers only, no $ or commas)
    - date (as shown on results, e.g. 3/4/26)
    """
    soup = BeautifulSoup(html, "html.parser")
    auctions: List[Dict[str, str]] = []

    for li in soup.select("ul.auctions-list.past-auctions li.auction-item"):
        text = li.get_text(" ", strip=True)

        if "Sold for" not in text and "Sold After for" not in text:
            continue

        m_price = re.search(r"Sold(?: After)? for \$([\d,]+)", text)
        sale_price = m_price.group(1).replace(",", "") if m_price else ""

        m_date = re.search(r"Ended ([0-9]{1,2}/[0-9]{1,2}/[0-9]{2,4})", text)
        date = m_date.group(1) if m_date else ""

        a = li.select_one('a[href^="/auctions/"]')
        if not a or not a.has_attr("href"):
            continue

        rel_url = a["href"].split("?", 1)[0]
        full_url = urljoin(BASE_URL, rel_url)

        auctions.append(
            {
                "url": full_url,
                "sale_price": sale_price,
                "date": date,
            }
        )

    return auctions


def parse_auction_page(html: str, url: str) -> Dict[str, str]:
    """
    Parse an individual auction page and pull specs from the
    quick-facts dl/dt/dd blocks and count modifications.
    """
    soup = BeautifulSoup(html, "html.parser")

    quick_facts = soup.select_one("div.cnb-details-quick-facts")
    specs: Dict[str, str] = {}

    if quick_facts:
        for dl in quick_facts.find_all("dl"):
            dts = dl.find_all("dt")
            dds = dl.find_all("dd")
            for dt, dd in zip(dts, dds):
                key = dt.get_text(strip=True)
                value = dd.get_text(" ", strip=True)
                specs[key] = value

    # Count modifications
    num_modifications = 0
    modifications_section = soup.select_one("div.detail-modifications")
    if modifications_section:
        # Usually mods are listed as <li> items within this section
        list_items = modifications_section.find_all("li")
        num_modifications = len(list_items)
        # If there are no list items but there is text (e.g., "None known"), it might be 0.
        if num_modifications == 0:
            text = modifications_section.get_text(strip=True).lower()
            if "none" not in text and "stock" not in text:
                # Fallback to count line breaks or sentences if not a list, but usually it's a list.
                pass

    # Extract year from URL slug (e.g., .../2019-mazda-mx-5-miata-club)
    year = url.split('/')[-1].split('-')[0] if url else ""

    return {
        "year": year,
        "make": specs.get("Make", ""),
        "model": specs.get("Model", "").replace(" Save", ""),
        "mileage": specs.get("Mileage", ""),
        "title_status": specs.get("Title Status", ""),
        "location": specs.get("Location", ""),
        "engine": specs.get("Engine", ""),
        "drivetrain": specs.get("Drivetrain", ""),
        "transmission": specs.get("Transmission", ""),
        "body_style": specs.get("Body Style", ""),
        "exterior_color": specs.get("Exterior Color", ""),
        "interior_color": specs.get("Interior Color", ""),
        "num_modifications": str(num_modifications),
    }


def scrape_model(driver, search_path: str, max_pages: Optional[int] = None) -> List[Dict[str, str]]:
    all_rows: List[Dict[str, str]] = []
    seen_urls: Set[str] = set()

    page = 1
    while True:
        if max_pages is not None and page > max_pages:
            break

        search_url = f"{BASE_URL}{search_path}"
        if page > 1:
            search_url = f"{search_url}?page={page}"

        print(f"Fetching search page {page}: {search_url}")
        driver.get(search_url)

        try:
            WebDriverWait(driver, 20).until(
                EC.presence_of_element_located(
                    (
                        By.CSS_SELECTOR,
                        "ul.auctions-list.past-auctions li.auction-item",
                    )
                )
            )
            # Scroll to trigger lazy loading and wait for prices to populate asynchronously
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(3)
        except TimeoutException:
            print("  Timed out waiting for past auction results; stopping.")
            break

        html = driver.page_source
        page_auctions = parse_search_page(html)
        page_auctions = [a for a in page_auctions if a["url"] not in seen_urls]

        if not page_auctions:
            break

        for a in page_auctions:
            seen_urls.add(a["url"])
            print(f"  Fetching auction: {a['url']}")

            try:
                time.sleep(3) # Wait between requests to avoid rate limits
                driver.get(a["url"])
                try:
                    WebDriverWait(driver, 20).until(
                        EC.presence_of_element_located(
                            (By.CSS_SELECTOR, "div.cnb-details-quick-facts")
                        )
                    )
                except TimeoutException:
                    print(f"    Warning: timed out waiting for cnb-details-quick-facts on {a['url']}")
                    
                html = driver.page_source
                details = parse_auction_page(html, a['url'])
                
                row = {**a, **details}
                all_rows.append(row)

            except Exception as e:
                print(f"    Error fetching/parsing auction {a['url']}: {e}")
                continue

        page += 1

    return all_rows


def write_csv(rows: List[Dict[str, str]], output_path: str) -> None:
    if not rows:
        return

    fieldnames = [
        "year",
        "make",
        "model",
        "mileage",
        "title_status",
        "location",
        "engine",
        "drivetrain",
        "transmission",
        "body_style",
        "exterior_color",
        "interior_color",
        "num_modifications",
        "sale_price",
        "date",
        "url",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    # Setup selenium driver
    chrome_options = ChromeOptions()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument(f"user-agent={HEADERS['User-Agent']}")

    driver = webdriver.Chrome(
        service=ChromeService(ChromeDriverManager().install()),
        options=chrome_options,
    )

    try:
        for model in MODELS_TO_SCRAPE:
            print(f"--- Scraping {model['slug']} ---")
            rows = scrape_model(driver, model["path"])
            
            output_file = os.path.join("data", "raw", f"{model['slug']}_raw.csv")
            write_csv(rows, output_file)
            print(f"Wrote {len(rows)} rows to {output_file}\n")
    finally:
        driver.quit()


if __name__ == "__main__":
    main()
