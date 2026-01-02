import requests
import time
import pandas as pd
import os
from datetime import datetime

def fetch_satellite_paper_count(sat_name, api_key, 
                                limit_per_request=100, 
                                pause_sec=1.1, 
                                max_retries=3, 
                                max_total=1000):
    """
    Fetch paper count for a single satellite using Semantic Scholar API.
    Returns the number of papers (capped at max_total).
    """
    total_papers = 0
    offset = 0
    headers = {"x-api-key": api_key}

    while offset < max_total:
        url = "https://api.semanticscholar.org/graph/v1/paper/search"
        query_string = f'"{sat_name}" satellite'
        params = {
            "query": query_string,
            "limit": min(limit_per_request, max_total - offset),
            "offset": offset,
            "fields": "title"
        }

        retries = 0
        wait = pause_sec
        papers = []

        while retries <= max_retries:
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                data = response.json()
                papers = data.get("data", [])
                total_papers += len(papers)
                break
            elif response.status_code == 429:
                retries += 1
                if retries > max_retries:
                    print(f" Too many retries for {sat_name}. Giving up.")
                    return total_papers
                print(f"⚠️ Rate limit hit for {sat_name}. Retry {retries}/{max_retries} after {wait:.1f}s...")
                time.sleep(wait)
                wait *= 1.5
                continue
            else:
                print(f" Error {response.status_code} for {sat_name}")
                return total_papers

        if len(papers) < params["limit"]:
            break

        offset += len(papers)
        time.sleep(pause_sec)

    return total_papers


def get_eo_paper_counts(df_payloads, api_key, save_csv=True):
    """
    Fetch paper counts for all EO satellites in df_payloads.
    Returns:
    - raw_counts: {sat_name: num_papers}
    - normalized_weights: {sat_name: weight}
    Saves results to CSV in 'data/EO_paper_count_yyyymmdd.csv'.
    """
    eo_sats = df_payloads[df_payloads["mission_category"] == "Earth Observation"]
    sat_names = eo_sats["attributes.name"].tolist()

    raw_counts = {}
    print("\n Fetching Semantic Scholar paper counts (up to 1000 per satellite)...\n")

    for sat in sat_names:
        count = fetch_satellite_paper_count(sat, api_key)
        raw_counts[sat] = count
        print(f" - {sat}: {count} papers")

    # Normalize weights
    total = sum(raw_counts.values())
    normalized_weights = {k: (v/total if total > 0 else 0) for k, v in raw_counts.items()}

    # Save to CSV
    if save_csv:
        os.makedirs("data", exist_ok=True)
        today_str = datetime.today().strftime("%Y%m%d")
        filename = f"EO_paper_count_{today_str}.csv"
        filepath = os.path.join("data", filename)
        df = pd.DataFrame({
            "Satellite": list(raw_counts.keys()),
            "PaperCount": list(raw_counts.values()),
            "Weight": [normalized_weights[sat] for sat in raw_counts.keys()]
        })
        df.to_csv(filepath, index=False)
        print(f" Saved results to {filepath}")

    return raw_counts, normalized_weights
