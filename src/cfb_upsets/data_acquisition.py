"""
Pulls raw game and betting line data from the CollegeFootballData (CFBD) API
and saves it.
"""

import json
import os
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load CFBD_API_KEY from .env into the environment
load_dotenv()

API_KEY = os.getenv("CFBD_API_KEY")
BASE_URL = "https://api.collegefootballdata.com"

# Repo root is two levels up from this file (src/cfb_upsets/data_aquisition.py)
REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"


def _get(endpoint: str, params: dict) -> list[dict]:
    """
    Internal helper: makes an authenticated GET request to a CFBD endpoint
    and returns the parsed JSON response.

    Raises an exception if the request fails, so calling code can catch
    and handle it (rather than silently returning bad data).
    """
    if not API_KEY:
        raise RuntimeError(
            "CFBD_API_KEY not found. Make sure it's set in your .env file "
            "at the repo root."
        )

    headers = {"Authorization": f"Bearer {API_KEY}"}
    url = f"{BASE_URL}{endpoint}"

    response = requests.get(url, headers=headers, params=params, timeout=30)
    response.raise_for_status()  # raises an error for 4xx/5xx responses

    return response.json()


def get_games(season: int, season_type: str = "both") -> list[dict]:
    """
    Fetch all FBS games for a given season.

    season_type: "regular", "postseason", or "both". "both" gets us bowl
    games and championship games too, which matter for upset analysis.
    """
    params = {
        "year": season,
        "seasonType": season_type,
        "division": "fbs",
    }
    return _get("/games", params)


def get_lines(season: int, season_type: str = "both") -> list[dict]:
    """
    Fetch betting lines for all games in a given season.

    Each item in the response corresponds to one game, but contains a
    nested "lines" list with entries from multiple providers (DraftKings,
    consensus, etc). We save this raw and pick a provider later in cleaning.py.
    """
    params = {
        "year": season,
        "seasonType": season_type,
    }
    return _get("/lines", params)


def save_raw(data: list[dict], filename: str) -> Path:
    """
    Save raw JSON data to data/raw/. Returns the path written to, so
    calling code can log/confirm it.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = RAW_DATA_DIR / filename

    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)

    return filepath


def pull_season(season: int) -> None:
    """
    Pull both games and lines for one season and save them to data/raw/.
    """
    print(f"Fetching games for {season}...")
    games = get_games(season)
    games_path = save_raw(games, f"games_{season}.json")
    print(f"  -> saved {len(games)} games to {games_path}")

    # Be polite to the API -- small delay between calls
    time.sleep(1)

    print(f"Fetching lines for {season}...")
    lines = get_lines(season)
    lines_path = save_raw(lines, f"lines_{season}.json")
    print(f"  -> saved {len(lines)} line entries to {lines_path}")


def main():
    seasons = [2021, 2022, 2023, 2024, 2025]

    for season in seasons:
        try:
            pull_season(season)
        except requests.exceptions.HTTPError as e:
            print(f"FAILED for {season}: {e}")
            print("Skipping to next season.")
        time.sleep(1)  # pause between seasons too


if __name__ == "__main__":
    main()