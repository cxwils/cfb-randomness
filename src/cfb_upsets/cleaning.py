"""
Loads raw games and lines JSON files from data/raw/, filters to FBS-vs-FBS
games only
"""

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = REPO_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = REPO_ROOT / "data" / "processed"

SEASONS = [2021, 2022, 2023, 2024, 2025]

# Preference order for which betting line provider to use per game.
# We try each in order and take the first one that has a spread available.
PROVIDER_PREFERENCE = ["consensus", "Bovada", "teamrankings", "William Hill (New Jersey)"]


def load_games(season: int) -> pd.DataFrame:
    """Load one season's raw games JSON into a DataFrame."""
    filepath = RAW_DATA_DIR / f"games_{season}.json"
    with open(filepath) as f:
        games = json.load(f)
    return pd.DataFrame(games)


def load_lines(season: int) -> pd.DataFrame:
    """Load one season's raw lines JSON into a DataFrame."""
    filepath = RAW_DATA_DIR / f"lines_{season}.json"
    with open(filepath) as f:
        lines = json.load(f)
    return pd.DataFrame(lines)


def filter_fbs_only(games_df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only games where both the home and away team are FBS.
    The API's division param doesn't filter server-side (confirmed by
    inspection), so we filter client-side here instead.
    """
    mask = (games_df["homeClassification"] == "fbs") & (
        games_df["awayClassification"] == "fbs"
    )
    return games_df[mask].copy()


def extract_best_line(lines_row_lines: list) -> dict:
    """
    Given the nested 'lines' list for one game (multiple providers),
    pick the best available spread according to PROVIDER_PREFERENCE.

    Returns a dict with 'spread' and 'provider_used', or
    {'spread': None, 'provider_used': None} if no usable line is found.
    """
    if not lines_row_lines:
        return {"spread": None, "provider_used": None}

    # Index available providers by name for quick lookup
    by_provider = {entry["provider"]: entry for entry in lines_row_lines}

    for provider in PROVIDER_PREFERENCE:
        if provider in by_provider:
            spread = by_provider[provider].get("spread")
            if spread is not None:
                return {"spread": spread, "provider_used": provider}

    # Fallback: take the first entry with a non-null spread, whatever
    # provider it is, rather than losing the game entirely.
    for entry in lines_row_lines:
        if entry.get("spread") is not None:
            return {"spread": entry["spread"], "provider_used": entry["provider"]}

    return {"spread": None, "provider_used": None}


def build_lines_lookup(lines_df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the nested lines data down to one row per game_id, with a
    single chosen spread and which provider it came from.
    """
    extracted = lines_df["lines"].apply(extract_best_line).apply(pd.Series)
    result = pd.concat([lines_df[["id"]], extracted], axis=1)
    return result.rename(columns={"id": "game_id"})


def merge_games_and_lines(games_df: pd.DataFrame, lines_lookup: pd.DataFrame) -> pd.DataFrame:
    """
    Left-join games with their lines on game id. Left join because we
    want to keep every FBS-vs-FBS game even if no line was found for it
    -- we'll decide later (in EDA) whether to drop those or keep them.
    """
    merged = games_df.merge(
        lines_lookup, left_on="id", right_on="game_id", how="left"
    )
    return merged


def clean_season(season: int) -> pd.DataFrame:
    """Full pipeline for one season: load, filter, join."""
    games_df = load_games(season)
    lines_df = load_lines(season)

    games_df = filter_fbs_only(games_df)
    lines_lookup = build_lines_lookup(lines_df)

    merged = merge_games_and_lines(games_df, lines_lookup)
    return merged


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    all_seasons = []

    for season in SEASONS:
        print(f"Cleaning {season}...")
        cleaned = clean_season(season)

        n_total = len(cleaned)
        n_with_line = cleaned["spread"].notna().sum()
        print(
            f"  -> {n_total} FBS-vs-FBS games, "
            f"{n_with_line} with a betting line ({n_with_line / n_total:.1%})"
        )

        out_path = PROCESSED_DATA_DIR / f"games_clean_{season}.csv"
        cleaned.to_csv(out_path, index=False)
        print(f"  -> saved to {out_path}")

        all_seasons.append(cleaned)

    combined = pd.concat(all_seasons, ignore_index=True)
    combined_path = PROCESSED_DATA_DIR / "games_clean_all.csv"
    combined.to_csv(combined_path, index=False)
    print(f"\nSaved combined dataset ({len(combined)} games) to {combined_path}")


if __name__ == "__main__":
    main()