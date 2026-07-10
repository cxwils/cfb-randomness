"""
Determines which games are upsets based on the point spread and the actual game results.
"""

import pandas as pd


def add_upset_label(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    home_won = df["homePoints"] > df["awayPoints"]
    away_won = df["awayPoints"] > df["homePoints"]

    df["favorite"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df["spread"] < 0, "favorite"] = "home"
    df.loc[df["spread"] > 0, "favorite"] = "away"
    # spread == 0 or NaN spread -> favorite stays NA

    upset = pd.Series(pd.NA, index=df.index, dtype="object")
    upset.loc[(df["favorite"] == "home") & away_won] = 1
    upset.loc[(df["favorite"] == "home") & home_won] = 0
    upset.loc[(df["favorite"] == "away") & home_won] = 1
    upset.loc[(df["favorite"] == "away") & away_won] = 0
    # favorite is NA (pick'em or missing spread) -> upset stays NA

    df["upset"] = upset

    return df


def upset_label_summary(df: pd.DataFrame) -> None:
    total = len(df)
    n_upset = (df["upset"] == 1).sum()
    n_not_upset = (df["upset"] == 0).sum()
    n_excluded = df["upset"].isna().sum()

    print(f"Total games: {total}")
    print(f"  Upsets:          {n_upset} ({n_upset / total:.1%})")
    print(f"  Non-upsets:      {n_not_upset} ({n_not_upset / total:.1%})")
    print(f"  Excluded (no favorite / missing spread): {n_excluded} ({n_excluded / total:.1%})")