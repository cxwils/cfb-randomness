import pandas as pd


def add_upset_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds two columns to the games DataFrame:

    - 'favorite': 'home' or 'away', whichever side has the negative spread
                  (or NA for pick'em games where spread == 0)
    - 'upset':    1 if the underdog won, 0 if the favorite won,
                  NaN for pick'em games (no favorite to upset) or
                  games missing a spread entirely

    Sign convention (confirmed against real data):
        spread < 0  -> home team favored
        spread > 0  -> away team favored
        spread == 0 -> pick'em, no favorite
    """
    df = df.copy()

    home_won = df["homePoints"] > df["awayPoints"]
    away_won = df["awayPoints"] > df["homePoints"]

    df["favorite"] = pd.Series(pd.NA, index=df.index, dtype="object")
    df.loc[df["spread"] < 0, "favorite"] = "home"
    df.loc[df["spread"] > 0, "favorite"] = "away"

    upset = pd.Series(pd.NA, index=df.index, dtype="object")
    upset.loc[(df["favorite"] == "home") & away_won] = 1
    upset.loc[(df["favorite"] == "home") & home_won] = 0
    upset.loc[(df["favorite"] == "away") & home_won] = 1
    upset.loc[(df["favorite"] == "away") & away_won] = 0

    df["upset"] = pd.to_numeric(upset, errors="coerce")

    return df


def upset_label_summary(df: pd.DataFrame) -> None:
    """Quick sanity-check printout after labeling."""
    total = len(df)
    n_upset = (df["upset"] == 1).sum()
    n_not_upset = (df["upset"] == 0).sum()
    n_excluded = df["upset"].isna().sum()

    print(f"Total games: {total}")
    print(f"  Upsets:          {n_upset} ({n_upset / total:.1%})")
    print(f"  Non-upsets:      {n_not_upset} ({n_not_upset / total:.1%})")
    print(f"  Excluded (no favorite / missing spread): {n_excluded} ({n_excluded / total:.1%})")


def add_spread_features(df: pd.DataFrame) -> pd.DataFrame:
    """Feature 1: spread magnitude (how big the favorite/underdog gap is)."""
    df = df.copy()
    df["spread_magnitude"] = df["spread"].abs()
    return df


def add_is_home_underdog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 2: 1 if the home team is the underdog, 0 if home is favored,
    NaN for pick'em / missing spread games. Requires 'favorite' column
    (from add_upset_label) to already exist.
    """
    df = df.copy()
    is_home_underdog = pd.Series(pd.NA, index=df.index, dtype="object")
    is_home_underdog.loc[df["favorite"] == "away"] = 1  # home is underdog
    is_home_underdog.loc[df["favorite"] == "home"] = 0  # home is favored
    df["is_home_underdog"] = pd.to_numeric(is_home_underdog, errors="coerce")
    return df


def add_underdog_conference(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 3: the conference of the underdog team (categorical).
    Requires 'favorite' column to already exist.
    """
    df = df.copy()

    def get_underdog_conf(row):
        if pd.isna(row["favorite"]):
            return pd.NA
        elif row["favorite"] == "home":
            return row["awayConference"]
        elif row["favorite"] == "away":
            return row["homeConference"]
        return pd.NA

    df["underdog_conference"] = df.apply(get_underdog_conf, axis=1)
    return df


def add_elo_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 5: pregame Elo rating differential (home - away).
    CFBD provides homePregameElo / awayPregameElo directly in the games
    data, so no extra computation or API calls are needed here.
    """
    df = df.copy()
    df["elo_diff"] = df["homePregameElo"] - df["awayPregameElo"]
    return df


def compute_rolling_win_pct(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds each team's in-season win percentage ENTERING each game
    (i.e., based only on prior games that season, never the current one
    -- this avoids leaking the outcome into the feature).

    A team's very first game of a season has no prior games, so its
    win_pct_prior defaults to 0.5 (neutral / no information yet).

    Returns a long-format lookup: one row per (game id, team) with
    that team's win_pct_prior for that game.
    """
    home = df[["id", "season", "startDate", "homeTeam", "homePoints", "awayPoints"]].copy()
    home = home.rename(columns={"homeTeam": "team"})
    home["win"] = (home["homePoints"] > home["awayPoints"]).astype(int)

    away = df[["id", "season", "startDate", "awayTeam", "homePoints", "awayPoints"]].copy()
    away = away.rename(columns={"awayTeam": "team"})
    away["win"] = (away["awayPoints"] > away["homePoints"]).astype(int)

    long = pd.concat(
        [home[["id", "season", "startDate", "team", "win"]],
         away[["id", "season", "startDate", "team", "win"]]],
        ignore_index=True,
    )

    # Sort chronologically within each team-season so cumulative counts
    # only look backward in time, never forward.
    long = long.sort_values(["team", "season", "startDate"])

    games_played_prior = long.groupby(["team", "season"]).cumcount()
    wins_cumulative = long.groupby(["team", "season"])["win"].cumsum()
    wins_prior = wins_cumulative - long["win"]  # exclude current game's own result

    win_pct_prior = wins_prior / games_played_prior
    win_pct_prior = win_pct_prior.fillna(0.5)  # first game of season -> neutral prior

    long["win_pct_prior"] = win_pct_prior
    return long[["id", "team", "win_pct_prior"]]


def add_rolling_win_pct_diff(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature 6: difference in each team's in-season prior win percentage
    (home_win_pct_prior - away_win_pct_prior), as a proxy for recent
    team performance / momentum beyond a static Elo rating.
    """
    df = df.copy()
    lookup = compute_rolling_win_pct(df)

    home_lookup = lookup.rename(columns={"team": "homeTeam", "win_pct_prior": "home_win_pct_prior"})
    away_lookup = lookup.rename(columns={"team": "awayTeam", "win_pct_prior": "away_win_pct_prior"})

    df = df.merge(home_lookup[["id", "homeTeam", "home_win_pct_prior"]], on=["id", "homeTeam"], how="left")
    df = df.merge(away_lookup[["id", "awayTeam", "away_win_pct_prior"]], on=["id", "awayTeam"], how="left")

    df["rolling_win_pct_diff"] = df["home_win_pct_prior"] - df["away_win_pct_prior"]
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature pipeline. Runs every step in the correct order and
    returns a DataFrame with the target variable ('upset') plus all
    6 modeling features:

      1. spread_magnitude
      2. is_home_underdog
      3. underdog_conference
      4. week                  (already present in raw data, no-op here)
      5. elo_diff
      6. rolling_win_pct_diff

    Call this once, on the full cleaned dataset, before any train/test
    split or modeling.
    """
    df = add_upset_label(df)
    df = add_spread_features(df)
    df = add_is_home_underdog(df)
    df = add_underdog_conference(df)
    df = add_elo_diff(df)
    df = add_rolling_win_pct_diff(df)
    return df