"""
FOOTBALL PREDICTION ENGINE — v10 FINAL
1/X/2 + GG/NG

Design goals:
- Strict chronological / no-leakage feature construction.
- Multi-league safe state: (league, team) is always the key.
- Explicit season handling + season regression.
- Elo + weighted form + home/away splits + rest + H2H.
- League scoring baseline.
- Bad-form / mutual-bad-form / volatility / entropy features.
- Optional real xG / shots / shots-on-target / big-chances / possession / cards.
- XGBoost + RandomForest + Poisson/Dixon-Coles ensemble.
- Validation-only ensemble weights and temperature calibration.
- Multi-fold walk-forward and final holdout reports.
- Empirical confidence calibration from unseen walk-forward predictions.
- Production refit only AFTER evaluation parameters are learned.
- Football-Data.co.uk raw columns are auto-mapped (FTHG/FTAG/FTR etc.).
- Bookmaker odds are NEVER training features; they are for market comparison/value only.
- Optional manual availability adjustment for future fixtures only.

Expected minimal input columns (either native or football-data style):
Date, HomeTeam, AwayTeam, HomeGoals, AwayGoals
OR football-data style:
Date, HomeTeam, AwayTeam, FTHG, FTAG

Optional:
League, Season, HomeXG, AwayXG, HomeShots, AwayShots,
HomeShotsOnTarget, AwayShotsOnTarget, HomeBigChances, AwayBigChances,
HomePossession, AwayPossession, HomeCards, AwayCards,
B365H/B365D/B365A (and other bookmaker odds; value-only, never training features)

Typical use:
1) df = load_data("matches.csv")
2) report = engine.backtest(df)
3) engine.save("engine_v10_final.joblib")
4) pred = engine.predict_match("Team A", "Team B", league="EPL")
"""

from __future__ import annotations

import math
import warnings
import io
import os
import urllib.request
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson as sp_poisson

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except Exception:
    HAS_LGBM = False


# ============================================================
# CONFIG
# ============================================================

@dataclass
class EngineConfig:
    initial_elo: float = 1500.0
    elo_k: float = 15.0
    home_advantage_elo: float = 50.0
    season_regression: float = 1.0 / 3.0
    season_gap_days: int = 60

    form_window: int = 5
    form_decay: float = 0.85
    form_prior_games: float = 5.0
    prior_pts_per_game: float = 1.35
    prior_gf: float = 1.35
    prior_ga: float = 1.10

    h2h_window: int = 10
    h2h_decay: float = 0.80
    h2h_half_life_days: float = 730.0
    h2h_prior_games: float = 5.0

    default_rest_days: float = 7.0
    max_rest_days: float = 30.0

    min_xg: float = 0.10
    max_xg: float = 5.50
    poisson_max_goals: int = 10

    # Holdout
    train_frac: float = 0.65
    val_frac: float = 0.15
    min_train_matches: int = 150
    min_holdout_test_matches: int = 30

    # Walk-forward
    wf_initial_train_frac: float = 0.55
    wf_val_frac: float = 0.15
    wf_test_frac: float = 0.10
    wf_min_test_matches: int = 30
    wf_max_folds: int = 8

    # ML
    xgb_estimators: int = 450
    xgb_max_depth: int = 4
    xgb_learning_rate: float = 0.035
    xgb_subsample: float = 0.85
    xgb_colsample: float = 0.85
    xgb_reg_lambda: float = 2.0
    xgb_min_child_weight: float = 3.0

    rf_estimators: int = 700
    rf_max_depth: int = 8
    rf_min_samples_leaf: int = 5
    rf_max_features: str = "sqrt"

    random_state: int = 42
    use_xgb: bool = True
    use_rf: bool = True
    use_lgbm: bool = True
    lgbm_estimators: int = 450
    lgbm_max_depth: int = 6
    lgbm_learning_rate: float = 0.03
    lgbm_subsample: float = 0.80
    lgbm_colsample: float = 0.85
    lgbm_min_child_samples: int = 15
    lgbm_reg_alpha: float = 0.10
    lgbm_reg_lambda: float = 1.50
    calibration_l2: float = 1.0
    optimize_btts_threshold: bool = True
    btts_threshold_min: float = 0.40
    btts_threshold_max: float = 0.60
    btts_threshold_step: float = 0.01
    market_blend_weight: float = 0.0
    run_walk_forward: bool = True

    # Calibration search
    temperature_min: float = 0.40
    temperature_max: float = 4.00


# ============================================================
# UTILITIES / DATA NORMALIZATION
# ============================================================

RESULT_TO_INT = {"X": 0, "1": 1, "2": 2}
INT_TO_RESULT = {0: "X", 1: "1", 2: "2"}


def _safe_float(x, default=np.nan) -> float:
    try:
        v = float(x)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _clean_team(x) -> str:
    return str(x).strip()


def _clean_league(x) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "DEFAULT"
    s = str(x).strip()
    return s if s else "DEFAULT"


def _season_fallback(date: pd.Timestamp) -> str:
    # European-friendly fallback; explicit Season is preferred.
    return f"{date.year}-{date.year + 1}" if date.month >= 7 else f"{date.year - 1}-{date.year}"


def _clip01(v):
    return float(np.clip(v, 0.0, 1.0))


def entropy3(p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=float), 1e-12, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(3.0))


def entropy2(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1 - 1e-12))
    return float(-(p * np.log(p) + (1 - p) * np.log(1 - p)) / np.log(2.0))


def sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _gd_multiplier(gd: float) -> float:
    """V9 FIX 1: Football Elo goal-difference multiplier."""
    gd = abs(float(gd))
    if gd <= 1.0:
        return 1.0
    if gd == 2.0:
        return 1.5
    return (11.0 + gd) / 8.0


def safe_predict_proba(model, X, n_classes: int) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    out = np.zeros((len(X), n_classes), dtype=float)
    classes = getattr(model, "classes_", None)
    if classes is None:
        if raw.shape[1] == n_classes:
            out = raw
        else:
            raise ValueError("Model returned unexpected probability shape")
    else:
        for i, cls in enumerate(classes):
            c = int(cls)
            if 0 <= c < n_classes:
                out[:, c] = raw[:, i]
    out = np.clip(out, 1e-9, None)
    out /= out.sum(axis=1, keepdims=True)
    return out


def safe_binary_proba(model, X) -> np.ndarray:
    raw = np.asarray(model.predict_proba(X), dtype=float)
    classes = list(getattr(model, "classes_", []))
    if 1 in classes:
        p = raw[:, classes.index(1)]
    elif len(classes) == 1:
        p = np.full(len(X), float(classes[0] == 1))
    else:
        p = raw[:, 1]
    return np.clip(p, 1e-7, 1 - 1e-7)


# football-data.co.uk / common alternate names -> engine schema
COLUMN_ALIASES = {
    "HomeGoals": ["HomeGoals", "FTHG", "HG", "Home_Goals"],
    "AwayGoals": ["AwayGoals", "FTAG", "AG", "Away_Goals"],
    "HomeTeam": ["HomeTeam", "Home", "Home Team"],
    "AwayTeam": ["AwayTeam", "Away", "Away Team"],
    "Date": ["Date", "date", "MatchDate", "match_date"],
    "League": ["League", "league", "Competition", "Div", "Division"],
    "Season": ["Season", "season", "SeasonKey"],
}

OPTIONAL_PAIRS = {
    "xg_actual": ("HomeXG", "AwayXG"),
    "shots": ("HomeShots", "AwayShots"),
    "sot": ("HomeShotsOnTarget", "AwayShotsOnTarget"),
    "big_chances": ("HomeBigChances", "AwayBigChances"),
    "possession": ("HomePossession", "AwayPossession"),
    "cards": ("HomeCards", "AwayCards"),
}

ODDS_PREFIXES = ["B365", "BW", "IW", "PS", "WH", "VC", "Max", "Avg"]


def _rename_by_aliases(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower = {str(c).strip().lower(): c for c in df.columns}
    ren = {}
    for target, aliases in COLUMN_ALIASES.items():
        if target in df.columns:
            continue
        for alias in aliases:
            if alias.lower() in lower:
                ren[lower[alias.lower()]] = target
                break
    return df.rename(columns=ren)


def load_data(path_or_df: Any, default_league: str = "DEFAULT") -> pd.DataFrame:
    """Load and normalize common football datasets, including Football-Data.co.uk CSVs."""
    if isinstance(path_or_df, pd.DataFrame):
        df = path_or_df.copy()
    else:
        df = pd.read_csv(path_or_df)

    df = _rename_by_aliases(df)
    required = {"Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=True)
    # If dayfirst parsing creates too many NaT, retry conventional parsing.
    if df["Date"].isna().mean() > 0.05:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce", dayfirst=False)
    if df["Date"].isna().any():
        raise ValueError("Some Date values could not be parsed.")

    df["HomeTeam"] = df["HomeTeam"].map(_clean_team)
    df["AwayTeam"] = df["AwayTeam"].map(_clean_team)
    df["HomeGoals"] = pd.to_numeric(df["HomeGoals"], errors="coerce")
    df["AwayGoals"] = pd.to_numeric(df["AwayGoals"], errors="coerce")
    if df[["HomeGoals", "AwayGoals"]].isna().any().any():
        raise ValueError("HomeGoals/AwayGoals contains invalid values.")
    if (df[["HomeGoals", "AwayGoals"]] < 0).any().any():
        raise ValueError("Negative goal values are invalid.")
    if (df["HomeTeam"] == df["AwayTeam"]).any():
        raise ValueError("Found a match with HomeTeam == AwayTeam.")

    if "League" not in df.columns:
        df["League"] = default_league
    df["League"] = df["League"].map(_clean_league)

    has_explicit_season = "Season" in df.columns and df["Season"].notna().any()
    if has_explicit_season:
        df["SeasonKey"] = df["Season"].astype(str).str.strip()
    else:
        df["SeasonKey"] = df["Date"].map(_season_fallback)

    df["HomeGoals"] = df["HomeGoals"].astype(int)
    df["AwayGoals"] = df["AwayGoals"].astype(int)
    df["Result"] = np.where(
        df["HomeGoals"] > df["AwayGoals"], 1,
        np.where(df["HomeGoals"] < df["AwayGoals"], 2, 0)
    ).astype(int)
    df["BTTS"] = ((df["HomeGoals"] > 0) & (df["AwayGoals"] > 0)).astype(int)

    # Remove exact duplicate match rows.
    dup_cols = ["Date", "League", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"]
    df = df.drop_duplicates(dup_cols).copy()

    # Numeric optional stats. A pair is active only when both columns exist.
    present_stats = {}
    for prefix, (hc, ac) in OPTIONAL_PAIRS.items():
        if hc in df.columns and ac in df.columns:
            df[hc] = pd.to_numeric(df[hc], errors="coerce")
            df[ac] = pd.to_numeric(df[ac], errors="coerce")
            if df[[hc, ac]].notna().any().any():
                present_stats[prefix] = (hc, ac)

    df = df.sort_values(["Date", "League", "HomeTeam", "AwayTeam"], kind="mergesort").reset_index(drop=True)
    df.attrs["present_stats"] = present_stats
    df.attrs["has_explicit_season"] = bool(has_explicit_season)
    df.attrs["source_columns"] = list(df.columns)
    return df


# ============================================================
# STATE TRACKERS — ALL KEYS ARE LEAGUE-AWARE
# ============================================================

class LeagueAwareElo:
    """League-aware Elo with optional time decay toward the neutral prior."""
    def __init__(self, cfg: EngineConfig):
        self.ratings: Dict[Tuple[str, str], float] = {}
        self.last_date: Dict[Tuple[str, str], pd.Timestamp] = {}
        self.cfg = cfg

    def key(self, league, team):
        return (_clean_league(league), _clean_team(team))

    def get(self, league, team, current_date=None):
        k = self.key(league, team)
        rating = float(self.ratings.get(k, self.cfg.initial_elo))
        if current_date is not None and k in self.last_date:
            days = max(0.0, float((pd.Timestamp(current_date) - self.last_date[k]).days))
            if days > 0:
                decay = math.exp(-0.03 * days / 365.0)
                rating = self.cfg.initial_elo + (rating - self.cfg.initial_elo) * decay
        return rating

    def update(self, league, home, away, hg, ag, match_date=None):
        kh = self.key(league, home)
        ka = self.key(league, away)
        rh0 = self.get(league, home, match_date)
        ra0 = self.get(league, away, match_date)
        rh = rh0 + self.cfg.home_advantage_elo
        ra = ra0
        expected_home = 1.0 / (1.0 + 10.0 ** ((ra - rh) / 400.0))
        actual_home = 1.0 if hg > ag else (0.5 if hg == ag else 0.0)
        # V9 FIX 1: scale Elo update by goal difference.
        multiplier = _gd_multiplier(abs(hg - ag))
        change = self.cfg.elo_k * multiplier * (actual_home - expected_home)
        self.ratings[kh] = rh0 + change
        self.ratings[ka] = ra0 - change
        if match_date is not None:
            self.last_date[kh] = pd.Timestamp(match_date)
            self.last_date[ka] = pd.Timestamp(match_date)

    def regress_league(self, league):
        league = _clean_league(league)
        for (lg, team), old in list(self.ratings.items()):
            if lg == league:
                self.ratings[(lg, team)] = old + self.cfg.season_regression * (self.cfg.initial_elo - old)


class FormTracker:
    def __init__(self, cfg: EngineConfig):
        self.overall: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}
        self.home: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}
        self.away: Dict[Tuple[str, str], List[Tuple[float, float, float]]] = {}
        self.cfg = cfg

    def key(self, league, team):
        return (_clean_league(league), _clean_team(team))

    def _weighted(self, games):
        games = games[-self.cfg.form_window:]
        if not games:
            return self.cfg.prior_pts_per_game, self.cfg.prior_gf, self.cfg.prior_ga, 0.0
        n = len(games)
        w = np.array([self.cfg.form_decay ** (n - 1 - i) for i in range(n)], dtype=float)
        vals = np.asarray(games, dtype=float)
        ws = w.sum()
        return float((vals[:, 0] * w).sum() / ws), float((vals[:, 1] * w).sum() / ws), float((vals[:, 2] * w).sum() / ws), float(n)

    def _shrink(self, pts, gf, ga, n):
        weight = n / (n + self.cfg.form_prior_games)
        pts = weight * pts + (1 - weight) * self.cfg.prior_pts_per_game
        gf = weight * gf + (1 - weight) * self.cfg.prior_gf
        ga = weight * ga + (1 - weight) * self.cfg.prior_ga
        return float(pts), float(gf), float(ga)

    def _current(self, store, league, team):
        raw = self._weighted(store.get(self.key(league, team), []))
        return (*self._shrink(raw[0], raw[1], raw[2], raw[3]), raw[3])

    def current_overall(self, league, team):
        return self._current(self.overall, league, team)

    def current_home(self, league, team):
        return self._current(self.home, league, team)

    def current_away(self, league, team):
        return self._current(self.away, league, team)

    def add(self, league, home, away, hg, ag):
        kh, ka = self.key(league, home), self.key(league, away)
        hp = 3.0 if hg > ag else (1.0 if hg == ag else 0.0)
        ap = 3.0 if ag > hg else (1.0 if hg == ag else 0.0)
        self.overall.setdefault(kh, []).append((hp, hg, ag))
        self.overall.setdefault(ka, []).append((ap, ag, hg))
        self.home.setdefault(kh, []).append((hp, hg, ag))
        self.away.setdefault(ka, []).append((ap, ag, hg))

    def recent_behavior(self, league, team):
        games = self.overall.get(self.key(league, team), [])[-self.cfg.form_window:]
        if not games:
            return {"losses": 0.0, "loss_rate": 0.0, "current_loss_streak": 0.0, "all_losses": 0.0, "n": 0.0, "draws": 0.0}
        losses = sum(1 for pts, _, _ in games if pts == 0.0)
        draws = sum(1 for pts, _, _ in games if pts == 1.0)
        streak = 0
        for pts, _, _ in reversed(games):
            if pts == 0.0:
                streak += 1
            else:
                break
        n = len(games)
        return {
            "losses": float(losses),
            "loss_rate": float(losses / n),
            "current_loss_streak": float(streak),
            "all_losses": float(losses == n),
            "draws": float(draws / n),
            "n": float(n),
        }


class RestTracker:
    def __init__(self):
        self.last_date: Dict[Tuple[str, str], pd.Timestamp] = {}

    def key(self, league, team):
        return (_clean_league(league), _clean_team(team))

    def days_since(self, league, team, current_date):
        last = self.last_date.get(self.key(league, team))
        if last is None:
            return None
        return max(0, int((pd.Timestamp(current_date) - last).days))

    def update(self, league, team, date):
        self.last_date[self.key(league, team)] = pd.Timestamp(date)


class H2HTracker:
    def __init__(self, cfg: EngineConfig):
        self.history: Dict[Tuple[str, Tuple[str, str]], List[dict]] = {}
        self.cfg = cfg

    def key(self, league, a, b):
        return (_clean_league(league), tuple(sorted([_clean_team(a), _clean_team(b)])))

    def add(self, league, home, away, hg, ag, date):
        self.history.setdefault(self.key(league, home, away), []).append({
            "home": _clean_team(home),
            "away": _clean_team(away),
            "hg": float(hg),
            "ag": float(ag),
            "date": pd.Timestamp(date),
        })

    def current(self, league, home, away, current_date):
        games = self.history.get(self.key(league, home, away), [])[-self.cfg.h2h_window:]
        if not games:
            return 0.0, 0.0, 0.0, 0.0
        ref = pd.Timestamp(current_date)
        weights, diffs, draws, btts = [], [], [], []
        for g in games:
            age_days = max(0.0, float((ref - g["date"]).days))
            time_w = 0.5 ** (age_days / max(self.cfg.h2h_half_life_days, 1.0))
            dif = g["hg"] - g["ag"] if g["home"] == _clean_team(home) else g["ag"] - g["hg"]
            weights.append(time_w)
            diffs.append(dif)
            draws.append(float(g["hg"] == g["ag"]))
            btts.append(float(g["hg"] > 0 and g["ag"] > 0))
        w = np.asarray(weights, dtype=float)
        n = len(games)
        shrink = n / (n + self.cfg.h2h_prior_games)
        return (
            float(np.average(diffs, weights=w) * shrink),
            float(n),
            float(np.average(draws, weights=w)),
            float(np.average(btts, weights=w)),
        )


class LeagueScoringTracker:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.stats: Dict[Tuple[str, str], dict] = {}
        self.league_global: Dict[str, dict] = {}

    def current(self, league, season):
        key = (_clean_league(league), str(season))
        s = self.stats.get(key)
        if not s or s["n"] == 0:
            # Long-run league prior if available, else generic prior.
            lg = self.league_global.get(_clean_league(league))
            if lg and lg["n"] > 0:
                return lg["home"] / lg["n"], lg["away"] / lg["n"], 0.0
            return self.cfg.prior_gf, self.cfg.prior_ga, 0.0
        return s["home"] / s["n"], s["away"] / s["n"], float(s["n"])

    def add(self, league, season, hg, ag):
        league = _clean_league(league)
        season = str(season)
        key = (league, season)
        s = self.stats.setdefault(key, {"home": 0.0, "away": 0.0, "n": 0})
        s["home"] += float(hg)
        s["away"] += float(ag)
        s["n"] += 1
        g = self.league_global.setdefault(league, {"home": 0.0, "away": 0.0, "n": 0})
        g["home"] += float(hg)
        g["away"] += float(ag)
        g["n"] += 1


class StatTracker:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.history: Dict[Tuple[str, str], List[float]] = {}

    def key(self, league, team):
        return (_clean_league(league), _clean_team(team))

    def current(self, league, team):
        games = self.history.get(self.key(league, team), [])[-self.cfg.form_window:]
        if not games:
            return 0.0, 1.0, 0.0
        n = len(games)
        w = np.array([self.cfg.form_decay ** (n - 1 - i) for i in range(n)], dtype=float)
        return float(np.average(games, weights=w)), 0.0, float(n)

    def add(self, league, team, value):
        if value is None:
            return
        v = _safe_float(value)
        if np.isnan(v):
            return
        self.history.setdefault(self.key(league, team), []).append(float(v))


# ============================================================
# FEATURES
# ============================================================

BASE_FEATURE_NAMES = [
    # Elo
    "elo_home", "elo_away", "elo_diff",
    # Overall form
    "form_home", "form_away", "form_diff",
    "gf_home", "ga_home", "gf_away", "ga_away",
    "gf_ga_diff1", "gf_ga_diff2",
    # Venue split
    "home_venue_pts", "home_venue_gf", "home_venue_ga",
    "away_venue_pts", "away_venue_gf", "away_venue_ga",
    "home_form_games", "away_form_games", "home_venue_games", "away_venue_games",
    # Rest
    "rest_home", "rest_away", "rest_diff", "rest_home_missing", "rest_away_missing",
    # H2H
    "h2h_goal_diff", "h2h_games", "h2h_draw_rate", "h2h_btts_rate",
    # League scoring
    "league_home_goals_avg", "league_away_goals_avg", "league_matches_seen",
    # Bad-form / volatility
    "home_losses_last5", "away_losses_last5",
    "home_loss_rate_last5", "away_loss_rate_last5",
    "home_loss_streak", "away_loss_streak",
    "both_bad_form", "both_lost_all5", "mutual_bad_form_index", "mutual_loss_streak_index",
    "form_closeness", "bad_form_gap",
    "home_draw_rate_last5", "away_draw_rate_last5",
    "elo_closeness", "form_entropy_proxy",
    "home_win_rate_last3", "away_win_rate_last3",
    "home_draw_rate_last3", "away_draw_rate_last3",
    "home_loss_rate_last3", "away_loss_rate_last3",
    "home_avg_goals_last3", "away_avg_goals_last3",
    "home_avg_conceded_last3", "away_avg_conceded_last3",
    "home_goal_volatility_l5", "away_goal_volatility_l5",
    "home_scoring_volatility_l5", "away_scoring_volatility_l5",
    "home_clean_sheet_l5", "away_clean_sheet_l5",
    "home_failed_to_score_l5", "away_failed_to_score_l5",
    "draw_pressure", "competitive_balance",
    "attack_balance", "defense_balance",
]


def build_feature_names(stat_prefixes):
    names = list(BASE_FEATURE_NAMES)
    for prefix in sorted(stat_prefixes):
        names.extend([
            f"{prefix}_home", f"{prefix}_away",
            f"{prefix}_home_missing", f"{prefix}_away_missing",
            f"{prefix}_diff",
        ])
    return names


def assemble_features(elo, form, rest, h2h, league_stats, stat_trackers,
                       league, season, home, away, current_date) -> List[float]:
    helo = elo.get(league, home, current_date)
    aelo = elo.get(league, away, current_date)
    hf, hgf, hga, hfn = form.current_overall(league, home)
    af, agf, aga, afn = form.current_overall(league, away)
    hf_h, hgf_h, hga_h, hhn = form.current_home(league, home)
    af_a, agf_a, aga_a, aan = form.current_away(league, away)

    rest_h_raw = rest.days_since(league, home, current_date)
    rest_a_raw = rest.days_since(league, away, current_date)
    rest_h_missing = float(rest_h_raw is None)
    rest_a_missing = float(rest_a_raw is None)
    rest_h = 7.0 if rest_h_raw is None else min(float(rest_h_raw), 30.0)
    rest_a = 7.0 if rest_a_raw is None else min(float(rest_a_raw), 30.0)

    h2h_diff, h2h_n, h2h_draw, h2h_btts = h2h.current(league, home, away, current_date)
    league_hg, league_ag, league_n = league_stats.current(league, season)

    hb = form.recent_behavior(league, home)
    ab = form.recent_behavior(league, away)
    both_bad = float(hb["losses"] >= 3 and ab["losses"] >= 3)
    both_all5 = float(hb["all_losses"] >= 1 and ab["all_losses"] >= 1)
    mutual_bad = math.sqrt(max(0.0, hb["loss_rate"] * ab["loss_rate"]))
    mutual_streak = min(hb["current_loss_streak"], ab["current_loss_streak"]) / max(1.0, float(5.0))
    form_closeness = _clip01(1.0 - abs(hf - af) / 3.0)
    bad_gap = abs(hb["loss_rate"] - ab["loss_rate"])
    elo_close = _clip01(1.0 - abs(helo - aelo) / 250.0)
    # High when recent point accumulation is similarly poor / unstable.
    form_entropy_proxy = float((entropy2(hb["loss_rate"]) + entropy2(ab["loss_rate"])) / 2.0)

    def _lastn(team, n):
        games = form.overall.get(form.key(league, team), [])[-n:]
        if not games:
            return (0.33, 0.33, 0.33, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0)
        pts = np.asarray([g[0] for g in games], dtype=float)
        gf = np.asarray([g[1] for g in games], dtype=float)
        ga = np.asarray([g[2] for g in games], dtype=float)
        return (float(np.mean(pts == 3.0)), float(np.mean(pts == 1.0)),
                float(np.mean(pts == 0.0)), float(gf.mean()), float(ga.mean()),
                float(gf.std()), float(ga.std()), float(np.mean(ga == 0.0)),
                float(np.mean(gf == 0.0)))

    hr3, ar3 = _lastn(home, 3), _lastn(away, 3)
    hr5, ar5 = _lastn(home, 5), _lastn(away, 5)
    draw_pressure = float(np.clip((hb["draws"] + ab["draws"]) / 2.0 + 0.50 * elo_close, 0.0, 1.0))
    competitive_balance = float(np.clip(0.65 * elo_close + 0.35 * form_closeness, 0.0, 1.0))
    attack_balance = float(np.clip(1.0 - abs(hgf - agf) / 3.0, 0.0, 1.0))
    defense_balance = float(np.clip(1.0 - abs(hga - aga) / 3.0, 0.0, 1.0))

    feat = [
        helo, aelo, helo - aelo,
        hf, af, hf - af,
        hgf, hga, agf, aga,
        hgf - aga, agf - hga,
        hf_h, hgf_h, hga_h,
        af_a, agf_a, aga_a,
        hfn, afn, hhn, aan,
        rest_h, rest_a, rest_h - rest_a,
        rest_h_missing, rest_a_missing,
        h2h_diff, h2h_n, h2h_draw, h2h_btts,
        league_hg, league_ag, league_n,
        hb["losses"], ab["losses"],
        hb["loss_rate"], ab["loss_rate"],
        hb["current_loss_streak"], ab["current_loss_streak"],
        both_bad, both_all5, mutual_bad, mutual_streak,
        form_closeness, bad_gap,
        hb["draws"], ab["draws"],
        elo_close, form_entropy_proxy,
        hr3[0], ar3[0], hr3[1], ar3[1], hr3[2], ar3[2],
        hr3[3], ar3[3], hr3[4], ar3[4],
        hr5[6], ar5[6], hr5[5], ar5[5],
        hr5[7], ar5[7], hr5[8], ar5[8],
        draw_pressure, competitive_balance, attack_balance, defense_balance,
    ]

    for prefix in sorted(stat_trackers):
        tr = stat_trackers[prefix]
        hv, hm, hn = tr.current(league, home)
        av, am, an = tr.current(league, away)
        feat.extend([hv, av, hm, am, hv - av])

    return [float(v) for v in feat]


def build_features(df: pd.DataFrame, cfg: EngineConfig):
    """Build chronological features without allowing same-day result leakage.

    Football-Data datasets usually provide dates but not reliable kickoff times.
    Therefore all fixtures on a calendar date are featurized from the state that
    existed before that date; their results are added only after every fixture
    for that date has received its features.
    """
    elo = LeagueAwareElo(cfg)
    form = FormTracker(cfg)
    rest = RestTracker()
    h2h = H2HTracker(cfg)
    league_stats = LeagueScoringTracker(cfg)
    present_stats = df.attrs.get("present_stats", {})
    stat_trackers = {prefix: StatTracker(cfg) for prefix in present_stats}

    X, yr, yb, hg_list, ag_list = [], [], [], [], []
    previous_season: Dict[str, str] = {}
    previous_date: Dict[str, pd.Timestamp] = {}

    # Keep source order within each date, but do not update trackers until the
    # complete date has been featurized.
    for date_value, day in df.groupby(df["Date"].dt.normalize(), sort=True):
        date = pd.Timestamp(date_value)

        # Apply each league's season regression once, before the day's features.
        first_for_league = day.drop_duplicates("League", keep="first")
        for _, row in first_for_league.iterrows():
            league = row["League"]
            season = row["SeasonKey"]
            old_season = previous_season.get(league)
            old_date = previous_date.get(league)
            if old_season is not None:
                season_changed = str(season) != str(old_season)
                if not bool(df.attrs.get("has_explicit_season", False)):
                    season_changed = bool((date - old_date).days > cfg.season_gap_days) if old_date is not None else False
                if season_changed:
                    elo.regress_league(league)

        day_rows = []
        for idx, row in day.iterrows():
            league = row["League"]
            season = row["SeasonKey"]
            home, away = row["HomeTeam"], row["AwayTeam"]
            feat = assemble_features(
                elo, form, rest, h2h, league_stats, stat_trackers,
                league, season, home, away, date,
            )
            X.append(feat)
            hg, ag = float(row["HomeGoals"]), float(row["AwayGoals"])
            yr.append(int(row["Result"]))
            yb.append(int(row["BTTS"]))
            hg_list.append(hg)
            ag_list.append(ag)
            day_rows.append((row, league, season, home, away, hg, ag))

        # Only after all same-day feature rows are captured may results update state.
        for row, league, season, home, away, hg, ag in day_rows:
            elo.update(league, home, away, hg, ag, date)
            form.add(league, home, away, hg, ag)
            rest.update(league, home, date)
            rest.update(league, away, date)
            h2h.add(league, home, away, hg, ag, date)
            league_stats.add(league, season, hg, ag)
            for prefix, (hc, ac) in present_stats.items():
                stat_trackers[prefix].add(league, home, row[hc])
                stat_trackers[prefix].add(league, away, row[ac])
            previous_season[league] = str(season)
            previous_date[league] = date

    return (
        np.asarray(X, dtype=float),
        np.asarray(yr, dtype=int),
        np.asarray(yb, dtype=int),
        np.asarray(hg_list, dtype=float),
        np.asarray(ag_list, dtype=float),
        elo, form, rest, h2h, league_stats, stat_trackers,
    )


# ============================================================
# POISSON / DIXON-COLES
# ============================================================

def poisson_prediction(home_xg, away_xg, rho=0.0, max_goals=10):
    home_xg = float(np.clip(home_xg, 0.10, 5.50))
    away_xg = float(np.clip(away_xg, 0.10, 5.50))
    rho = float(np.clip(rho, -0.20, 0.20))

    goals = np.arange(max_goals)
    ph = sp_poisson.pmf(goals, home_xg)
    pa = sp_poisson.pmf(goals, away_xg)
    grid = np.outer(ph, pa)

    tau00 = 1 - home_xg * away_xg * rho
    tau01 = 1 + home_xg * rho
    tau10 = 1 + away_xg * rho
    tau11 = 1 - rho

    grid[0, 0] *= max(0.0, tau00)
    grid[0, 1] *= max(0.0, tau01)
    grid[1, 0] *= max(0.0, tau10)
    grid[1, 1] *= max(0.0, tau11)

    s = float(grid.sum())
    if not np.isfinite(s) or s <= 0:
        grid = np.outer(ph, pa)
        s = float(grid.sum())
    grid /= s

    hi, ai = np.indices(grid.shape)
    p1 = float(grid[hi > ai].sum())
    px = float(grid[hi == ai].sum())
    p2 = float(grid[hi < ai].sum())
    pgg = float(grid[(hi > 0) & (ai > 0)].sum())
    return {"1": p1, "X": px, "2": p2, "GG": pgg, "NG": 1.0 - pgg}


def fit_rho(hg_arr, ag_arr, home_lam, away_lam):
    hg_arr = np.asarray(hg_arr, dtype=float)
    ag_arr = np.asarray(ag_arr, dtype=float)
    lh = np.clip(np.asarray(home_lam, dtype=float), 0.05, 8.0)
    la = np.clip(np.asarray(away_lam, dtype=float), 0.05, 8.0)

    def nll(rho_arr):
        rho = float(rho_arr[0])
        tau = np.ones_like(lh)
        m00 = (hg_arr == 0) & (ag_arr == 0)
        m01 = (hg_arr == 0) & (ag_arr == 1)
        m10 = (hg_arr == 1) & (ag_arr == 0)
        m11 = (hg_arr == 1) & (ag_arr == 1)
        tau[m00] = 1 - lh[m00] * la[m00] * rho
        tau[m01] = 1 + lh[m01] * rho
        tau[m10] = 1 + la[m10] * rho
        tau[m11] = 1 - rho
        if np.any(tau <= 0):
            return 1e100
        p = sp_poisson.pmf(hg_arr, lh) * sp_poisson.pmf(ag_arr, la) * tau
        return float(-np.log(np.clip(p, 1e-12, None)).sum())

    try:
        res = minimize(nll, x0=np.array([-0.02]), bounds=[(-0.15, 0.15)], method="L-BFGS-B")
        if res.success and np.isfinite(res.fun):
            return float(res.x[0])
    except Exception:
        pass
    return 0.0


# ============================================================
# MODELS
# ============================================================

def build_xgb(n_classes, cfg: EngineConfig):
    if not HAS_XGB or not cfg.use_xgb:
        return None
    params = dict(
        n_estimators=cfg.xgb_estimators,
        max_depth=cfg.xgb_max_depth,
        learning_rate=cfg.xgb_learning_rate,
        subsample=cfg.xgb_subsample,
        colsample_bytree=cfg.xgb_colsample,
        reg_lambda=cfg.xgb_reg_lambda,
        min_child_weight=cfg.xgb_min_child_weight,
        random_state=cfg.random_state,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="mlogloss" if n_classes == 3 else "logloss",
    )
    if n_classes == 3:
        params.update(objective="multi:softprob", num_class=3)
    else:
        params.update(objective="binary:logistic")
    return Pipeline([("scaler", StandardScaler()), ("model", XGBClassifier(**params))])


def build_rf(n_classes, cfg: EngineConfig):
    if not cfg.use_rf:
        return None
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", RandomForestClassifier(
            n_estimators=cfg.rf_estimators,
            max_depth=cfg.rf_max_depth,
            min_samples_leaf=cfg.rf_min_samples_leaf,
            max_features=cfg.rf_max_features,
            class_weight="balanced_subsample",
            random_state=cfg.random_state,
            n_jobs=-1,
        )),
    ])


def build_lgbm(n_classes, cfg: EngineConfig):
    if not HAS_LGBM or not cfg.use_lgbm:
        return None
    params = dict(
        n_estimators=cfg.lgbm_estimators, max_depth=cfg.lgbm_max_depth,
        learning_rate=cfg.lgbm_learning_rate, subsample=cfg.lgbm_subsample,
        colsample_bytree=cfg.lgbm_colsample, min_child_samples=cfg.lgbm_min_child_samples,
        reg_alpha=cfg.lgbm_reg_alpha, reg_lambda=cfg.lgbm_reg_lambda,
        random_state=cfg.random_state, n_jobs=-1, verbosity=-1,
    )
    if n_classes == 3:
        params.update(objective="multiclass", num_class=3)
    else:
        params.update(objective="binary")
    return LGBMClassifier(**params)


def build_poisson_regressor():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", PoissonRegressor(alpha=1.0, max_iter=2000)),
    ])


def fit_models(X, yr, yb, hg, ag, cfg: EngineConfig):
    result_models, btts_models = [], []
    if len(np.unique(yr)) < 3:
        raise ValueError("Training slice must contain all 1/X/2 classes.")
    if len(np.unique(yb)) < 2:
        raise ValueError("Training slice must contain both GG/NG classes.")

    for builder in (build_xgb, build_rf, build_lgbm):
        m = builder(3, cfg)
        if m is not None:
            m.fit(X, yr)
            result_models.append(m)
        m = builder(2, cfg)
        if m is not None:
            m.fit(X, yb)
            btts_models.append(m)

    if not result_models or not btts_models:
        raise RuntimeError("At least one ML learner must be enabled.")

    home_xg = build_poisson_regressor().fit(X, hg)
    away_xg = build_poisson_regressor().fit(X, ag)
    home_lam = np.clip(home_xg.predict(X), cfg.min_xg, cfg.max_xg)
    away_lam = np.clip(away_xg.predict(X), cfg.min_xg, cfg.max_xg)
    rho = fit_rho(hg, ag, home_lam, away_lam)

    return {
        "result_models": result_models,
        "btts_models": btts_models,
        "home_xg_model": home_xg,
        "away_xg_model": away_xg,
        "rho": float(rho),
    }


def average_ml_result(models, X):
    probs = [safe_predict_proba(m, X, 3) for m in models]
    return np.mean(probs, axis=0), probs


def average_ml_btts(models, X):
    probs = [safe_binary_proba(m, X) for m in models]
    return np.mean(probs, axis=0), probs


def poisson_prob_block(models, X, cfg: EngineConfig):
    hx = np.clip(models["home_xg_model"].predict(X), cfg.min_xg, cfg.max_xg)
    ax = np.clip(models["away_xg_model"].predict(X), cfg.min_xg, cfg.max_xg)
    p1x2 = np.asarray([
        [poisson_prediction(h, a, models["rho"], cfg.poisson_max_goals)[k] for k in ("X", "1", "2")]
        for h, a in zip(hx, ax)
    ])
    pgg = np.asarray([
        poisson_prediction(h, a, models["rho"], cfg.poisson_max_goals)["GG"]
        for h, a in zip(hx, ax)
    ])
    return p1x2, pgg, hx, ax


# ============================================================
# SIMPLEX ENSEMBLE + CALIBRATION
# ============================================================

def fit_simplex_weights(prob_blocks: List[np.ndarray], y, classes):
    k = len(prob_blocks)
    if k == 1:
        return np.array([1.0]), float(log_loss(y, prob_blocks[0], labels=classes))

    def softmax(z):
        z = np.asarray(z, dtype=float)
        e = np.exp(z - np.max(z))
        return e / e.sum()

    def objective(z):
        w = softmax(z)
        p = np.sum([w[i] * prob_blocks[i] for i in range(k)], axis=0)
        p = np.clip(p, 1e-9, 1.0)
        p /= p.sum(axis=1, keepdims=True)
        return log_loss(y, p, labels=classes)

    try:
        res = minimize(objective, np.zeros(k), method="BFGS")
        if res.success and np.isfinite(res.fun):
            w = softmax(res.x)
            return w, float(res.fun)
    except Exception:
        pass
    return np.ones(k) / k, float(objective(np.zeros(k)))


def fit_binary_weights(prob_blocks: List[np.ndarray], y):
    k = len(prob_blocks)
    if k == 1:
        return np.array([1.0]), float(log_loss(y, prob_blocks[0], labels=[0, 1]))

    def softmax(z):
        e = np.exp(z - np.max(z))
        return e / e.sum()

    def objective(z):
        w = softmax(z)
        p = np.sum([w[i] * prob_blocks[i] for i in range(k)], axis=0)
        p = np.clip(p, 1e-7, 1 - 1e-7)
        return log_loss(y, p, labels=[0, 1])

    try:
        res = minimize(objective, np.zeros(k), method="BFGS")
        if res.success and np.isfinite(res.fun):
            return softmax(res.x), float(res.fun)
    except Exception:
        pass
    return np.ones(k) / k, float(objective(np.zeros(k)))


def apply_temperature_multiclass(p, t):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    t = max(float(t), 1e-6)
    z = np.log(p) / t
    z -= z.max(axis=1, keepdims=True)
    out = np.exp(z)
    out /= out.sum(axis=1, keepdims=True)
    return out


def fit_temperature_multiclass(p, y, cfg: EngineConfig):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    lp = np.log(p)

    def obj(t_arr):
        t = float(t_arr[0])
        z = lp / t
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z)
        q /= q.sum(axis=1, keepdims=True)
        return log_loss(y, q, labels=[0, 1, 2])

    try:
        res = minimize(obj, [1.0], bounds=[(cfg.temperature_min, cfg.temperature_max)], method="L-BFGS-B")
        if res.success and np.isfinite(res.fun):
            return float(res.x[0])
    except Exception:
        pass
    return 1.0


def apply_class_bias_multiclass(p, bias):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    z = np.log(p) + np.asarray(bias, dtype=float).reshape(1, -1)
    z -= z.max(axis=1, keepdims=True)
    q = np.exp(z); q /= q.sum(axis=1, keepdims=True)
    return q


def fit_class_bias_multiclass(p, y, l2=1.0):
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0)
    lp = np.log(p); y = np.asarray(y, dtype=int)
    def obj(b):
        z = lp + np.asarray(b).reshape(1, -1)
        z -= z.max(axis=1, keepdims=True)
        q = np.exp(z); q /= q.sum(axis=1, keepdims=True)
        return float(log_loss(y, q, labels=[0,1,2]) + l2 * np.sum(np.asarray(b)**2) / max(len(y),1))
    try:
        res = minimize(obj, np.zeros(3), method="BFGS")
        if res.success and np.isfinite(res.fun):
            b = np.asarray(res.x, dtype=float); b -= b.mean(); return b
    except Exception:
        pass
    return np.zeros(3)


def apply_temperature_binary(p, t):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    t = max(float(t), 1e-6)
    logits = np.log(p / (1 - p))
    return np.clip(sigmoid(logits / t), 1e-7, 1 - 1e-7)


def fit_temperature_binary(p, y, cfg: EngineConfig):
    p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
    logits = np.log(p / (1 - p))

    def obj(t_arr):
        t = float(t_arr[0])
        q = sigmoid(logits / t)
        return log_loss(y, q, labels=[0, 1])

    try:
        res = minimize(obj, [1.0], bounds=[(cfg.temperature_min, cfg.temperature_max)], method="L-BFGS-B")
        if res.success and np.isfinite(res.fun):
            return float(res.x[0])
    except Exception:
        pass
    return 1.0


def combine_models(models, X, weights, temps, cfg: EngineConfig):
    ml_result, _ = average_ml_result(models["result_models"], X)
    ml_btts, _ = average_ml_btts(models["btts_models"], X)
    pois_result, pois_btts, hx, ax = poisson_prob_block(models, X, cfg)

    p1 = weights["result"][0] * ml_result + weights["result"][1] * pois_result
    p1 = np.clip(p1, 1e-9, 1.0)
    p1 /= p1.sum(axis=1, keepdims=True)
    p1 = apply_temperature_multiclass(p1, temps["result"])
    if "class_bias" in weights:
        p1 = apply_class_bias_multiclass(p1, weights["class_bias"])

    pgg = weights["btts"][0] * ml_btts + weights["btts"][1] * pois_btts
    pgg = apply_temperature_binary(pgg, temps["btts"])
    return p1, pgg, hx, ax


# ============================================================
# EVALUATION / BASELINES / EMPIRICAL CONFIDENCE
# ============================================================

def evaluate_probs(p1x2, pgg, yr, yb):
    pred = np.argmax(p1x2, axis=1)
    pred_btts = (pgg >= 0.5).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        yr, pred, labels=[0, 1, 2], zero_division=0
    )
    return {
        "n": int(len(yr)),
        "accuracy_1x2": float(accuracy_score(yr, pred)),
        "logloss_1x2": float(log_loss(yr, p1x2, labels=[0, 1, 2])),
        "accuracy_btts": float(accuracy_score(yb, pred_btts)),
        "logloss_btts": float(log_loss(yb, pgg, labels=[0, 1])),
        "brier_btts": float(brier_score_loss(yb, pgg)),
        "precision_X_1_2": precision.tolist(),
        "recall_X_1_2": recall.tolist(),
        "f1_X_1_2": f1.tolist(),
        "confusion": confusion_matrix(yr, pred, labels=[0, 1, 2]).tolist(),
    }


def naive_baselines(yr, yb):
    cls = np.bincount(yr, minlength=3)
    p_res = cls / cls.sum()
    pred_res = int(np.argmax(p_res))
    p_b = float(np.mean(yb))
    pred_b = int(p_b >= 0.5)
    return {
        "majority_1x2_accuracy": float(np.mean(yr == pred_res)),
        "historical_1x2_probs": p_res.tolist(),
        "majority_btts_accuracy": float(np.mean(yb == pred_b)),
        "historical_btts_prob": p_b,
    }


def make_empirical_bins(probabilities, outcomes, n_bins=10):
    probabilities = np.asarray(probabilities, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    bins = []
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (probabilities >= lo) & (probabilities <= hi if i == n_bins - 1 else probabilities < hi)
        n = int(mask.sum())
        if n:
            bins.append({
                "lo": float(lo), "hi": float(hi), "n": n,
                "mean_pred": float(probabilities[mask].mean()),
                "hit_rate": float(outcomes[mask].mean()),
            })
    return bins


def empirical_rate_for_probability(p, bins):
    p = float(p)
    if not bins:
        return None
    for b in bins:
        if b["lo"] <= p <= b["hi"]:
            return float(b["hit_rate"])
    return None


def eval_bad_form_regime(p1x2, pgg, yr, yb, X, feature_names):
    if "both_bad_form" not in feature_names:
        return None
    idx = feature_names.index("both_bad_form")
    mask = X[:, idx] >= 0.5
    if mask.sum() < 5:
        return None
    r = evaluate_probs(p1x2[mask], pgg[mask], yr[mask], yb[mask])
    r["regime"] = "both_bad_form"
    return r


# ============================================================
# ONE FOLD
# ============================================================

def fit_params_on_validation(models, X_val, yr_val, yb_val, cfg):
    ml_r, _ = average_ml_result(models["result_models"], X_val)
    ml_b, _ = average_ml_btts(models["btts_models"], X_val)
    pois_r, pois_b, _, _ = poisson_prob_block(models, X_val, cfg)

    rw, _ = fit_simplex_weights([ml_r, pois_r], yr_val, [0, 1, 2])
    bw, _ = fit_binary_weights([ml_b, pois_b], yb_val)

    p_r = rw[0] * ml_r + rw[1] * pois_r
    p_r = np.clip(p_r, 1e-9, 1.0)
    p_r /= p_r.sum(axis=1, keepdims=True)
    p_b = np.clip(bw[0] * ml_b + bw[1] * pois_b, 1e-7, 1 - 1e-7)

    t_r = fit_temperature_multiclass(p_r, yr_val, cfg)
    t_b = fit_temperature_binary(p_b, yb_val, cfg)
    p_r_cal = apply_temperature_multiclass(p_r, t_r)
    class_bias = fit_class_bias_multiclass(p_r_cal, yr_val, l2=cfg.calibration_l2)

    threshold = 0.50
    if cfg.optimize_btts_threshold:
        candidates = np.arange(cfg.btts_threshold_min, cfg.btts_threshold_max + 1e-9, cfg.btts_threshold_step)
        threshold = float(max(candidates, key=lambda th: accuracy_score(yb_val, (p_b >= th).astype(int))))

    return {
        "weights": {"result": rw.tolist(), "btts": bw.tolist(), "class_bias": class_bias.tolist()},
        "temps": {"result": float(t_r), "btts": float(t_b)},
        "btts_threshold": threshold,
    }


def predict_with_params(models, X, params, cfg):
    return combine_models(
        models,
        X,
        {
            "result": np.asarray(params["weights"]["result"], dtype=float),
            "btts": np.asarray(params["weights"]["btts"], dtype=float),
        },
        params["temps"],
        cfg,
    )


# ============================================================
# ENGINE
# ============================================================

class FootballPredictionEngineV10:
    version = "v10-FINAL"

    def __init__(self, cfg: Optional[EngineConfig] = None):
        self.cfg = cfg or EngineConfig()
        self.engine_state = None
        self.feature_names = None
        self.last_backtest = None

    def _fit_holdout(self, df, X, yr, yb, hg, ag):
        n = len(X)
        i_train = int(n * self.cfg.train_frac)
        i_val = int(n * (self.cfg.train_frac + self.cfg.val_frac))
        test_n = n - i_val
        if i_train < self.cfg.min_train_matches or test_n < self.cfg.min_holdout_test_matches:
            raise ValueError(
                f"Not enough matches for holdout: train={i_train}, test={test_n}. "
                f"Need at least train={self.cfg.min_train_matches}, test={self.cfg.min_holdout_test_matches}."
            )

        models = fit_models(X[:i_train], yr[:i_train], yb[:i_train], hg[:i_train], ag[:i_train], self.cfg)
        params = fit_params_on_validation(models, X[i_train:i_val], yr[i_train:i_val], yb[i_train:i_val], self.cfg)
        p1, pg, _, _ = predict_with_params(models, X[i_val:], params, self.cfg)
        report = evaluate_probs(p1, pg, yr[i_val:], yb[i_val:])
        regime = eval_bad_form_regime(p1, pg, yr[i_val:], yb[i_val:], X[i_val:], self.feature_names)
        report.update({
            "train_matches": int(i_train),
            "validation_matches": int(i_val - i_train),
            "test_matches": int(n - i_val),
            "alpha_result_ml": float(params["weights"]["result"][0]),
            "alpha_result_poisson": float(params["weights"]["result"][1]),
            "alpha_btts_ml": float(params["weights"]["btts"][0]),
            "alpha_btts_poisson": float(params["weights"]["btts"][1]),
            "temperature_1x2": float(params["temps"]["result"]),
            "temperature_btts": float(params["temps"]["btts"]),
            "rho": float(models["rho"]),
            "bad_form_regime": regime,
            "test_start_date": str(df.iloc[i_val]["Date"].date()),
            "test_end_date": str(df.iloc[-1]["Date"].date()),
        })
        return report

    def _walk_forward(self, df, X, yr, yb, hg, ag):
        n = len(X)
        initial = int(n * self.cfg.wf_initial_train_frac)
        val_size = max(25, int(n * self.cfg.wf_val_frac))
        test_size = max(self.cfg.wf_min_test_matches, int(n * self.cfg.wf_test_frac))
        folds = []
        all_probs = []
        all_y = []
        all_btts_p = []
        all_btts_y = []
        train_end = initial
        fold_id = 1

        while train_end + val_size + self.cfg.wf_min_test_matches <= n and fold_id <= self.cfg.wf_max_folds:
            val_start = train_end
            val_end = min(val_start + val_size, n)
            test_start = val_end
            test_end = min(test_start + test_size, n)
            if test_end - test_start < self.cfg.wf_min_test_matches:
                break

            yr_tr, yb_tr = yr[:train_end], yb[:train_end]
            if len(np.unique(yr_tr)) < 3 or len(np.unique(yb_tr)) < 2:
                train_end = test_end
                fold_id += 1
                continue

            models = fit_models(X[:train_end], yr_tr, yb_tr, hg[:train_end], ag[:train_end], self.cfg)
            params = fit_params_on_validation(models, X[val_start:val_end], yr[val_start:val_end], yb[val_start:val_end], self.cfg)
            p1, pg, _, _ = predict_with_params(models, X[test_start:test_end], params, self.cfg)
            rep = evaluate_probs(p1, pg, yr[test_start:test_end], yb[test_start:test_end])
            rep.update({
                "fold": fold_id,
                "train_end": int(train_end),
                "val_size": int(val_end - val_start),
                "test_size": int(test_end - test_start),
                "test_start_date": str(df.iloc[test_start]["Date"].date()),
                "test_end_date": str(df.iloc[test_end - 1]["Date"].date()),
                "weights": params["weights"],
                "temperatures": params["temps"],
                "rho": float(models["rho"]),
                "bad_form_regime": eval_bad_form_regime(
                    p1, pg, yr[test_start:test_end], yb[test_start:test_end],
                    X[test_start:test_end], self.feature_names
                ),
            })
            folds.append(rep)
            all_probs.append(p1)
            all_y.append(yr[test_start:test_end])
            all_btts_p.append(pg)
            all_btts_y.append(yb[test_start:test_end])

            train_end = test_end
            fold_id += 1

        if not folds:
            raise ValueError("Walk-forward produced no valid folds. Need more historical matches.")

        total_n = sum(f["n"] for f in folds)
        weighted = {
            "accuracy_1x2": sum(f["accuracy_1x2"] * f["n"] for f in folds) / total_n,
            "logloss_1x2": sum(f["logloss_1x2"] * f["n"] for f in folds) / total_n,
            "accuracy_btts": sum(f["accuracy_btts"] * f["n"] for f in folds) / total_n,
            "logloss_btts": sum(f["logloss_btts"] * f["n"] for f in folds) / total_n,
            "brier_btts": sum(f["brier_btts"] * f["n"] for f in folds) / total_n,
        }

        p_all = np.vstack(all_probs)
        y_all = np.concatenate(all_y)
        pg_all = np.concatenate(all_btts_p)
        yb_all = np.concatenate(all_btts_y)
        # Empirical hit-rate calibration for the most probable 1X2 class.
        top_prob = p_all.max(axis=1)
        top_hit = (np.argmax(p_all, axis=1) == y_all).astype(int)
        empirical_1x2 = make_empirical_bins(top_prob, top_hit, 10)
        empirical_btts = make_empirical_bins(pg_all, yb_all, 10)

        return {
            "folds": folds,
            "n_total_test": int(total_n),
            "weighted": weighted,
            "empirical_confidence_1x2": empirical_1x2,
            "empirical_confidence_btts": empirical_btts,
            # Kept only for post-backtest 7-match diagnostics. These are
            # strictly out-of-sample walk-forward predictions.
            "_raw_oos": {
                "p1": p_all.tolist(),
                "y1": y_all.tolist(),
                "pg": pg_all.tolist(),
                "yb": yb_all.tolist(),
            },
        }

    def backtest(self, path_or_df, default_league="DEFAULT"):
        df = load_data(path_or_df, default_league=default_league)
        if len(df) < self.cfg.min_train_matches + self.cfg.min_holdout_test_matches:
            raise ValueError(f"Dataset has only {len(df)} matches; use a larger historical dataset.")
        if len(np.unique(df["Result"])) < 3 or len(np.unique(df["BTTS"])) < 2:
            raise ValueError("Dataset must contain 1/X/2 and GG/NG examples.")

        X, yr, yb, hg, ag, *states = build_features(df, self.cfg)
        self.feature_names = build_feature_names(df.attrs.get("present_stats", {}).keys())
        holdout = self._fit_holdout(df, X, yr, yb, hg, ag)
        wf = self._walk_forward(df, X, yr, yb, hg, ag) if self.cfg.run_walk_forward else None
        baseline = naive_baselines(yr, yb)

        report = {
            "version": self.version,
            "rows": int(len(df)),
            "date_start": str(df["Date"].min().date()),
            "date_end": str(df["Date"].max().date()),
            "leagues": sorted(df["League"].unique().tolist()),
            "optional_stats": sorted(df.attrs.get("present_stats", {}).keys()),
            "feature_count": int(len(self.feature_names)),
            "holdout": holdout,
            "walk_forward": wf,
            "baseline": baseline,
            "has_xgb": HAS_XGB and self.cfg.use_xgb,
            "has_lgbm": HAS_LGBM and self.cfg.use_lgbm,
        }
        self.last_backtest = report
        return report

    def fit_production(self, path_or_df, params_from_backtest=None, default_league="DEFAULT"):
        df = load_data(path_or_df, default_league=default_league)
        X, yr, yb, hg, ag, elo, form, rest, h2h, league_stats, stat_trackers = build_features(df, self.cfg)
        self.feature_names = build_feature_names(df.attrs.get("present_stats", {}).keys())
        if len(np.unique(yr)) < 3 or len(np.unique(yb)) < 2:
            raise ValueError("Production dataset must contain 1/X/2 and GG/NG.")

        production_models = fit_models(X, yr, yb, hg, ag, self.cfg)

        # Preferred: use median parameters learned only from unseen WF validation/test process.
        if params_from_backtest is None and self.last_backtest and self.last_backtest.get("walk_forward"):
            folds = self.last_backtest["walk_forward"]["folds"]
            wr = np.array([f["weights"]["result"] for f in folds], dtype=float)
            wb = np.array([f["weights"]["btts"] for f in folds], dtype=float)
            tr = np.array([f["temperatures"]["result"] for f in folds], dtype=float)
            tb = np.array([f["temperatures"]["btts"] for f in folds], dtype=float)
            params = {
                "weights": {
                    "result": np.median(wr, axis=0).tolist(),
                    "btts": np.median(wb, axis=0).tolist(),
                    "class_bias": np.median(np.asarray([f["weights"].get("class_bias", [0.0,0.0,0.0]) for f in folds], dtype=float), axis=0).tolist(),
                },
                "temps": {"result": float(np.median(tr)), "btts": float(np.median(tb))},
                "btts_threshold": float(np.median([f.get("btts_threshold", 0.5) for f in folds])),
            }
        elif params_from_backtest is not None:
            params = params_from_backtest
        else:
            # Neutral weights; no in-sample tuning.
            params = {
                "weights": {"result": [0.5, 0.5], "btts": [0.5, 0.5], "class_bias": [0.0,0.0,0.0]},
                "temps": {"result": 1.0, "btts": 1.0},
                "btts_threshold": 0.5,
            }

        empirical = None
        if self.last_backtest and self.last_backtest.get("walk_forward"):
            empirical = {
                "1x2": self.last_backtest["walk_forward"].get("empirical_confidence_1x2", []),
                "btts": self.last_backtest["walk_forward"].get("empirical_confidence_btts", []),
            }

        self.engine_state = {
            "version": self.version,
            "config": asdict(self.cfg),
            "feature_names": self.feature_names,
            "result_models": production_models["result_models"],
            "btts_models": production_models["btts_models"],
            "home_xg_model": production_models["home_xg_model"],
            "away_xg_model": production_models["away_xg_model"],
            "rho": float(production_models["rho"]),
            "weights": params["weights"],
            "temperatures": params["temps"],
            "btts_threshold": float(params.get("btts_threshold", 0.5)),
            "elo": elo,
            "form": form,
            "rest": rest,
            "h2h": h2h,
            "league_stats": league_stats,
            "stat_trackers": stat_trackers,
            "last_date": str(df["Date"].max().date()),
            "empirical_confidence": empirical,
            "optional_stats": sorted(df.attrs.get("present_stats", {}).keys()),
        }
        return self.engine_state

    def fit(self, path_or_df, default_league="DEFAULT"):
        """Recommended full route: backtest first, then production refit."""
        report = self.backtest(path_or_df, default_league=default_league)
        self.fit_production(path_or_df, default_league=default_league)
        return report

    def predict_match(self, home, away, match_date=None, league="DEFAULT",
                      season=None, home_availability=0.0, away_availability=0.0):
        if self.engine_state is None:
            raise RuntimeError("Engine is not fitted. Run engine.fit(...) or engine.fit_production(...).")
        if match_date is None:
            match_date = pd.Timestamp.today().normalize()
        else:
            match_date = pd.Timestamp(match_date)
        if season is None:
            season = _season_fallback(match_date)

        state = self.engine_state
        features = np.asarray([assemble_features(
            state["elo"], state["form"], state["rest"], state["h2h"], state["league_stats"],
            state["stat_trackers"], league, season, home, away, match_date,
        )], dtype=float)

        p1, pgg, hx, ax = predict_with_params(
            {
                "result_models": state["result_models"],
                "btts_models": state["btts_models"],
                "home_xg_model": state["home_xg_model"],
                "away_xg_model": state["away_xg_model"],
                "rho": state["rho"],
            },
            features,
            {
                "weights": state["weights"],
                "temps": state["temperatures"],
            },
            self.cfg,
        )

        # Optional future-only manual availability adjustment.
        ha = float(np.clip(home_availability, -1.0, 0.0))
        aa = float(np.clip(away_availability, -1.0, 0.0))
        home_xg = float(np.clip(hx[0] * (1.0 + 0.15 * ha), self.cfg.min_xg, self.cfg.max_xg))
        away_xg = float(np.clip(ax[0] * (1.0 + 0.15 * aa), self.cfg.min_xg, self.cfg.max_xg))
        # Rebuild Poisson component after availability shift, then recombine with ML.
        pois = poisson_prediction(home_xg, away_xg, state["rho"], self.cfg.poisson_max_goals)
        ml_r, _ = average_ml_result(state["result_models"], features)
        ml_b, _ = average_ml_btts(state["btts_models"], features)
        final_r = state["weights"]["result"][0] * ml_r[0] + state["weights"]["result"][1] * np.array([pois["X"], pois["1"], pois["2"]])
        final_r = apply_temperature_multiclass((final_r / final_r.sum()).reshape(1, -1), state["temperatures"]["result"])
        final_r = apply_class_bias_multiclass(final_r, state["weights"].get("class_bias", [0.0,0.0,0.0]))[0]
        final_g = float(apply_temperature_binary(np.array([
            state["weights"]["btts"][0] * ml_b[0] + state["weights"]["btts"][1] * pois["GG"]
        ]), state["temperatures"]["btts"])[0])

        labels = {"1": float(final_r[1]), "X": float(final_r[0]), "2": float(final_r[2])}
        ranked = sorted(labels.items(), key=lambda kv: kv[1], reverse=True)
        prediction = ranked[0][0]
        margin = float((ranked[0][1] - ranked[1][1]) * 100.0)
        entropy = entropy3(final_r)
        empirical = None
        conf_bins = (state.get("empirical_confidence") or {}).get("1x2")
        if conf_bins:
            empirical = empirical_rate_for_probability(ranked[0][1], conf_bins)

        return {
            "home": _clean_team(home),
            "away": _clean_team(away),
            "league": _clean_league(league),
            "1": round(labels["1"] * 100, 2),
            "X": round(labels["X"] * 100, 2),
            "2": round(labels["2"] * 100, 2),
            "GG": round(final_g * 100, 2),
            "NG": round((1.0 - final_g) * 100, 2),
            "Prediction": prediction,
            "BTTS": "GG" if final_g >= state.get("btts_threshold", 0.5) else "NG",
            "TopProbability": round(ranked[0][1] * 100, 2),
            "Margin": round(margin, 2),
            "ProbabilityEntropy": round(entropy, 4),
            "HistoricalBinHitRate": None if empirical is None else round(empirical * 100, 2),
            "xG_Home": round(home_xg, 3),
            "xG_Away": round(away_xg, 3),
            "rho": round(float(state["rho"]), 5),
            "availability_home": round(ha, 3),
            "availability_away": round(aa, 3),
        }

    def feature_importance(self, top_n=20):
        if self.engine_state is None:
            return []
        rows = []
        for model in self.engine_state.get("result_models", []):
            try:
                inner = model.named_steps["model"]
                imp = getattr(inner, "feature_importances_", None)
                if imp is not None:
                    rows.append(np.asarray(imp, dtype=float))
            except Exception:
                pass
        if not rows:
            return []
        imp = np.mean(rows, axis=0)
        order = np.argsort(-imp)[:top_n]
        return [{"feature": self.feature_names[int(i)], "importance": float(imp[i])} for i in order]

    @staticmethod
    def normalize_market_odds(home_odds, draw_odds, away_odds):
        odds = np.asarray([_safe_float(home_odds), _safe_float(draw_odds), _safe_float(away_odds)], dtype=float)
        if np.any(~np.isfinite(odds)) or np.any(odds <= 1.0):
            raise ValueError("Decimal odds must be > 1.0")
        inv = 1.0 / odds
        return inv / inv.sum()

    def predict_match_with_market(self, home, away, home_odds, draw_odds, away_odds, market_weight=None, **kwargs):
        base = self.predict_match(home, away, **kwargs)
        market = self.normalize_market_odds(home_odds, draw_odds, away_odds)
        w = float(self.cfg.market_blend_weight if market_weight is None else market_weight)
        w = float(np.clip(w, 0.0, 1.0))
        model_p = np.asarray([base["X"], base["1"], base["2"]], dtype=float) / 100.0
        p = (1.0-w)*model_p + w*market; p /= p.sum()
        labels = {"1": p[1]*100, "X": p[0]*100, "2": p[2]*100}
        base.update({"1": round(labels["1"],2), "X": round(labels["X"],2), "2": round(labels["2"],2),
                     "Prediction": max(labels, key=labels.get), "MarketBlendWeight": round(w,3),
                     "Market_1": round(market[1]*100,2), "Market_X": round(market[0]*100,2), "Market_2": round(market[2]*100,2)})
        return base

    def save(self, path="engine_v8_hybrid.joblib"):
        joblib.dump(self, path)
        return path

    @staticmethod
    def load(path="engine_v10_final.joblib"):
        return joblib.load(path)


# V10 public engine: obsolete V7/V8 aliases are intentionally removed.

# ============================================================
# REPORTING
# ============================================================

def print_report(report):
    print("\n" + "=" * 70)
    print("FOOTBALL PREDICTION ENGINE v10 FINAL — BACKTEST")
    print("=" * 70)
    print(f"Rows: {report['rows']}")
    print(f"Date: {report['date_start']} -> {report['date_end']}")
    print(f"Leagues: {len(report['leagues'])}")
    print(f"Features: {report['feature_count']}")
    print("Optional stats:", ", ".join(report["optional_stats"]) or "none")
    print("\nBASELINE")
    print(f"Majority 1/X/2 accuracy: {report['baseline']['majority_1x2_accuracy']*100:.2f}%")
    print(f"Majority GG/NG accuracy: {report['baseline']['majority_btts_accuracy']*100:.2f}%")

    h = report["holdout"]
    print("\nFINAL HOLDOUT")
    print(f"Train={h['train_matches']} | Val={h['validation_matches']} | Test={h['test_matches']}")
    print(f"1/X/2 Accuracy={h['accuracy_1x2']*100:.2f}% | LogLoss={h['logloss_1x2']:.4f}")
    print(f"GG/NG Accuracy={h['accuracy_btts']*100:.2f}% | LogLoss={h['logloss_btts']:.4f} | Brier={h['brier_btts']:.4f}")
    print(f"rho={h['rho']:.4f} | T(1X2)={h['temperature_1x2']:.3f} | T(GG)={h['temperature_btts']:.3f}")
    print("Confusion [X,1,2]:")
    print(np.asarray(h["confusion"]))

    if report.get("walk_forward"):
        w = report["walk_forward"]
        print("\nWALK-FORWARD")
        print(f"Folds={len(w['folds'])} | Unseen test matches={w['n_total_test']}")
        print(f"1/X/2 Accuracy={w['weighted']['accuracy_1x2']*100:.2f}% | LogLoss={w['weighted']['logloss_1x2']:.4f}")
        print(f"GG/NG Accuracy={w['weighted']['accuracy_btts']*100:.2f}% | LogLoss={w['weighted']['logloss_btts']:.4f} | Brier={w['weighted']['brier_btts']:.4f}")

    print("=" * 70 + "\n")


# ============================================================
# CLI EXAMPLE
# ============================================================



# ============================================================
# 7-MATCH FINAL INTERFACE
# ============================================================

class SevenMatchPredictionEngineV10:
    """Final user-facing V10 engine.

    Contract: the prediction CSV must contain exactly 7 fixtures.
    The engine never selects/replaces fixtures and preserves their order.
    """
    VERSION = "V10-7-MATCH-FINAL-1.0"

    def __init__(self, cfg=None):
        self.cfg = cfg or EngineConfig()
        self.base = FootballPredictionEngineV10(self.cfg)
        self.fitted = False

    def fit(self, history_path, default_league="DEFAULT", run_backtest=True):
        if run_backtest:
            report = self.base.fit(history_path, default_league=default_league)
        else:
            self.base.fit_production(history_path, default_league=default_league)
            report = None
        self.fitted = True
        try:
            _h = load_data(history_path, default_league)
            self.known_teams = set(_h["HomeTeam"]) | set(_h["AwayTeam"])
        except Exception:
            self.known_teams = None
        return report

    @staticmethod
    def _read_exactly_7(fixtures_path):
        fx = pd.read_csv(fixtures_path)
        fx = fx.dropna(how="all").copy()
        if len(fx) != 7:
            raise ValueError(f"Expected exactly 7 fixtures, got {len(fx)}")
        needed = {"HomeTeam", "AwayTeam"}
        missing = sorted(needed - set(fx.columns))
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))
        for c in ["HomeTeam", "AwayTeam"]:
            fx[c] = fx[c].astype(str).str.strip()
            if (fx[c] == "").any() or (fx[c].str.lower() == "nan").any():
                raise ValueError(f"Column {c} contains an empty fixture")
        return fx.reset_index(drop=True)

    @staticmethod
    def _band(p):
        p = float(p)
        if p >= 0.82: return "A+"
        if p >= 0.76: return "A"
        if p >= 0.70: return "B"
        if p >= 0.62: return "C"
        return "D"

    def predict_7(self, fixtures):
        if not self.fitted or self.base.engine_state is None:
            raise RuntimeError("Model is not fitted. Train it first.")
        if not isinstance(fixtures, pd.DataFrame):
            fixtures = pd.DataFrame(fixtures)
        fixtures = fixtures.copy()
        if len(fixtures) != 7:
            raise ValueError(f"Expected exactly 7 fixtures, got {len(fixtures)}")
        required = {"HomeTeam", "AwayTeam"}
        missing = sorted(required - set(fixtures.columns))
        if missing:
            raise ValueError("Missing required columns: " + ", ".join(missing))

        rows=[]
        for i,r in fixtures.reset_index(drop=True).iterrows():
            league = r.get("League", r.get("Div", "DEFAULT"))
            season = r.get("Season", None)
            dt = r.get("Date", None)
            if pd.isna(dt): dt = None
            def num(name):
                try:
                    v=float(r.get(name,0.0)); return 0.0 if not np.isfinite(v) else v
                except Exception: return 0.0
            pred=self.base.predict_match(
                r["HomeTeam"], r["AwayTeam"], match_date=dt, league=league, season=season,
                home_availability=num("home_availability"), away_availability=num("away_availability")
            )
            p1x2=float(pred["TopProbability"])/100.0
            pgg=max(float(pred["GG"]),float(pred["NG"]))/100.0
            pred["MatchNo"]=i+1
            pred["1X2_Confidence"]=round(p1x2*100,2)
            pred["GGNG_Confidence"]=round(pgg*100,2)
            pred["1X2_Grade"]=self._band(p1x2)
            pred["GGNG_Grade"]=self._band(pgg)
            rows.append(pred)
        out=pd.DataFrame(rows)
        kt=getattr(self,"known_teams",None)
        if kt is not None:
            fx=fixtures.reset_index(drop=True)
            out["Coverage"]=["OK" if (_clean_team(fx.loc[i,"HomeTeam"]) in kt and _clean_team(fx.loc[i,"AwayTeam"]) in kt) else "NO_DATA" for i in range(len(fx))]
        # The two confidence columns are rankings only; they are NOT guarantees.
        out["PriorityScore"]=((out["1X2_Confidence"]/100.0)+(out["GGNG_Confidence"]/100.0))/2.0
        return out

    def predict_7_from_csv(self, fixtures_path):
        fx=self._read_exactly_7(fixtures_path)
        return self.predict_7(fx)

    def seven_batch_report(self):
        """Summarize consecutive out-of-sample predictions in blocks of 7.

        This is a diagnostic only, not a guarantee for future betting slips.
        It does not tune the model.
        """
        rep=self.base.last_backtest or {}
        wf=rep.get("walk_forward")
        if not wf or "_raw_oos" not in wf:
            return None
        p1=np.asarray(wf["_raw_oos"]["p1"])
        y1=np.asarray(wf["_raw_oos"]["y1"])
        pg=np.asarray(wf["_raw_oos"]["pg"])
        yb=np.asarray(wf["_raw_oos"]["yb"])
        n=min(len(p1),len(pg))
        groups=[]
        for start in range(0,n,7):
            end=min(start+7,n)
            if end-start<7: break
            c1=int((np.argmax(p1[start:end],axis=1)==y1[start:end]).sum())
            cg=int(((pg[start:end]>=0.5).astype(int)==yb[start:end]).sum())
            groups.append({"batch":len(groups)+1,"matches":7,"correct_1x2":c1,"accuracy_1x2":c1/7,
                           "correct_ggng":cg,"accuracy_ggng":cg/7})
        if not groups: return {"batches":[]}
        return {
            "batches":groups,
            "n_batches":len(groups),
            "mean_correct_1x2":float(np.mean([g["correct_1x2"] for g in groups])),
            "mean_correct_ggng":float(np.mean([g["correct_ggng"] for g in groups])),
            "perfect_7of7_1x2":int(sum(g["correct_1x2"]==7 for g in groups)),
            "perfect_7of7_ggng":int(sum(g["correct_ggng"]==7 for g in groups)),
        }

    def save(self,path="football_prediction_engine_v10_7match_final.joblib"):
        joblib.dump(self,path)
        return path

    @staticmethod
    def load(path):
        return joblib.load(path)


# Alias: the CLI and auto mode below use this name.
SevenMatchPredictionEngine = SevenMatchPredictionEngineV10


def print_7_predictions(df):
    show=["MatchNo","home","away","Prediction","1","X","2","GG","NG",
          "1X2_Confidence","GGNG_Confidence","1X2_Grade","GGNG_Grade","xG_Home","xG_Away","Coverage"]
    show=[c for c in show if c in df.columns]
    print("\n"+"="*112)
    print("7-MATCH FINAL PREDICTION")
    print("="*112)
    print(df[show].to_string(index=False))
    if "Coverage" in df.columns and (df["Coverage"]=="NO_DATA").any():
        print("\nWARNING: NO_DATA = the engine has no history for one of these teams. Its numbers there are generic, ignore them.")
    print("\nIMPORTANT: confidence is calibrated probability, not a guarantee.\n")


def print_7_backtest(report):
    if not report: return
    print("\n"+"="*72)
    print("7-MATCH WALK-FORWARD DIAGNOSTIC")
    print("="*72)
    print(f"Batches of 7: {report.get('n_batches',0)}")
    print(f"Mean correct per 7 — 1X2: {report.get('mean_correct_1x2',0):.2f}/7")
    print(f"Mean correct per 7 — GG/NG: {report.get('mean_correct_ggng',0):.2f}/7")
    print(f"Perfect 7/7 batches — 1X2: {report.get('perfect_7of7_1x2',0)}")
    print(f"Perfect 7/7 batches — GG/NG: {report.get('perfect_7of7_ggng',0)}")
    print("="*72)


# ============================================================
# AUTO HISTORICAL DATA PIPELINE
# ============================================================

FOOTBALL_DATA_BASE = "https://www.football-data.co.uk/mmz4281"

# Main leagues supported by Football-Data.co.uk.
# Add/remove codes without changing the engine itself.
AUTO_LEAGUES = {
    "E0": "EPL",
    "E1": "Championship",
    "E2": "League1",
    "E3": "League2",
    "D1": "Bundesliga",
    "D2": "Bundesliga2",
    "I1": "SerieA",
    "I2": "SerieB",
    "SP1": "LaLiga",
    "SP2": "LaLiga2",
    "F1": "Ligue1",
    "F2": "Ligue2",
    "N1": "Eredivisie",
    "P1": "Portugal",
    "B1": "Belgium",
    "T1": "Turkey",
    "G1": "Greece",
    "SC0": "ScotlandPrem",
    "SC1": "ScotlandChamp",
}

def _season_code(year_start: int) -> str:
    """2025 -> 2526, representing the 2025/26 season."""
    return f"{year_start % 100:02d}{(year_start + 1) % 100:02d}"

def _download_csv(url: str, timeout: int = 30) -> pd.DataFrame:
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 FootballPredictionEngine/10"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        raw = r.read()
    # Football-Data files are generally cp1252/latin-1 encoded.
    for enc in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc)
        except Exception:
            pass
    raise ValueError(f"Could not decode CSV: {url}")

def download_historical_data(
    output_csv: str = "history.csv",
    seasons: int = 10,
    leagues: Optional[Dict[str, str]] = None,
    min_year: Optional[int] = None,
) -> pd.DataFrame:
    """
    Download recent historical match results automatically.

    The downloader is deliberately restricted to completed seasons. It does
    not invent results and silently skips unavailable season/league URLs.
    """
    leagues = leagues or AUTO_LEAGUES

    # Current season is determined from the system date. We only request
    # seasons whose season-start year is strictly before the current one.
    current_year = pd.Timestamp.now().year
    current_month = pd.Timestamp.now().month
    # Football seasons normally start around Aug; before Aug, the latest
    # completed season started two calendar years ago.
    # Include the CURRENT season: the Football-Data file lists only matches already played.
    latest_completed_start = current_year if current_month >= 8 else current_year - 1
    first_year = min_year if min_year is not None else latest_completed_start - seasons + 1

    frames = []
    errors = []

    for year in range(first_year, latest_completed_start + 1):
        code = _season_code(year)
        for league_code, league_name in leagues.items():
            url = f"{FOOTBALL_DATA_BASE}/{code}/{league_code}.csv"
            try:
                df = _download_csv(url)
                if df is None or df.empty:
                    continue

                # Native Football-Data columns -> engine schema.
                rename = {
                    "FTHG": "HomeGoals",
                    "FTAG": "AwayGoals",
                    "HomeTeam": "HomeTeam",
                    "AwayTeam": "AwayTeam",
                }
                df = df.rename(columns=rename)

                required = {"Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"}
                if not required.issubset(df.columns):
                    errors.append(f"{code}/{league_code}: missing required columns")
                    continue

                df["League"] = league_name
                df["Season"] = f"{year}/{str(year + 1)[-2:]}"
                df["SourceLeague"] = league_code
                df["SourceURL"] = url
                frames.append(df)

                print(f"[AUTO-DATA] {code} {league_name}: {len(df)} matches")
            except Exception as exc:
                # 404/unavailable leagues or seasons are expected; continue.
                errors.append(f"{code}/{league_code}: {exc}")

    if not frames:
        raise RuntimeError(
            "No historical data could be downloaded. "
            "Check internet access and Football-Data.co.uk availability."
        )

    out = pd.concat(frames, ignore_index=True, sort=False)

    out["Date"] = pd.to_datetime(out["Date"], dayfirst=True, errors="coerce")
    out["HomeGoals"] = pd.to_numeric(out["HomeGoals"], errors="coerce")
    out["AwayGoals"] = pd.to_numeric(out["AwayGoals"], errors="coerce")

    out = out.dropna(subset=["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals"])
    out = out[(out["HomeGoals"] >= 0) & (out["AwayGoals"] >= 0)]
    out = out.drop_duplicates(
        subset=["Date", "HomeTeam", "AwayTeam", "HomeGoals", "AwayGoals", "League"]
    )
    out = out.sort_values(["Date", "League", "HomeTeam", "AwayTeam"]).reset_index(drop=True)

    out.to_csv(output_csv, index=False)
    print(f"\n[AUTO-DATA] Saved {len(out):,} matches -> {output_csv}")
    print(f"[AUTO-DATA] Date range: {out['Date'].min().date()} -> {out['Date'].max().date()}")
    print(f"[AUTO-DATA] Leagues: {out['League'].nunique()}")

    if errors:
        print(f"[AUTO-DATA] Skipped/unavailable sources: {len(errors)}")

    return out

def auto_train_and_evaluate(
    history_csv: str = "history.csv",
    model_path: str = "football_prediction_engine_v10_auto.joblib",
    seasons: int = 10,
    no_wf: bool = False,
) -> None:
    """
    End-to-end mode:
      1) download historical data
      2) save history.csv
      3) chronological backtest / walk-forward
      4) print accuracy diagnostics
      5) refit production model
      6) save .joblib
    """
    download_historical_data(
        output_csv=history_csv,
        seasons=seasons,
    )

    cfg = EngineConfig(run_walk_forward=not no_wf)
    engine = SevenMatchPredictionEngine(cfg)

    print("\n[AUTO-MODE] Running chronological backtest...")
    report = engine.fit(
        history_csv,
        default_league="DEFAULT",
        run_backtest=not no_wf,
    )

    if report is not None:
        print_report(report)
        batch = engine.seven_batch_report()
        print_7_backtest(batch)

        # Compact machine-readable accuracy summary.
        summary = {
            "history_csv": history_csv,
            "model_path": model_path,
            "matches": int(len(pd.read_csv(history_csv))),
            "walk_forward_enabled": not no_wf,
            "seven_match_report": batch,
        }
        pd.DataFrame([summary]).to_json(
            "v10_auto_accuracy.json",
            orient="records",
            indent=2,
        )
        print("\n[AUTO-MODE] Saved accuracy summary: v10_auto_accuracy.json")

    engine.save(model_path)
    print(f"[AUTO-MODE] Saved production model: {model_path}")


# ============================================================
# FINAL CLI
# ============================================================

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser(description="Football Prediction Engine — V10 FINAL 7-MATCH MODE")
    sub=parser.add_subparsers(dest="cmd", required=True)

    ptrain=sub.add_parser("train", help="Train on historical data")
    ptrain.add_argument("--history", required=True)
    ptrain.add_argument("--model", default="football_prediction_engine_v10_7match_final.joblib")
    ptrain.add_argument("--default-league", default="DEFAULT")
    ptrain.add_argument("--no-wf", action="store_true")

    pp=sub.add_parser("predict", help="Predict exactly 7 user-supplied fixtures")
    pp.add_argument("--model", required=True)
    pp.add_argument("--fixtures", required=True)
    pp.add_argument("--output", default="seven_predictions.csv")

    pbt=sub.add_parser("backtest", help="Run full chronological backtest + 7-match diagnostic")
    pbt.add_argument("--history", required=True)
    pbt.add_argument("--default-league", default="DEFAULT")
    pbt.add_argument("--no-wf", action="store_true")

    pa=sub.add_parser("auto", help="Download historical data, backtest, evaluate and train automatically")
    pa.add_argument("--history", default="history.csv")
    pa.add_argument("--model", default="football_prediction_engine_v10_auto.joblib")
    pa.add_argument("--seasons", type=int, default=10)
    pa.add_argument("--no-wf", action="store_true")

    args=parser.parse_args()

    if args.cmd=="train":
        cfg=EngineConfig(run_walk_forward=not args.no_wf)
        m=SevenMatchPredictionEngine(cfg)
        report=m.fit(args.history, default_league=args.default_league, run_backtest=not args.no_wf)
        if report is not None: print_report(report)
        m.save(args.model)
        print(f"Saved model: {args.model}")
    elif args.cmd=="predict":
        m=SevenMatchPredictionEngine.load(args.model)
        out=m.predict_7_from_csv(args.fixtures)
        out.to_csv(args.output,index=False)
        print_7_predictions(out)
        print(f"Saved predictions: {args.output}")
    elif args.cmd=="backtest":
        cfg=EngineConfig(run_walk_forward=not args.no_wf)
        m=SevenMatchPredictionEngine(cfg)
        report=m.fit(args.history, default_league=args.default_league, run_backtest=not args.no_wf)
        if report is not None:
            print_report(report)
            print_7_backtest(m.seven_batch_report())
    elif args.cmd=="auto":
        auto_train_and_evaluate(
            history_csv=args.history,
            model_path=args.model,
            seasons=args.seasons,
            no_wf=args.no_wf,
        )
