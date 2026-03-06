"""
phase_3/adapters/model_adapter.py
==================================
Bridges the existing phase_2 trained model to 2026 inputs.

THE CORE PROBLEM THIS SOLVES
-----------------------------
Your trained model has LabelEncoders that were fitted on 2018-2025 data.
They know "VER", "HAM", "LEC" — but not "HAD" (Hadjar), "ANT" (Antonelli),
or "CAD" (Cadillac). They know years 2021-2025 — not 2026. If you pass an
unknown label to a LabelEncoder it raises a ValueError and crashes.

THEORY — The Adapter Pattern
------------------------------
Rather than retraining the model (which would lose all the carefully tuned
weights in best_lap_time_predictor.pth), we intercept inputs BEFORE they
hit the model and translate them into something the model already understands.

This is exactly the Adapter design pattern from software engineering:
    [2026 Input] → [Adapter] → [2025-format Input] → [Existing Model] → [Prediction]

The model never knows it's being asked about 2026. It thinks it's seeing
a 2025 race with known drivers. The adapter handles all the translation.

HOW EACH GAP IS BRIDGED
------------------------
GAP 1: year=2026
    The le_year encoder doesn't know 2026.
    Fix: map 2026 → 2025. The model's year embedding learned "recent season"
    patterns. 2025 is the closest proxy. The 2026 scaling factor applied
    afterwards corrects for regulation-change pace delta.

GAP 2: New drivers (Hadjar, Antonelli, Bortoleto, Colapinto, Bearman)
    Fix: map each to their closest historical analogue in the encoder.
    We pick the analogue based on similar career stage + team tier.
    THEORY: This is called "warm-start initialisation" — we give the model
    a meaningful prior rather than a random one.

GAP 3: New team "Cadillac"
    Fix: map to the closest historical midfield team in the encoder.
    Cadillac is a brand-new constructor — closest profile is a well-funded
    midfield team in their first full season (e.g. Racing Point 2019).

GAP 4: New circuit "Madrid Grand Prix"
    Fix: map to a street-hybrid circuit with similar lap count + layout.
    Albert Park (Australia) is the closest match in the encoder.

GAP 5: Perez + Bottas at Cadillac (known drivers, new team context)
    Fix: driver encoding is fine (they're known). Team is mapped via GAP 3.
    The pace_anchor.py testing data provides the actual Cadillac baseline.
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import torch
import joblib

# ── Canonical hardness weights — single source of truth ──────────────────────
# Imported here so _predict_rival_lap_time uses the EXACT same deg_score
# formula that retrain_no_leakage.py used during training.
# Do NOT redefine COMPOUND_HARDNESS inline anywhere in this file.
_HERE_ADAPTER = os.path.dirname(os.path.abspath(__file__))
_PHASE3_CORE  = os.path.abspath(os.path.join(_HERE_ADAPTER, "../core"))
if _PHASE3_CORE not in sys.path:
    sys.path.insert(0, _PHASE3_CORE)
from pace_anchor import COMPOUND_HARDNESS  # noqa: E402
from pace_anchor import PIT_LOSS_BY_CIRCUIT  # noqa: E402

# ── Path setup ────────────────────────────────────────────────────────────────
# Ensures both the project root (f1/) and phase_2 are importable
_HERE         = os.path.dirname(os.path.abspath(__file__))   # backend/phase_3/adapters/
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../../.."))  # f1-strategy-app/
PHASE2_ROOT   = os.path.join(_PROJECT_ROOT, "backend/phase_2")

for _p in [_PROJECT_ROOT, PHASE2_ROOT]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Load phase_2 artifacts ────────────────────────────────────────────────────
# These paths match your actual project structure exactly
_PREP_PATH  = os.path.join(_PROJECT_ROOT, "data/trained_models/f1_preprocessors.joblib")
_MODEL_PATH = os.path.join(_PROJECT_ROOT, "data/trained_models/best_lap_time_predictor.pth")
_DATA_PATH  = os.path.join(_PROJECT_ROOT, "data/feature_data/features_dataset_with_targets.csv")

_preprocessors  = joblib.load(_PREP_PATH)
_historical_df  = pd.read_csv(_DATA_PATH)
_device         = torch.device("cpu")

# Re-instantiate the model architecture (identical copy from phase_2)
# THEORY: We can't import from main.py directly because it starts FastAPI
# on import. So we redeclare the architecture here — safe because the
# weights are loaded from the .pth file, not re-trained.
import torch.nn as nn

class _F1LapTimePredictor(nn.Module):
    def __init__(self, num_gps, num_drivers, num_teams,
                 num_numerical_features, num_years, num_weather_conditions):
        super().__init__()
        self.gp_emb      = nn.Embedding(num_gps,               16)
        self.driver_emb  = nn.Embedding(num_drivers,           16)
        self.team_emb    = nn.Embedding(num_teams,             16)
        self.year_emb    = nn.Embedding(num_years,              4)
        self.weather_emb = nn.Embedding(num_weather_conditions, 4)
        input_dim = 16 + 16 + 16 + 4 + 4 + num_numerical_features
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.SiLU(), nn.Dropout(0.1),
            nn.Linear(64, 32),         nn.BatchNorm1d(32),  nn.SiLU(),
            nn.Linear(32, 1),
        )

    def forward(self, gp_idx, driver_idx, team_idx, year_idx,
                weather_idx, numerical_features):
        x = torch.cat([
            self.gp_emb(gp_idx), self.driver_emb(driver_idx),
            self.team_emb(team_idx), self.year_emb(year_idx),
            self.weather_emb(weather_idx), numerical_features,
        ], dim=1)
        return self.net(x)

_model = _F1LapTimePredictor(
    num_gps               = len(_preprocessors["le_gp"].classes_),
    num_drivers           = len(_preprocessors["le_driver"].classes_),
    num_teams             = len(_preprocessors["le_team"].classes_),
    num_numerical_features= len(_preprocessors["num_cols"]),
    num_years             = len(_preprocessors["le_year"].classes_),
    num_weather_conditions= len(_preprocessors["le_weather"].classes_),
)
_model.load_state_dict(torch.load(_MODEL_PATH, map_location=_device))
_model.eval()

print(f"[ModelAdapter] ✓ Loaded model from {_MODEL_PATH}")
print(f"[ModelAdapter] ✓ Known drivers : {len(_preprocessors['le_driver'].classes_)}")
print(f"[ModelAdapter] ✓ Known GPs     : {len(_preprocessors['le_gp'].classes_)}")


# =============================================================================
# TRANSLATION TABLES
# These are the core of the adapter. Each maps a 2026 entity to the closest
# known 2025 entity in the encoder.
# =============================================================================

# Driver analogues — ONLY for drivers whose code is not in the encoder.
#
# Who needs an analogue and who doesn't:
#   HAD, ANT, BOR, COL, BEA — all debuted in 2025. Their full season of
#   data IS in features_dataset_with_targets.csv and therefore in le_driver.
#   They pass through resolve_driver_code() unchanged, no substitution needed.
#
#   LIN (Arvid Lindblad) — the ONLY true 2026 rookie. Zero F1 races before
#   2026. His driver code does not exist in le_driver at all. He needs an
#   analogue so the encoder doesn't crash.
#
# THEORY — Why Lawson as Lindblad's analogue?
#   Same team (Racing Bulls), same trajectory (dominant F2/F3 career →
#   Red Bull junior → Racing Bulls seat). Lawson's 2023 substitute stint
#   is the closest data point we have for "elite junior, first F1 laps,
#   Racing Bulls machinery". The model's Lawson embedding gives us a
#   sensible prior that we scale with Lindblad's testing pace once available.
DRIVER_ANALOGUES_2026 = {
    "LIN": "LAW",   # Arvid Lindblad → Liam Lawson (only analogue needed)
}

# Team analogues for new 2026 constructors
TEAM_ANALOGUES_2026 = {
    # Cadillac — brand-new constructor, well-funded, Ferrari customer engine.
    # Best historical analogue: Racing Point 2019 (well-funded, customer Mercedes,
    # building toward competitiveness, midfield P8-P9 in WCC first season).
    "Cadillac F1 Team": "Racing Point",

    # Audi (formerly Sauber) — same physical team, rebranded.
    # The encoder likely knows "Alfa Romeo" or "Sauber" — use Alfa Romeo 2022.
    "Audi":             "Alfa Romeo",
}

# GP analogues for new 2026 circuits
GP_ANALOGUES_2026 = {
    # Madrid Grand Prix — new street/hybrid circuit in Spain.
    # Closest in terms of layout complexity + lap count: Australian GP.
    # Both are semi-street, similar track length, similar overtaking difficulty.
    "Madrid Grand Prix": "Australian Grand Prix",
}

# 2026 regulation pace scaling factor per team
# THEORY: The model was trained on 2018-2025 data. The 2026 active-aero
# regulations produce different lap times. Rather than retraining, we apply
# a multiplicative correction AFTER the model predicts, anchored to the
# February 2026 Bahrain testing fastest times.
#
# Formula: scaling_factor = testing_lap_time / model_predicted_baseline
# A factor < 1.0 means the car is FASTER than the model expects (e.g. Ferrari)
# A factor > 1.0 means SLOWER (e.g. Cadillac — new team, no baseline data)
#
# These are initialised from testing data and updated as the 2026 season
# progresses via the interactive input system in insights_engine.py.
TEAM_PACE_SCALING_2026 = {
    "Ferrari":          0.982,  # Leclerc P1 in testing: 1:31.992 — exceptionally fast
    "McLaren":          0.989,  # Piastri/Norris P3/P4 in testing
    "Red Bull":         0.993,  # Verstappen P5 in testing
    "Mercedes":         0.991,  # Antonelli P2, Russell P6
    "Alpine":           0.996,  # Gasly P8, Colapinto P11
    "Haas":             0.997,  # Bearman P9, Ocon P14
    "Racing Bulls":     1.001,  # Lindblad P13, Hadjar P15
    "Williams":         1.003,  # Sainz P16 — limited testing data
    "Aston Martin":     1.008,  # Stroll battery issues — very limited data
    "Audi":             0.998,  # Hulkenberg P12, Bortoleto P10
    "Cadillac F1 Team": 1.025,  # No testing time — estimated from constructor gap
}

# New-team adjustment: positions added to predicted finish for Cadillac
# in early races (typically takes 4-6 races to find their feet)
CADILLAC_RAMP_UP = {
    1: 4.0, 2: 3.5, 3: 3.0, 4: 2.5, 5: 2.0,
    6: 1.5, 7: 1.0, 8: 0.5, 9: 0.0,  # Race 9+ = full pace achieved
}

# =============================================================================
# PER-DRIVER PACE DELTA — 2026
# =============================================================================
# Multiplicative delta applied ON TOP of the team scaling factor.
# Captures intra-team performance gaps: if Hamilton consistently laps faster
# than the Ferrari team baseline, his delta converges below 1.0 over the season.
#
# Initialised from Bahrain testing intra-team gaps.
# THEORY — Why multiplicative, not additive?
#   An additive offset (+0.3s) means the same thing regardless of circuit.
#   A multiplicative factor (×0.997) scales naturally with circuit lap time —
#   a 0.3% edge at Monaco (80s laps) = 0.24s, at Monza (80s) = similar.
#   This is more physically meaningful than a fixed-second offset.
#
# Formula per driver:   final_time = team_scaled_time × driver_delta
# A delta < 1.0 → driver is FASTER than team baseline (good)
# A delta > 1.0 → driver is SLOWER than team baseline (bad)
#
# Initialisation from testing:
#   Within each team, the faster testing driver gets delta = base - gap/2,
#   the slower gets delta = base + gap/2. Both start close to 1.0 since
#   testing data is noisy (different programmes, fuel loads, tyre sets).
#   The EMA updates pull them apart as real race data arrives.
DRIVER_PACE_DELTA_2026 = {
    # McLaren — Piastri 0.010s faster than Norris in testing (essentially equal)
    "NOR": 1.0001,
    "PIA": 0.9999,

    # Ferrari — Leclerc 1.416s faster than Hamilton in testing
    # Meaningful gap but Hamilton is known to need time with a new car
    "LEC": 0.9970,
    "HAM": 1.0030,

    # Red Bull — Verstappen 1.151s faster than Hadjar (rookie gap expected)
    "VER": 0.9935,
    "HAD": 1.0065,

    # Mercedes — Antonelli 0.394s faster than Russell (surprising, but testing)
    "ANT": 0.9979,
    "RUS": 1.0021,

    # Aston Martin — Alonso estimated faster than Stroll (Stroll had battery issues)
    "ALO": 0.9970,
    "STR": 1.0030,

    # Alpine — Gasly 0.397s faster than Colapinto in testing
    "GAS": 0.9979,
    "COL": 1.0021,

    # Haas — Bearman 0.714s faster than Ocon in testing
    "BEA": 0.9962,
    "OCO": 1.0038,

    # Racing Bulls — Lindblad 0.383s faster than Lawson in testing
    # (surprising for a rookie — but limited data)
    "LIN": 0.9980,
    "LAW": 1.0020,

    # Williams — Sainz 0.358s faster than Albon (estimated)
    "SAI": 0.9981,
    "ALB": 1.0019,

    # Audi — Bortoleto 0.232s faster than Hulkenberg in testing
    "BOR": 0.9988,
    "HUL": 1.0012,

    # Cadillac — both estimated, very limited testing data
    "PER": 0.9990,
    "BOT": 1.0010,
}

# Confidence weight per driver: starts low (prior from testing only),
# grows as more races are observed. Used to modulate how aggressively
# the EMA updates. After ~5 races the weight saturates at 1.0 and the
# EMA alpha takes over as the primary control.
DRIVER_DELTA_CONFIDENCE = {d: 0.0 for d in DRIVER_PACE_DELTA_2026}

# Number of race observations per driver (used to compute confidence)
DRIVER_DELTA_OBSERVATIONS = {d: 0 for d in DRIVER_PACE_DELTA_2026}


# =============================================================================
# PUBLIC API — These are the only functions the rest of phase_3 calls
# =============================================================================

def resolve_driver_code(driver_code: str) -> str:
    """
    Translate a 2026 driver code to a known encoder code.
    Known drivers pass through unchanged. New drivers get their analogue.
    """
    known = set(_preprocessors["le_driver"].classes_)
    if driver_code in known:
        return driver_code
    analogue = DRIVER_ANALOGUES_2026.get(driver_code)
    if analogue and analogue in known:
        return analogue
    # Last resort: find closest known driver (alphabetical — crude but safe)
    warnings.warn(
        f"Driver '{driver_code}' unknown and no analogue defined. "
        f"Falling back to 'HAM' as generic baseline.",
        UserWarning,
    )
    return "HAM"


def resolve_team_name(team_name: str) -> str:
    """Translate a 2026 team name to a known encoder team name."""
    known = set(_preprocessors["le_team"].classes_)
    if team_name in known:
        return team_name
    analogue = TEAM_ANALOGUES_2026.get(team_name)
    if analogue and analogue in known:
        return analogue
    warnings.warn(f"Team '{team_name}' unknown. Falling back to 'Williams'.")
    return "Williams"


def resolve_gp_name(gp_name: str) -> str:
    """Translate a 2026 GP name to a known encoder GP name."""
    known = set(_preprocessors["le_gp"].classes_)
    if gp_name in known:
        return gp_name
    analogue = GP_ANALOGUES_2026.get(gp_name)
    if analogue and analogue in known:
        return analogue
    warnings.warn(f"GP '{gp_name}' unknown. Falling back to 'Australian Grand Prix'.")
    return "Australian Grand Prix"


def get_reference_race_info(
    gp_name: str,
    driver_code: str,
    team_name: str,
    grid_position: int = 10,
    total_laps: int = 57,
    weather_code: float = 1.0,
) -> pd.Series:
    """
    Build a race_info Series that predict_single can consume.

    For 2026 races (no historical row exists yet), we construct a synthetic
    row using:
      1. The resolved GP's average numerical values from historical data
      2. Driver's own historical averages where available
      3. Sensible defaults otherwise

    THEORY — Why not just use last year's row directly?
        Last year's row hardcodes last year's grid position, pit stop timing,
        and positional averages for that specific race. We want the model to
        predict from scratch for 2026, using only what we feed it (the user's
        strategy). So we build a neutral baseline row and let the strategy
        features do the talking.
    """
    resolved_gp     = resolve_gp_name(gp_name)
    resolved_driver = resolve_driver_code(driver_code)

    # Try to find a recent (2024-2025) row for the resolved GP + driver
    lookup = _historical_df[
        (_historical_df["gp_name"]     == resolved_gp) &
        (_historical_df["driver_code"] == resolved_driver) &
        (_historical_df["Year"]        >= 2023)
    ].sort_values("Year", ascending=False)

    if not lookup.empty:
        base = lookup.iloc[0].copy()
    else:
        # No exact match — use GP averages as neutral baseline
        gp_rows = _historical_df[_historical_df["gp_name"] == resolved_gp]
        if gp_rows.empty:
            gp_rows = _historical_df  # Ultimate fallback: global averages
        base = gp_rows.mean(numeric_only=True)
        base["gp_name"]     = resolved_gp
        base["driver_code"] = resolved_driver
        base["weather_code"]= weather_code

    # Overlay the 2026-specific values we actually know
    base["grid_position"]        = grid_position
    base["total_laps_completed"] = total_laps
    base["weather_code"]         = weather_code

    return base


def predict_2026(
    gp_name: str,
    driver_code: str,
    team_name: str,
    num_pit_stops: int,
    strategy_features: dict,
    grid_position: int = 10,
    total_laps: int = 57,
    weather_code: float = 1.0,
    race_number: int = 1,
) -> dict:
    """
    Main prediction entry point for 2026 season.
    Wraps the existing predict_single with full 2026 translation layer.

    Returns a dict with:
        predicted_avg_lap_time_raw   : model output before 2026 scaling
        predicted_avg_lap_time_2026  : after applying team pace scaling factor
        pace_scaling_factor          : the multiplier applied
        resolved_driver              : what the model actually saw
        resolved_team                : what the model actually saw
        resolved_gp                  : what the model actually saw
    """
    # Step 1: Translate 2026 → known encoder values
    resolved_driver = resolve_driver_code(driver_code)
    resolved_team   = resolve_team_name(team_name)
    resolved_gp     = resolve_gp_name(gp_name)

    # Step 2: Encode (year always maps to 2025 — closest known year)
    year_proxy = _preprocessors["le_year"].classes_[-1]  # Most recent year in encoder

    try:
        gp_idx      = int(_preprocessors["le_gp"].transform([resolved_gp])[0])
        driver_idx  = int(_preprocessors["le_driver"].transform([resolved_driver])[0])
        team_idx    = int(_preprocessors["le_team"].transform([resolved_team])[0])
        year_idx    = int(_preprocessors["le_year"].transform([year_proxy])[0])
        weather_idx = int(_preprocessors["le_weather"].transform([weather_code])[0])
    except ValueError as e:
        raise ValueError(f"Encoding failed after resolution: {e}")

    # Step 3: Build the race_info baseline row
    race_info = get_reference_race_info(
        gp_name, driver_code, team_name,
        grid_position, total_laps, weather_code
    )

    # Step 4: Build numerical feature vector (same logic as phase_2)
    num_dict = _build_num_dict(race_info, float(num_pit_stops), strategy_features)
    ordered  = pd.DataFrame(
        [[num_dict[col] for col in _preprocessors["num_cols"]]],
        columns=_preprocessors["num_cols"]
    )
    scaled = _preprocessors["scaler"].transform(ordered)

    # Step 5: Run inference
    with torch.no_grad():
        raw_pred = _model(
            torch.tensor([gp_idx],      dtype=torch.long).to(_device),
            torch.tensor([driver_idx],  dtype=torch.long).to(_device),
            torch.tensor([team_idx],    dtype=torch.long).to(_device),
            torch.tensor([year_idx],    dtype=torch.long).to(_device),
            torch.tensor([weather_idx], dtype=torch.long).to(_device),
            torch.tensor(scaled,        dtype=torch.float32).to(_device),
        )
    raw_delta = raw_pred.item()

    # Step 5b: Reconstruct absolute lap time if model was trained on circuit-delta target.
    # If target_is_delta is False (old model) or circuit_means is missing, use raw output.
    if _preprocessors.get("target_is_delta", False):
        circuit_means = _preprocessors.get("circuit_means", {})
        circuit_mean  = circuit_means.get(resolved_gp,
                        _preprocessors.get("global_mean", 90.0))
        raw_time = round(raw_delta + circuit_mean, 3)
    else:
        raw_time = round(raw_delta, 3)

    # Guard: if raw_time is still implausibly small (old joblib with delta model weights)
    # fall back to global mean so the UI never shows negative times
    if raw_time < 60.0:
        fallback = _preprocessors.get("global_mean", 90.0)
        circuit_means = _preprocessors.get("circuit_means", {})
        raw_time = round(circuit_means.get(resolved_gp, fallback), 3)

    # Step 6: Apply 2026 pace scaling
    # THEORY: The model predicts a 2025-era lap time. We scale it to 2026
    # using the testing-anchored factor. A factor of 0.982 means the team
    # laps 1.8% faster in 2026 than the model's 2025 baseline predicts.
    scale        = TEAM_PACE_SCALING_2026.get(team_name, 1.0)
    scaled_time  = round(raw_time * scale, 3)

    # Step 6b: Apply per-driver delta on top of team scaling.
    # This captures intra-team performance gaps — e.g. Hamilton running
    # consistently faster or slower than the Ferrari team baseline.
    # predicted_time_before_driver_delta is passed to update_driver_delta()
    # by insights_engine so it can isolate the driver-specific error component.
    driver_delta       = DRIVER_PACE_DELTA_2026.get(driver_code, 1.0)
    final_scaled_time  = round(scaled_time * driver_delta, 3)

    # Step 7: Cadillac ramp-up adjustment (new constructor learning curve)
    position_penalty = 0.0
    if team_name == "Cadillac F1 Team":
        position_penalty = CADILLAC_RAMP_UP.get(race_number, 0.0)

    # Step 8: Rank predicted lap time against the full 2026 field.
    predicted_position = _estimate_position_vs_field(
        our_lap_time     = final_scaled_time,
        our_driver       = driver_code,
        gp_name          = gp_name,
        resolved_gp      = resolved_gp,
        total_laps       = total_laps,
        weather_code     = weather_code,
        year_proxy       = year_proxy,
        race_number      = race_number,
        position_penalty = position_penalty,
    )

    return {
        "predicted_avg_lap_time_raw":           raw_time,
        "predicted_avg_lap_time_2026":          final_scaled_time,
        "predicted_avg_lap_time_before_driver": scaled_time,   # exposed for insights_engine
        "predicted_position":                   predicted_position,
        "pace_scaling_factor":                  scale,
        "driver_delta":                         driver_delta,
        "driver_delta_confidence":              DRIVER_DELTA_CONFIDENCE.get(driver_code, 0.0),
        "cadillac_position_penalty":            position_penalty,
        "resolved_driver":                      resolved_driver,
        "resolved_team":                        resolved_team,
        "resolved_gp":                          resolved_gp,
        "year_proxy_used":                      int(year_proxy),
        "is_new_driver":                        driver_code == "LIN",
        "is_new_team":                          team_name   != resolved_team,
        "is_new_circuit":                       gp_name     != resolved_gp,
    }


# 2026 grid — driver code → team name.
# Used by _estimate_position_vs_field to predict a lap time for every rival.
_GRID_2026 = {
    "NOR": "McLaren",        "PIA": "McLaren",
    "LEC": "Ferrari",        "HAM": "Ferrari",
    "VER": "Red Bull",       "HAD": "Red Bull",
    "RUS": "Mercedes",       "ANT": "Mercedes",
    "ALO": "Aston Martin",   "STR": "Aston Martin",
    "GAS": "Alpine",         "COL": "Alpine",
    "OCO": "Haas",           "BEA": "Haas",
    "LAW": "Racing Bulls",   "LIN": "Racing Bulls",
    "ALB": "Williams",       "SAI": "Williams",
    "HUL": "Audi",           "BOR": "Audi",
    "PER": "Cadillac F1 Team", "BOT": "Cadillac F1 Team",
}

# Neutral 1-stop baseline strategy used when predicting rival lap times.
# Dry: Medium start → Hard finish, pit at 40% race distance.
# Wet: Intermediate start → Hard finish, pit at 40% race distance.
# Intentionally average — we don't want to assume rivals run perfect strategies.
_NEUTRAL_STRATEGY_DRY = {
    "laps_on_soft":         0,
    "laps_on_medium":       0,   # filled dynamically per total_laps
    "laps_on_hard":         0,   # filled dynamically per total_laps
    "laps_on_intermediate": 0,
    "laps_on_wet":          0,
    "first_pit_lap_pct":    0.40,
    "second_pit_lap_pct":   0.0,
    "third_pit_lap_pct":    0.0,
}

_NEUTRAL_STRATEGY_WET = {
    "laps_on_soft":         0,
    "laps_on_medium":       0,
    "laps_on_hard":         0,
    "laps_on_intermediate": 0,   # filled dynamically per total_laps
    "laps_on_wet":          0,   # filled dynamically per total_laps
    "first_pit_lap_pct":    0.55, # wet races tend to pit later in intermediates
    "second_pit_lap_pct":   0.0,
    "third_pit_lap_pct":    0.0,
}


def _predict_rival_lap_time(
    driver_code: str,
    team_name: str,
    resolved_gp: str,
    total_laps: int,
    weather_code: float,
    year_proxy,
) -> float:
    """
    Predict a single rival's average lap time using a neutral strategy.
    Used only for field comparison — not exposed via the API.
    """
    resolved_d = resolve_driver_code(driver_code)
    resolved_t = resolve_team_name(team_name)

    try:
        gp_idx      = int(_preprocessors["le_gp"].transform([resolved_gp])[0])
        driver_idx  = int(_preprocessors["le_driver"].transform([resolved_d])[0])
        team_idx    = int(_preprocessors["le_team"].transform([resolved_t])[0])
        year_idx    = int(_preprocessors["le_year"].transform([year_proxy])[0])
        weather_idx = int(_preprocessors["le_weather"].transform([weather_code])[0])
    except (ValueError, KeyError):
        return 999.0  # Unknown encoding — push to back of grid

    is_wet  = weather_code >= 3.0
    pit_lap = int(total_laps * (0.55 if is_wet else 0.40))

    if is_wet:
        # Neutral wet strategy: intermediates first stint, hard to finish
        # Matches the most common real wet race pattern in training data
        strategy = dict(_NEUTRAL_STRATEGY_WET)
        strategy["laps_on_intermediate"] = pit_lap - 1
        strategy["laps_on_hard"]         = total_laps - pit_lap + 1
    else:
        strategy = dict(_NEUTRAL_STRATEGY_DRY)
        strategy["laps_on_medium"] = pit_lap - 1
        strategy["laps_on_hard"]   = total_laps - pit_lap + 1

    # Degradation features for the neutral strategy
    if is_wet:
        stints = [("INTERMEDIATE", pit_lap - 1), ("HARD", total_laps - pit_lap + 1)]
    else:
        stints = [("MEDIUM", pit_lap - 1), ("HARD", total_laps - pit_lap + 1)]
    # COMPOUND_HARDNESS is imported from pace_anchor at module level —
    # do not redefine it here. Using the same values as retrain_no_leakage.py
    # ensures deg_score is on the same scale the model was trained on.
    lengths    = [s[1] for s in stints]
    avg_age    = sum(lengths) / len(lengths)
    max_deg    = max(laps * COMPOUND_HARDNESS.get(c, 1.0) for c, laps in stints)
    max_laps   = max(laps for _, laps in stints)
    strategy["avg_tyre_age"]   = round(avg_age, 2)
    strategy["max_stint_laps"] = float(max_laps)
    strategy["deg_score"]      = round(max_deg, 2)

    # Build a generic race_info for this rival using GP averages
    gp_rows = _historical_df[_historical_df["gp_name"] == resolved_gp]
    if gp_rows.empty:
        gp_rows = _historical_df
    base = gp_rows.mean(numeric_only=True)
    base["grid_position"]        = 11.0
    base["total_laps_completed"] = float(total_laps)
    base["weather_code"]         = weather_code

    num_dict = _build_num_dict(base, 1.0, strategy)
    ordered  = pd.DataFrame(
        [[num_dict[col] for col in _preprocessors["num_cols"]]],
        columns=_preprocessors["num_cols"]
    )
    scaled_input = _preprocessors["scaler"].transform(ordered)

    with torch.no_grad():
        raw = _model(
            torch.tensor([gp_idx],      dtype=torch.long).to(_device),
            torch.tensor([driver_idx],  dtype=torch.long).to(_device),
            torch.tensor([team_idx],    dtype=torch.long).to(_device),
            torch.tensor([year_idx],    dtype=torch.long).to(_device),
            torch.tensor([weather_idx], dtype=torch.long).to(_device),
            torch.tensor(scaled_input,  dtype=torch.float32).to(_device),
        )
    raw_delta = raw.item()

    # Reconstruct absolute time if model was trained on circuit-delta target
    if _preprocessors.get("target_is_delta", False):
        circuit_means = _preprocessors.get("circuit_means", {})
        circuit_mean  = circuit_means.get(resolved_gp, _preprocessors.get("global_mean", 90.0))
        raw_time = raw_delta + circuit_mean
    else:
        raw_time = raw_delta

    # Apply this team's 2026 pace scaling, then the driver's personal delta
    team_scale   = TEAM_PACE_SCALING_2026.get(team_name, 1.0)
    driver_delta = DRIVER_PACE_DELTA_2026.get(driver_code, 1.0)
    return raw_time * team_scale * driver_delta


def _estimate_position_vs_field(
    our_lap_time: float,
    our_driver: str,
    gp_name: str,
    resolved_gp: str,
    total_laps: int,
    weather_code: float,
    year_proxy,
    race_number: int,
    position_penalty: float,
) -> int:
    """
    Rank our driver's predicted lap time against every other driver
    on the 2026 grid and return the finishing position.

    THEORY:
        Average lap time is the single best proxy for finishing position
        across a race distance. The driver who laps fastest on average
        wins, all else being equal. By predicting a lap time for every
        rival under a neutral strategy, we get a realistic field to
        rank against. The user's strategy directly affects our_lap_time,
        so now strategy choices produce real position differences.

        Rivals use a neutral 1-stop strategy — not optimal, not terrible.
        This means the model slightly rewards good user strategies (beating
        rivals who aren't running the same optimal) and punishes bad ones.
    """
    field_times = {}

    for rival_code, rival_team in _GRID_2026.items():
        if rival_code == our_driver:
            continue
        rival_time = _predict_rival_lap_time(
            driver_code  = rival_code,
            team_name    = rival_team,
            resolved_gp  = resolved_gp,
            total_laps   = total_laps,
            weather_code = weather_code,
            year_proxy   = year_proxy,
        )
        # Apply Cadillac ramp-up penalty to rival Cadillac cars too
        if rival_team == "Cadillac F1 Team":
            cadillac_penalty = CADILLAC_RAMP_UP.get(race_number, 0.0)
            # Convert position penalty to approximate lap time penalty
            # (~0.5s per position is a reasonable F1 race pace proxy)
            rival_time += cadillac_penalty * 0.5

        field_times[rival_code] = rival_time

    # Rank: sort all times ascending, find where our time slots in
    all_times = sorted(field_times.values())
    position  = 1
    for rival_time in all_times:
        if rival_time < our_lap_time:
            position += 1

    # Apply Cadillac ramp-up to our own final position if applicable
    position = int(round(position + position_penalty))
    return max(1, min(22, position))


def update_pace_scaling(team_name: str, actual_lap_time: float, predicted_lap_time: float):
    """
    Bayesian update of the team's pace scaling factor after a real race result.
    Called by insights_engine.py when the user inputs actual race results.

    THEORY — Online Learning / Bayesian Update:
        After each real race, we have new evidence: the actual lap time vs
        what the model predicted. We use a weighted average (Exponential
        Moving Average) to update the scaling factor:

            new_scale = α * (actual/predicted) + (1-α) * old_scale

        α = 0.3 means we weight new evidence at 30% and history at 70%.
        This prevents overreacting to a single unusual result (e.g. rain)
        while still adapting to genuine pace trends over the season.
    """
    alpha        = 0.3
    observed_scale = actual_lap_time / predicted_lap_time
    old_scale      = TEAM_PACE_SCALING_2026.get(team_name, 1.0)
    new_scale      = alpha * observed_scale + (1 - alpha) * old_scale
    TEAM_PACE_SCALING_2026[team_name] = round(new_scale, 4)
    return new_scale


def update_driver_delta(
    driver_code: str,
    actual_lap_time: float,
    predicted_lap_time_before_driver_delta: float,
) -> float:
    """
    Bayesian EMA update of a driver's personal pace delta after a real race.
    Called by insights_engine.py alongside update_pace_scaling.

    THEORY — What this measures:
        The team scaling factor already corrects for how fast the car is
        relative to the model's 2025 baseline. The driver delta corrects
        for how this specific driver performs relative to their team's
        baseline — capturing intra-team skill gaps.

        Example: Ferrari's scaling factor might converge to 0.980.
        But Hamilton consistently laps 0.3% faster than that baseline
        while Leclerc laps 0.3% slower. After several races:
            HAM delta → ~0.997, LEC delta → ~1.003

        These two corrections are independent and multiplicative:
            final_time = raw_model_time × team_scale × driver_delta

    THEORY — Confidence-weighted alpha:
        For the first few races, the driver delta is highly uncertain
        (testing data is noisy). We use a lower effective alpha early on
        and ramp it up as more observations arrive, asymptoting at 0.25.

            effective_alpha = 0.25 × min(observations / 5, 1.0)

        This means after race 1 the update is gentle (alpha=0.05),
        after race 5 it's at full strength (alpha=0.25), and stays there.
        Result: early-season outliers (rain, safety cars) barely move
        the delta, but a consistent 5-race trend fully updates it.

    Parameters
    ----------
    driver_code : str
        Three-letter driver code e.g. "HAM"
    actual_lap_time : float
        The driver's actual average lap time from the real race result
    predicted_lap_time_before_driver_delta : float
        What the model predicted AFTER team scaling but BEFORE driver delta.
        This isolates the driver-specific component of any prediction error.

    Returns
    -------
    float : The updated driver delta
    """
    if driver_code not in DRIVER_PACE_DELTA_2026:
        return 1.0

    # Increment observation count and compute confidence-weighted alpha
    DRIVER_DELTA_OBSERVATIONS[driver_code] += 1
    n_obs = DRIVER_DELTA_OBSERVATIONS[driver_code]

    # Alpha ramps from 0.05 (race 1) to 0.25 (race 5+)
    # Low early = resistant to noise. Full strength by race 5 = adapts to trends.
    effective_alpha = 0.25 * min(n_obs / 5.0, 1.0)

    # The observed delta = how much faster/slower than team baseline this driver was
    observed_delta = actual_lap_time / predicted_lap_time_before_driver_delta

    old_delta = DRIVER_PACE_DELTA_2026[driver_code]
    new_delta = effective_alpha * observed_delta + (1 - effective_alpha) * old_delta

    # Clamp to ±3% — prevents runaway updates from weird races (SC, rain, DNF data)
    new_delta = max(0.970, min(1.030, new_delta))

    DRIVER_PACE_DELTA_2026[driver_code] = round(new_delta, 5)
    DRIVER_DELTA_CONFIDENCE[driver_code] = round(min(n_obs / 5.0, 1.0), 2)

    return new_delta


# =============================================================================
# INTERNAL HELPERS (mirror of phase_2 logic, isolated here)
# =============================================================================

def _build_num_dict(race_info: pd.Series, num_pit_stops: float, strategy_features: dict) -> dict:
    total_laps = float(race_info.get("total_laps_completed", 57))
    grid_pos   = float(race_info.get("grid_position", 10))

    pit_stops_per_lap = num_pit_stops / total_laps if total_laps > 0 else 0.0

    if num_pit_stops == 0:
        avg_pit_dur = 0.0
    else:
        # FIX 5: circuit-specific pit loss (same logic as main.py build_num_dict)
        gp_name_key = str(race_info.get("gp_name", ""))
        circuit_pit = PIT_LOSS_BY_CIRCUIT.get(gp_name_key)
        if circuit_pit is not None:
            avg_pit_dur = circuit_pit
        else:
            avg_pit_dur = float(race_info.get("avg_pit_duration",
                                              _preprocessors["avg_pit_mean"]))
            if pd.isna(avg_pit_dur) or avg_pit_dur <= 0:
                avg_pit_dur = float(_preprocessors["avg_pit_mean"])

    # FIX 3: fuel load proxy — mirror of main.py build_num_dict
    first_pit_pct      = strategy_features.get("first_pit_lap_pct", 0.0)
    stint1_fuel_weight = 1.0 if num_pit_stops == 0 else float(first_pit_pct)
    num_pit_stops_norm = min(num_pit_stops / 3.0, 1.0)

    # FIX 1: leaky positional features (best_position, worst_position,
    # avg_position, avg_position_vs_grid) are intentionally excluded here.
    # They were removed from SAFE_NUM_COLS in retrain_no_leakage.py because
    # they are post-race outcomes — passing them at inference was data leakage.
    # The retrained model's scaler does NOT expect them; including them here
    # would silently corrupt the feature vector if the old .joblib is loaded,
    # or crash with a KeyError if the new one is. Either way: wrong.
    base = {
        "grid_position":        grid_pos,
        "num_pit_stops":        num_pit_stops,
        "avg_pit_duration":     avg_pit_dur,
        "total_laps_completed": total_laps,
        "pit_stops_per_lap":    pit_stops_per_lap,
        "stint1_fuel_weight":   stint1_fuel_weight,
        "num_pit_stops_norm":   num_pit_stops_norm,
        **strategy_features,
    }

    # Ensure every column the scaler expects is present.
    # Degradation cols are included if the retrained model knows them.
    # If the current model was trained without them, they are simply absent
    # from num_cols and the scaler ignores them — no crash either way.
    for col in _preprocessors.get("degradation_cols", []):
        if col not in base:
            base[col] = 0.0

    return base