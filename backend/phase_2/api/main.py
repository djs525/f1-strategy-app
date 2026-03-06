"""
F1 Strategy Simulator API
=============================
POST /simulate

User sends their strategy as a starting compound + a list of pit stops,
each with the lap number and the compound going onto the car:

    {
      "year": 2024,
      "gp_name": "Spanish Grand Prix",
      "driver_code": "VER",
      "starting_compound": "SOFT",
      "pit_stops": [
        {"lap": 22, "compound": "HARD"},
        {"lap": 44, "compound": "MEDIUM"}
      ]
    }

The API automatically derives all 8 strategy features from this input:
    first_pit_lap_pct   = 22 / 66 = 0.333
    second_pit_lap_pct  = 44 / 66 = 0.667
    third_pit_lap_pct   = 0.0
    laps_on_soft        = 21  (laps 1 – 21)
    laps_on_hard        = 22  (laps 22 – 43)
    laps_on_medium      = 23  (laps 44 – 66)
    laps_on_intermediate= 0
    laps_on_wet         = 0

Two simulations run per request:
  1. user_strategy   — exactly what the user submitted
  2. optimal_strategy — brute-force search over all valid compound
                        sequences × pit timing combinations
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import itertools
import os
import sys

# ──────────────────────────────────────────────────────────────────────────────
# 1.  MODEL ARCHITECTURE  (must match lap_time_predictor.py exactly)
# ──────────────────────────────────────────────────────────────────────────────

class F1LapTimePredictor(nn.Module):
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


# ──────────────────────────────────────────────────────────────────────────────
# 2.  APP + DEVICE
# ──────────────────────────────────────────────────────────────────────────────

app    = FastAPI(title="F1 Strategy Simulator")
device = torch.device("cpu")

# ──────────────────────────────────────────────────────────────────────────────
# 3.  LOAD ARTIFACTS ON STARTUP
# ──────────────────────────────────────────────────────────────────────────────

print("Loading model and preprocessors...")

_ROOT      = os.path.dirname(os.path.abspath(__file__))  # backend/phase_2/api/
_DATA_ROOT = os.path.abspath(os.path.join(_ROOT, "../../../data/trained_models"))

preprocessors = joblib.load(os.path.join(_DATA_ROOT, "f1_preprocessors.joblib"))


model = F1LapTimePredictor(
    num_gps=len(preprocessors["le_gp"].classes_),
    num_drivers=len(preprocessors["le_driver"].classes_),
    num_teams=len(preprocessors["le_team"].classes_),
    num_numerical_features=len(preprocessors["num_cols"]),
    num_years=len(preprocessors["le_year"].classes_),
    num_weather_conditions=len(preprocessors["le_weather"].classes_),
)
model.load_state_dict(torch.load(
    os.path.join(_DATA_ROOT, "best_lap_time_predictor.pth"),
    map_location=device
))
model.eval()

historical_df = pd.read_csv(os.path.join(_DATA_ROOT, "../feature_data/features_dataset_with_targets.csv"))


COMPOUND_COLS    = preprocessors.get("compound_cols",   [
    "laps_on_soft", "laps_on_medium", "laps_on_hard",
    "laps_on_intermediate", "laps_on_wet",
])
PIT_TIMING_COLS  = preprocessors.get("pit_timing_cols", [
    "first_pit_lap_pct", "second_pit_lap_pct", "third_pit_lap_pct",
])

COMPOUND_TO_COL = {
    "SOFT":         "laps_on_soft",
    "MEDIUM":       "laps_on_medium",
    "HARD":         "laps_on_hard",
    "INTERMEDIATE": "laps_on_intermediate",
    "WET":          "laps_on_wet",
}

DEGRADATION_COLS = preprocessors.get("degradation_cols", [
    "avg_tyre_age",
    "max_stint_laps",
    "deg_score",
])

COMPOUND_HARDNESS = {
    "SOFT":         3.0,
    "MEDIUM":       2.0,
    "HARD":         1.0,
    "INTERMEDIATE": 1.5,
    "WET":          1.0,
}

print("✓ Model ready")

# ──────────────────────────────────────────────────────────────────────────────
# 4.  PYDANTIC SCHEMAS
# ──────────────────────────────────────────────────────────────────────────────

VALID_COMPOUNDS = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}


class PitStop(BaseModel):
    lap: int      = Field(..., description="Lap number the pit stop occurs on", example=22, ge=1)
    compound: str = Field(..., description="Compound fitted after this stop", example="HARD")

    @validator("compound")
    def compound_must_be_valid(cls, v):
        v = v.upper().strip()
        if v not in VALID_COMPOUNDS:
            raise ValueError(f"compound must be one of {VALID_COMPOUNDS}")
        return v


class StrategyInput(BaseModel):
    year: int            = Field(..., example=2024)
    gp_name: str         = Field(..., example="Spanish Grand Prix")
    driver_code: str     = Field(..., example="VER")
    starting_compound: str = Field(
        ...,
        description="Tyre compound the driver starts the race on",
        example="SOFT"
    )
    pit_stops: List[PitStop] = Field(
        ...,
        description=(
            "Ordered list of pit stops. Each entry specifies the lap number "
            "where the stop happens and the compound going on the car. "
            "Maximum 3 pit stops."
        ),
        max_items=3,
        example=[
            {"lap": 22, "compound": "HARD"},
            {"lap": 44, "compound": "MEDIUM"},
        ],
    )

    @validator("starting_compound")
    def starting_compound_must_be_valid(cls, v):
        v = v.upper().strip()
        if v not in VALID_COMPOUNDS:
            raise ValueError(f"starting_compound must be one of {VALID_COMPOUNDS}")
        return v

    @validator("pit_stops")
    def pit_laps_must_be_strictly_increasing(cls, pit_stops):
        laps = [p.lap for p in pit_stops]
        for i in range(len(laps) - 1):
            if laps[i] >= laps[i + 1]:
                raise ValueError(
                    f"Pit stop laps must be strictly increasing. "
                    f"Got lap {laps[i]} followed by lap {laps[i+1]}."
                )
        return pit_stops

    @validator("pit_stops")
    def no_same_compound_back_to_back(cls, pit_stops, values):
        starting = values.get("starting_compound", "")
        sequence = [starting] + [p.compound for p in pit_stops]
        for i in range(len(sequence) - 1):
            if sequence[i] == sequence[i + 1]:
                raise ValueError(
                    f"Back-to-back same compound not allowed. "
                    f"Stint {i+1} and {i+2} are both {sequence[i]}."
                )
        return pit_stops

    @validator("pit_stops")
    def dry_race_must_use_two_compounds(cls, pit_stops, values):
        starting = values.get("starting_compound", "")
        all_compounds = {starting} | {p.compound for p in pit_stops}
        dry = {"SOFT", "MEDIUM", "HARD"}
        dry_used = all_compounds & dry
        if len(pit_stops) > 0 and all_compounds <= dry and len(dry_used) < 2:
            raise ValueError(
                "Dry strategies must use at least 2 different compounds "
                "(F1 sporting regulation)."
            )
        return pit_stops


# ──────────────────────────────────────────────────────────────────────────────
# 5.  FEATURE DERIVATION
# ──────────────────────────────────────────────────────────────────────────────

def derive_strategy_features(
    starting_compound: str,
    pit_stops: List[PitStop],
    total_laps: int,
) -> dict:
    compound_laps = {col: 0 for col in COMPOUND_COLS}

    prev_lap         = 1
    current_compound = starting_compound.upper()

    for ps in pit_stops:
        laps_in_stint = ps.lap - prev_lap
        col = COMPOUND_TO_COL[current_compound]
        compound_laps[col] += laps_in_stint
        prev_lap         = ps.lap
        current_compound = ps.compound.upper()

    final_stint_laps = total_laps - prev_lap + 1
    col = COMPOUND_TO_COL[current_compound]
    compound_laps[col] += final_stint_laps

    pit_lap_list = [ps.lap for ps in pit_stops]

    first_pit_lap_pct  = pit_lap_list[0] / total_laps if len(pit_lap_list) > 0 else 0.0
    second_pit_lap_pct = pit_lap_list[1] / total_laps if len(pit_lap_list) > 1 else 0.0
    third_pit_lap_pct  = pit_lap_list[2] / total_laps if len(pit_lap_list) > 2 else 0.0

    return {
        **compound_laps,
        "first_pit_lap_pct":  first_pit_lap_pct,
        "second_pit_lap_pct": second_pit_lap_pct,
        "third_pit_lap_pct":  third_pit_lap_pct,
    }


def derive_degradation_features(
    starting_compound: str,
    pit_stops: List[PitStop],
    total_laps: int,
) -> dict:
    """
    Derives avg_tyre_age, max_stint_laps, deg_score from any strategy.
    These features tell the model HOW HARD tyres were pushed, not just
    which compounds were used. Bad strategies (e.g. 40-lap soft stint)
    get a high deg_score and the model predicts slower average lap times.
    """
    stints = []
    prev_lap         = 1
    current_compound = starting_compound.upper()

    for ps in pit_stops:
        stint_len = ps.lap - prev_lap
        stints.append((current_compound, stint_len))
        prev_lap         = ps.lap
        current_compound = ps.compound.upper()

    final_len = total_laps - prev_lap + 1
    stints.append((current_compound, max(final_len, 1)))

    stint_lengths = [s[1] for s in stints]
    avg_tyre_age  = sum(stint_lengths) / len(stint_lengths)

    # Find the most degradation-heavy stint
    max_deg  = 0.0
    max_laps = 0
    for compound, laps in stints:
        hardness = COMPOUND_HARDNESS.get(compound, 1.0)
        deg      = laps * hardness
        if deg > max_deg:
            max_deg  = deg
            max_laps = laps

    return {
        "avg_tyre_age":   round(avg_tyre_age, 2),
        "max_stint_laps": float(max_laps),
        "deg_score":      round(max_deg, 2),
    }


def build_num_dict(
    race_info: pd.Series,
    num_pit_stops: float,
    strategy_features: dict,
) -> dict:
    total_laps = float(race_info["total_laps_completed"])
    grid_pos   = float(race_info["grid_position"])

    pit_stops_per_lap = num_pit_stops / total_laps if total_laps > 0 else 0.0

    if num_pit_stops == 0:
        avg_pit_dur = 0.0
    else:
        avg_pit_dur = float(race_info["avg_pit_duration"])
        if pd.isna(avg_pit_dur) or avg_pit_dur <= 0:
            avg_pit_dur = float(preprocessors["avg_pit_mean"])

    # NOTE: best_position, worst_position, avg_position, avg_position_vs_grid
    # were removed from num_cols during retraining (leakage — post-race outcomes).
    # They are no longer passed to the model.

    base = {
        "grid_position":        grid_pos,
        "num_pit_stops":        num_pit_stops,
        "avg_pit_duration":     avg_pit_dur,
        "total_laps_completed": total_laps,
        "pit_stops_per_lap":    pit_stops_per_lap,
        **strategy_features,
    }
    # Ensure degradation cols are present — zero-filled if not in strategy_features
    for col in DEGRADATION_COLS:
        if col not in base:
            base[col] = 0.0
    return base


# ──────────────────────────────────────────────────────────────────────────────
# 6.  INFERENCE HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def predict_single(
    race_info: pd.Series,
    num_pit_stops: float,
    strategy_features: dict,
    gp_idx: int, driver_idx: int, team_idx: int,
    year_idx: int, weather_idx: int,
) -> float:
    num_dict = build_num_dict(race_info, num_pit_stops, strategy_features)
    ordered  = pd.DataFrame(
        [[num_dict[col] for col in preprocessors["num_cols"]]],
        columns=preprocessors["num_cols"]
    )
    scaled = preprocessors["scaler"].transform(ordered)

    with torch.no_grad():
        pred = model(
            torch.tensor([gp_idx],      dtype=torch.long).to(device),
            torch.tensor([driver_idx],  dtype=torch.long).to(device),
            torch.tensor([team_idx],    dtype=torch.long).to(device),
            torch.tensor([year_idx],    dtype=torch.long).to(device),
            torch.tensor([weather_idx], dtype=torch.long).to(device),
            torch.tensor(scaled, dtype=torch.float32).to(device),
        )
    raw = pred.item()

    # If the model was retrained with circuit-delta target, add circuit mean back
    if preprocessors.get("target_is_delta", False):
        gp_name      = race_info.get("gp_name", "")
        circuit_mean = preprocessors["circuit_means"].get(
            gp_name, preprocessors["global_mean"]
        )
        raw += circuit_mean

    return round(raw, 3)


def predict_batch(
    race_info: pd.Series,
    candidates: list,
    gp_idx: int, driver_idx: int, team_idx: int,
    year_idx: int, weather_idx: int,
    batch_size: int = 512,
) -> list:
    all_rows = []
    for cand in candidates:
        nd = build_num_dict(race_info, float(cand["num_pits"]), cand["sf"])
        all_rows.append([nd[col] for col in preprocessors["num_cols"]])

    scaled = preprocessors["scaler"].transform(
        pd.DataFrame(all_rows, columns=preprocessors["num_cols"])
    )
    n = len(scaled)

    gp_t  = torch.tensor([gp_idx]      * n, dtype=torch.long).to(device)
    dr_t  = torch.tensor([driver_idx]  * n, dtype=torch.long).to(device)
    tm_t  = torch.tensor([team_idx]    * n, dtype=torch.long).to(device)
    yr_t  = torch.tensor([year_idx]    * n, dtype=torch.long).to(device)
    wx_t  = torch.tensor([weather_idx] * n, dtype=torch.long).to(device)

    # Circuit mean for delta reconstruction (same for all candidates — same race)
    circuit_offset = 0.0
    if preprocessors.get("target_is_delta", False):
        gp_name = race_info.get("gp_name", "")
        circuit_offset = preprocessors["circuit_means"].get(
            gp_name, preprocessors["global_mean"]
        )

    preds = []
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end    = min(start + batch_size, n)
            nums_t = torch.tensor(scaled[start:end], dtype=torch.float32).to(device)
            out    = model(
                gp_t[start:end], dr_t[start:end], tm_t[start:end],
                yr_t[start:end], wx_t[start:end], nums_t,
            )
            batch_preds = out.squeeze().tolist() if (end - start) > 1 else [out.item()]
            preds.extend(p + circuit_offset for p in batch_preds)

    # Rank by TOTAL RACE TIME = avg_lap_time × laps + num_pit_stops × pit_loss
    # Ranking on avg_lap_time alone always favoured more stops since more stops
    # = shorter stints = lower deg = better avg. But 3 stops costs ~75s in pit lane.
    total_laps = int(race_info.get("total_laps_completed", 57))
    gp_name    = race_info.get("gp_name", "")
    pit_loss   = PIT_LOSS_BY_CIRCUIT.get(gp_name, float(preprocessors.get("avg_pit_mean", 24.0)))

    def total_race_time(avg_lap_time, cand):
        return avg_lap_time * total_laps + cand["num_pits"] * pit_loss

    return sorted(
        zip(preds, candidates),
        key=lambda x: total_race_time(x[0], x[1]),
    )


# ──────────────────────────────────────────────────────────────────────────────
# 7.  OPTIMIZER: GENERATE ALL VALID STRATEGY CANDIDATES
# ──────────────────────────────────────────────────────────────────────────────

MAX_STINT_LENGTHS_BY_TRACK = {
    # 🟢 VERY LOW DEGRADATION (Street circuits & smooth asphalt)
    "Monaco Grand Prix":          {"SOFT": 35, "MEDIUM": 55, "HARD": 78},
    "Azerbaijan Grand Prix":      {"SOFT": 22, "MEDIUM": 35, "HARD": 51},
    "Las Vegas Grand Prix":       {"SOFT": 20, "MEDIUM": 35, "HARD": 50},
    "Singapore Grand Prix":       {"SOFT": 22, "MEDIUM": 40, "HARD": 62},
    "Italian Grand Prix":         {"SOFT": 22, "MEDIUM": 35, "HARD": 53},
    "Russian Grand Prix":         {"SOFT": 22, "MEDIUM": 38, "HARD": 53},
    "Canadian Grand Prix":        {"SOFT": 25, "MEDIUM": 45, "HARD": 65},
    
    # 🟡 MEDIUM DEGRADATION (Standard racing circuits)
    "Abu Dhabi Grand Prix":       {"SOFT": 18, "MEDIUM": 32, "HARD": 45},
    "Saudi Arabian Grand Prix":   {"SOFT": 20, "MEDIUM": 35, "HARD": 50},
    "Australian Grand Prix":      {"SOFT": 20, "MEDIUM": 35, "HARD": 55},
    "Miami Grand Prix":           {"SOFT": 18, "MEDIUM": 32, "HARD": 45},
    "Mexico City Grand Prix":     {"SOFT": 22, "MEDIUM": 38, "HARD": 55},
    "Emilia Romagna Grand Prix":  {"SOFT": 22, "MEDIUM": 35, "HARD": 50},
    "Hungarian Grand Prix":       {"SOFT": 20, "MEDIUM": 35, "HARD": 50},
    "French Grand Prix":          {"SOFT": 18, "MEDIUM": 30, "HARD": 45},
    "Portuguese Grand Prix":      {"SOFT": 20, "MEDIUM": 35, "HARD": 50},
    
    # 🟠 HIGH DEGRADATION (High-speed corners, traction limits)
    "Austrian Grand Prix":        {"SOFT": 20, "MEDIUM": 35, "HARD": 50},
    "Styrian Grand Prix":         {"SOFT": 20, "MEDIUM": 35, "HARD": 50}, 
    "Dutch Grand Prix":           {"SOFT": 18, "MEDIUM": 30, "HARD": 50},
    "Spanish Grand Prix":         {"SOFT": 18, "MEDIUM": 30, "HARD": 48},
    "São Paulo Grand Prix":       {"SOFT": 18, "MEDIUM": 30, "HARD": 45},
    "United States Grand Prix":   {"SOFT": 16, "MEDIUM": 28, "HARD": 42},
    "Chinese Grand Prix":         {"SOFT": 16, "MEDIUM": 28, "HARD": 42},
    
    # 🔴 VERY HIGH DEGRADATION (Abrasive asphalt / extreme tire load)
    "Bahrain Grand Prix":         {"SOFT": 15, "MEDIUM": 25, "HARD": 40},
    "Japanese Grand Prix":        {"SOFT": 15, "MEDIUM": 25, "HARD": 40},
    "British Grand Prix":         {"SOFT": 15, "MEDIUM": 25, "HARD": 40},
    "Turkish Grand Prix":         {"SOFT": 15, "MEDIUM": 25, "HARD": 40},
    "Qatar Grand Prix":           {"SOFT": 15, "MEDIUM": 22, "HARD": 35},
    
    # 🔵 SPECIAL: LONG LAP EXCEPTION (Spa)
    "Belgian Grand Prix":         {"SOFT": 12, "MEDIUM": 20, "HARD": 30},
    
    # DEFAULT FALLBACK
    "DEFAULT":                    {"SOFT": 20, "MEDIUM": 32, "HARD": 45}
}


def validate_strategy(total_laps: int, starting_compound: str, pit_stops: List[PitStop], gp_name: str) -> bool:
    track_limits = MAX_STINT_LENGTHS_BY_TRACK.get(gp_name, MAX_STINT_LENGTHS_BY_TRACK["DEFAULT"])
    
    MAX_STINT_LENGTHS = {
        "SOFT": track_limits["SOFT"],
        "MEDIUM": track_limits["MEDIUM"],
        "HARD": track_limits["HARD"],
        "INTERMEDIATE": 40,
        "WET": 60
    }
    
    MIN_STINT_LENGTH = 10  
    
    compound_counts = {"SOFT": 0, "MEDIUM": 0, "HARD": 0, "INTERMEDIATE": 0, "WET": 0}
    compound_counts[starting_compound] += 1
    
    compounds_used = {starting_compound}
    current_lap = 1
    current_compound = starting_compound
    
    for stop in pit_stops:
        pit_lap = stop.lap
        next_compound = stop.compound
        compound_counts[next_compound] += 1
        
        stint_laps = pit_lap - current_lap
        
        if stint_laps < MIN_STINT_LENGTH:
            return False
        if stint_laps > MAX_STINT_LENGTHS.get(current_compound, total_laps):
            return False
            
        if current_compound == "HARD" and stint_laps < 18:
            return False
            
        compounds_used.add(next_compound)
        current_lap = pit_lap
        current_compound = next_compound
        
    final_stint_laps = total_laps - current_lap + 1
    if final_stint_laps > MAX_STINT_LENGTHS.get(current_compound, total_laps):
        return False
    if current_compound == "HARD" and final_stint_laps < 18:
        return False
        
    if compound_counts["HARD"] > 2 or compound_counts["MEDIUM"] > 2:
        return False
        
    if "INTERMEDIATE" not in compounds_used and "WET" not in compounds_used:
        if len(compounds_used) < 2:
            return False
            
    return True


def generate_candidates(
    total_laps: int,
    weather_code: float,
    gp_name: str,
    lap_step: int = 5,
    min_stint: int = 10,
) -> list:
    is_wet        = weather_code >= 3.0
    dry_compounds = ["SOFT", "MEDIUM", "HARD"]
    wet_compounds = ["INTERMEDIATE", "WET"]
    all_compounds = dry_compounds + (wet_compounds if is_wet else [])

    possible_laps = list(range(min_stint, total_laps - min_stint + 1, lap_step))
    candidates    = []

    for num_pits in range(0, 4):
        num_stints = num_pits + 1

        valid_seqs = []
        for seq in itertools.product(all_compounds, repeat=num_stints):
            if any(seq[i] == seq[i + 1] for i in range(len(seq) - 1)):
                continue
            if not is_wet and num_stints > 1:
                if len({c for c in seq if c in dry_compounds}) < 2:
                    continue
            valid_seqs.append(seq)

        if num_pits == 0:
            pit_combos = [()]
        else:
            pit_combos = []
            for combo in itertools.combinations(possible_laps, num_pits):
                boundaries = (0,) + combo + (total_laps,)
                if all(
                    boundaries[i + 1] - boundaries[i] >= min_stint
                    for i in range(len(boundaries) - 1)
                ):
                    pit_combos.append(combo)

        for seq in valid_seqs:
            for pit_combo in pit_combos:
                starting = seq[0]
                ps_list  = [
                    PitStop(lap=pit_combo[i], compound=seq[i + 1])
                    for i in range(num_pits)
                ]
                
                if not validate_strategy(total_laps, starting, ps_list, gp_name):
                    continue
                
                sf  = derive_strategy_features(starting, ps_list, total_laps)
                deg = derive_degradation_features(starting, ps_list, total_laps)
                # Merge strategy + degradation into one feature dict
                all_features = {**sf, **deg}

                candidates.append({
                    "num_pits": num_pits,
                    "sf": all_features,
                    "starting_compound": starting,
                    "pit_stops": ps_list
                })

    return candidates


# ──────────────────────────────────────────────────────────────────────────────
# 8.  ENDPOINT
# ──────────────────────────────────────────────────────────────────────────────

@app.post("/simulate")
def simulate(strategy: StrategyInput):
    baseline = historical_df[
        (historical_df["Year"]        == strategy.year) &
        (historical_df["gp_name"]     == strategy.gp_name) &
        (historical_df["driver_code"] == strategy.driver_code)
    ]
    if baseline.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No data for {strategy.driver_code} at "
                f"{strategy.gp_name} {strategy.year}. "
                f"Check driver_code, gp_name, and year."
            ),
        )

    race_info    = baseline.iloc[-1]
    team_name    = race_info["team_name"]
    starting_grid_position = int(race_info["grid_position"])  # GHOST RACE INTEGRATION
    weather_code = float(race_info["weather_code"])
    total_laps   = int(race_info["total_laps_completed"])
    actual_time  = float(race_info["avg_lap_time_circuit"])

    # Grab the whole grid for this specific race to compare against
    race_grid = historical_df[
        (historical_df["Year"] == strategy.year) &
        (historical_df["gp_name"] == strategy.gp_name)
    ]

    for ps in strategy.pit_stops:
        if ps.lap >= total_laps:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Pit stop on lap {ps.lap} is invalid — this race only "
                    f"has {total_laps} laps."
                ),
            )

    try:
        gp_idx      = int(preprocessors["le_gp"].transform([strategy.gp_name])[0])
        driver_idx  = int(preprocessors["le_driver"].transform([strategy.driver_code])[0])
        team_idx    = int(preprocessors["le_team"].transform([team_name])[0])
        year_idx    = int(preprocessors["le_year"].transform([strategy.year])[0])
        weather_idx = int(preprocessors["le_weather"].transform([weather_code])[0])
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Encoding error: {e}")

    shared = dict(
        race_info=race_info,
        gp_idx=gp_idx, driver_idx=driver_idx,
        team_idx=team_idx, year_idx=year_idx, weather_idx=weather_idx,
    )

    # ── 1. User strategy ──────────────────────────────────────────────────────
    user_sf   = derive_strategy_features(
        strategy.starting_compound, strategy.pit_stops, total_laps
    )
    user_deg  = derive_degradation_features(
        strategy.starting_compound, strategy.pit_stops, total_laps
    )
    user_all_features = {**user_sf, **user_deg}
    user_pits = len(strategy.pit_stops)
    user_time = predict_single(
        num_pit_stops=float(user_pits),
        strategy_features=user_all_features,
        **shared,
    )

    user_stints = _build_stint_summary(
        strategy.starting_compound, strategy.pit_stops, total_laps
    )

    user_result = {
        "starting_compound":      strategy.starting_compound,
        "pit_stops":              [{"lap": p.lap, "compound": p.compound}
                                   for p in strategy.pit_stops],
        "stints":                 user_stints,
        "num_pit_stops":          user_pits,
        "predicted_avg_lap_time": user_time,
        "actual_avg_lap_time":    round(actual_time, 3),
        "estimated_finishing_position": estimate_finishing_position(
            user_time, strategy.driver_code, race_grid,
            num_pit_stops=user_pits, gp_name=strategy.gp_name,
        )
    }

    # ── 2. Optimizer ──────────────────────────────────────────────────────────
    candidates = generate_candidates(total_laps, weather_code, strategy.gp_name)

    if not candidates:
        optimal_result = {**user_result, "note": "Optimizer found no alternatives."}
        best_time = user_time
    else:
        ranked    = predict_batch(candidates=candidates, **shared)
        best_time, best_cand = ranked[0]
        best_time = round(best_time, 3)

        opt_starting = best_cand["starting_compound"]
        opt_pit_laps = best_cand["pit_stops"]
        opt_pits     = best_cand["num_pits"]
        
        opt_stints = _build_stint_summary(opt_starting, opt_pit_laps, total_laps)

        optimal_result = {
            "starting_compound":      opt_starting,
            "pit_stops":              [{"lap": p.lap, "compound": p.compound}
                                       for p in opt_pit_laps],
            "stints":                 opt_stints,
            "num_pit_stops":          opt_pits,
            "predicted_avg_lap_time": best_time,
            "strategies_evaluated":   len(candidates),
            "estimated_finishing_position": estimate_finishing_position(
                best_time, strategy.driver_code, race_grid,
                num_pit_stops=opt_pits, gp_name=strategy.gp_name,
            )
        }

    # ── Response ──────────────────────────────────────────────────────────────
    delta = round(user_time - best_time, 3)
    
    user_pos = user_result["estimated_finishing_position"]
    opt_pos = optimal_result["estimated_finishing_position"]
    pos_gained = user_pos - opt_pos
    
    if pos_gained > 0:
        verdict = f"The optimal strategy gains you {pos_gained} positions! 🚀"
    elif pos_gained < 0:
        verdict = "Your strategy is actually better for track position!"
    else:
        verdict = f"Both strategies finish P{opt_pos}, but optimal is {abs(delta)}s/lap faster."

    return {
        "year":             strategy.year,
        "gp_name":          strategy.gp_name,
        "driver_code":      strategy.driver_code,
        "team_name":        team_name,
        "starting_grid_position": starting_grid_position,
        "race_laps":        total_laps,
        "user_strategy":    user_result,
        "optimal_strategy": optimal_result,
        "time_delta_per_lap": delta,
        "verdict":          verdict,
    }


# ──────────────────────────────────────────────────────────────────────────────
# 9.  RESPONSE FORMATTING HELPERS
# ──────────────────────────────────────────────────────────────────────────────

def _build_stint_summary(
    starting_compound: str,
    pit_stops: List[PitStop],
    total_laps: int,
) -> list:
    """Build a human-readable list of {compound, laps, lap_start, lap_end}."""
    stints   = []
    prev_lap = 1
    current  = starting_compound.upper()

    for ps in pit_stops:
        stints.append({
            "compound":  current,
            "lap_start": prev_lap,
            "lap_end":   ps.lap - 1,
            "laps":      ps.lap - prev_lap,
        })
        prev_lap = ps.lap
        current  = ps.compound.upper()

    stints.append({
        "compound":  current,
        "lap_start": prev_lap,
        "lap_end":   total_laps,
        "laps":      total_laps - prev_lap + 1,
    })
    return stints


def estimate_finishing_position(
    simulated_avg_lap_time: float,
    driver_code: str,
    race_grid: pd.DataFrame,
    num_pit_stops: int = 1,
    gp_name: str = "",
) -> int:
    """
    Ranks the simulated TOTAL RACE TIME against the field to estimate finish position.

    Total race time = (avg_lap_time × total_laps) + (num_pit_stops × pit_loss)

    Previously this ranked on avg_lap_time alone, which meant a 3-stop strategy
    always looked faster than a 1-stop even though 3 stops costs ~75s in pit lane.
    Now pit loss is factored in using circuit-specific values from PIT_LOSS_BY_CIRCUIT.
    """
    grid = race_grid[['driver_code', 'avg_lap_time_circuit', 'num_pit_stops',
                       'total_laps_completed']].copy()

    # Circuit-specific pit loss — fall back to global mean if not in dict
    pit_loss = PIT_LOSS_BY_CIRCUIT.get(gp_name, float(preprocessors["avg_pit_mean"]))

    # Compute total race time for every driver on the grid using their actual data
    grid['total_race_time'] = (
        grid['avg_lap_time_circuit'] * grid['total_laps_completed']
        + grid['num_pit_stops'].fillna(1) * pit_loss
    )

    # Overwrite our driver's total race time with the simulated strategy
    total_laps = float(race_grid['total_laps_completed'].iloc[0])
    our_total  = simulated_avg_lap_time * total_laps + num_pit_stops * pit_loss
    grid.loc[grid['driver_code'] == driver_code, 'total_race_time'] = our_total

    grid = grid.dropna(subset=['total_race_time'])
    grid = grid.sort_values('total_race_time').reset_index(drop=True)

    try:
        rank = grid.index[grid['driver_code'] == driver_code].tolist()[0] + 1
        return int(rank)
    except IndexError:
        return 20

# Add backend/ to sys.path so phase_3 is importable regardless of cwd
_BACKEND = os.path.abspath(os.path.join(_ROOT, "../.."))
if _BACKEND not in sys.path:
    sys.path.insert(0, _BACKEND)

from phase_3.core.pace_anchor import PIT_LOSS_BY_CIRCUIT
from phase_3.api.routes_2026 import router_2026
app.include_router(router_2026, prefix="/2026", tags=["2026 Season"])

from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])