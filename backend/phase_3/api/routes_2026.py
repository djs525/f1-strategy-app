"""
phase_3/api/routes_2026.py
============================
New FastAPI routes for the 2026 season predictor.
Mounts onto the existing phase_2 API as a sub-application
WITHOUT modifying a single line of phase_2/api/main.py.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, validator, conlist
from typing import List, Optional
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from phase_3.adapters.model_adapter import (
    predict_2026,
    TEAM_PACE_SCALING_2026,
    DRIVER_PACE_DELTA_2026,
    DRIVER_DELTA_CONFIDENCE,
    DRIVER_DELTA_OBSERVATIONS,
)
from phase_3.interactive.insights_engine import (
    generate_post_race_insights,
    format_insights_report,
    _state,
    FIA_POINTS,
)
from phase_3.core.pace_anchor import (
    TESTING_PACE_2026,
    derive_strategy_features_2026,
    derive_degradation_features,
)
from phase_3.core.roster_2026 import ROSTER_2026, TEAMS_2026, CALENDAR_2026, CIRCUIT_TYPES
from phase_3.core.drivers_config import DRIVERS

router_2026 = APIRouter()

VALID_COMPOUNDS     = {"SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"}
DRY_COMPOUNDS       = {"SOFT", "MEDIUM", "HARD"}
WET_COMPOUNDS       = {"INTERMEDIATE", "WET"}
# In a wet race you must use at least one wet/inter compound.
# In a dry race you cannot use wet/inter compounds.


class PitStop2026(BaseModel):
    lap:      int = Field(..., ge=1)
    compound: str

    @validator("compound")
    def validate_compound(cls, v):
        v = v.upper().strip()
        if v not in VALID_COMPOUNDS:
            raise ValueError(f"compound must be one of {VALID_COMPOUNDS}")
        return v

class PredictRequest(BaseModel):
    gp_name:            str            = Field(..., example="Australian Grand Prix")
    driver_code:        str            = Field(..., example="HAM")
    team_name:          str            = Field(..., example="Ferrari")
    starting_compound:  str            = Field(..., example="SOFT")
    pit_stops:          conlist(PitStop2026, max_length=3) = Field(...)
    grid_position:      int            = Field(default=10, ge=1, le=22)
    total_laps:         int            = Field(default=57, ge=1)
    weather_code:       float          = Field(default=1.0)
    race_number:        int            = Field(default=1, ge=1, le=24)

    @validator("starting_compound")
    def validate_starting(cls, v):
        v = v.upper().strip()
        if v not in VALID_COMPOUNDS:
            raise ValueError(f"starting_compound must be one of {VALID_COMPOUNDS}")
        return v

    @validator("pit_stops", always=True)
    def validate_wet_dry_consistency(cls, pit_stops, values):
        weather   = values.get("weather_code", 1.0)
        starting  = values.get("starting_compound", "SOFT")
        is_wet    = weather >= 3.0

        all_compounds = [starting] + [p.compound for p in pit_stops]

        if is_wet:
            # Wet race: must use at least one wet-weather compound somewhere in
            # the strategy, but dry slicks are allowed (track-drying transitions
            # are a normal and common part of wet races).
            has_wet_compound = any(c in WET_COMPOUNDS for c in all_compounds)
            if not has_wet_compound:
                raise ValueError(
                    "Wet race (weather_code ≥ 3.0) requires at least one "
                    "INTERMEDIATE or WET compound somewhere in the strategy."
                )
        else:
            # Dry race: no wet-weather compounds allowed at all.
            has_wet_compound = any(c in WET_COMPOUNDS for c in all_compounds)
            if has_wet_compound:
                raise ValueError(
                    "Dry race (weather_code < 3.0) cannot use INTERMEDIATE or WET compounds."
                )

        return pit_stops

class RaceResultDriver(BaseModel):
    driver:      str  = Field(..., example="HAM")
    team:        str  = Field(..., example="Ferrari")
    finish_pos:  int  = Field(..., ge=1, le=22)
    fastest_lap: bool = Field(default=False)
    dnf:         bool = Field(default=False)
    dnf_reason:  Optional[str] = None

class RaceResultInput(BaseModel):
    round_num:   int             = Field(..., ge=1, le=24, example=1)
    race_name:   str             = Field(..., example="Australian Grand Prix")
    circuit_id:  str             = Field(..., example="albert_park")
    results:     List[RaceResultDriver]
    pre_race_predictions: Optional[dict] = Field(default=None)

    @validator("results")
    def must_have_at_least_winner(cls, v):
        if not v:
            raise ValueError("Must include at least the race winner.")
        finish_positions = [r.finish_pos for r in v if not r.dnf]
        if 1 not in finish_positions:
            raise ValueError("Results must include P1 (race winner).")
        return v

# =============================================================================
# ENDPOINTS
# =============================================================================

@router_2026.post("/predict")
def predict_2026_strategy(req: PredictRequest):
    pit_stops_raw = [{"lap": p.lap, "compound": p.compound} for p in req.pit_stops]

    try:
        strategy_features = derive_strategy_features_2026(
            starting_compound=req.starting_compound,
            pit_stops=pit_stops_raw,
            total_laps=req.total_laps,
        )
        degradation_features = derive_degradation_features(
            starting_compound=req.starting_compound,
            pit_stops=pit_stops_raw,
            total_laps=req.total_laps,
        )
        # Merge into one feature dict — model adapter receives both
        all_features = {**strategy_features, **degradation_features}
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature derivation failed: {e}")

    try:
        result = predict_2026(
            gp_name           = req.gp_name,
            driver_code       = req.driver_code,
            team_name         = req.team_name,
            num_pit_stops     = len(req.pit_stops),
            strategy_features = all_features,
            grid_position     = req.grid_position,
            total_laps        = req.total_laps,
            weather_code      = req.weather_code,
            race_number       = req.race_number,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    stints = _build_stint_summary(req.starting_compound, req.pit_stops, req.total_laps)

    adaptations = []
    if result["is_new_driver"]:
        adaptations.append(f"Model used '{result['resolved_driver']}' as historical analogue.")
    if result["is_new_team"]:
        adaptations.append(f"Team '{req.team_name}' is new — model used '{result['resolved_team']}' as proxy.")
    if result["is_new_circuit"]:
        adaptations.append(f"'{req.gp_name}' is a new circuit — model used '{result['resolved_gp']}' as proxy.")

    adaptations.append(f"2026 team pace scaling for {req.team_name}: ×{result['pace_scaling_factor']:.4f}")

    driver_delta     = result["driver_delta"]
    delta_confidence = result["driver_delta_confidence"]
    n_obs            = DRIVER_DELTA_OBSERVATIONS.get(req.driver_code, 0)
    delta_source     = "pre-season testing" if n_obs == 0 else f"{n_obs} race{'s' if n_obs != 1 else ''} of data"
    adaptations.append(
        f"Driver delta for {req.driver_code}: ×{driver_delta:.5f} "
        f"({'faster' if driver_delta < 1.0 else 'slower'} than team baseline, "
        f"{delta_confidence:.0%} confidence from {delta_source})."
    )

    if result.get("cadillac_position_penalty", 0) > 0:
        adaptations.append(f"Cadillac ramp-up penalty applied: +{result['cadillac_position_penalty']:.1f} positions.")

    testing_context = None
    if req.driver_code in TESTING_PACE_2026:
        testing_context = {
            "driver":         req.driver_code,
            "testing_best":   TESTING_PACE_2026[req.driver_code]["best_lap"],
            "testing_rank":   TESTING_PACE_2026[req.driver_code]["testing_rank"],
        }

    return {
        "gp_name":                req.gp_name,
        "driver_code":            req.driver_code,
        "team_name":              req.team_name,
        "race_number":            req.race_number,
        "strategy": {
            "starting_compound":  req.starting_compound,
            "pit_stops":          [{"lap": p.lap, "compound": p.compound} for p in req.pit_stops],
            "stints":             stints,
            "num_pit_stops":      len(req.pit_stops),
        },
        "prediction": {
            "avg_lap_time_raw":   result["predicted_avg_lap_time_raw"],
            "avg_lap_time_2026":  result["predicted_avg_lap_time_2026"],
            "predicted_position": result.get("predicted_position", "Pending position logic"), # FIXED HERE
            "pace_scaling":       result["pace_scaling_factor"],
            "year_proxy":         result["year_proxy_used"],
        },
        "testing_context":        testing_context,
        "model_adaptations":      adaptations,
        "note": "Prediction uses the phase_2 model with 2026 regulation scaling.",
    }

@router_2026.post("/race-result")
def enter_race_result(req: RaceResultInput):
    results_dicts = [r.dict() for r in req.results]

    insights = generate_post_race_insights(
        round_num            = req.round_num,
        race_name            = req.race_name,
        circuit_id           = req.circuit_id,
        actual_results       = results_dicts,
        pre_race_predictions = req.pre_race_predictions,
    )
    text_report = format_insights_report(insights)

    return {
        "status":      "Race result recorded successfully.",
        "insights":    insights,
        "text_report": text_report,
    }

@router_2026.get("/standings")
def get_standings():
    wdc  = _state.get_wdc_standings()
    wcc  = _state.get_wcc_standings()
    done = _state.state["races_completed"]
    max_rem = (24 - done) * 26

    wdc_with_math = []
    leader_pts    = wdc[0][1] if wdc else 0
    for driver, pts in wdc:
        gap         = leader_pts - pts
        can_win     = (pts + max_rem) >= leader_pts
        
        # Clean 3-letter lookup
        cfg = DRIVERS.get(driver.upper(), {})

        wdc_with_math.append({
            "position":              wdc.index((driver, pts)) + 1,
            "driver":                driver,
            "points":                pts,
            "gap_to_leader":         gap,
            "mathematically_alive":  can_win,
            "team":                  cfg.get("team_2026", "Unknown"),
        })

    return {
        "season":           2026,
        "races_completed":  done,
        "races_remaining":  24 - done,
        "max_remaining_pts": max_rem,
        "wdc":              wdc_with_math,
        "wcc": [{"position": i+1, "team": t, "points": p} for i, (t, p) in enumerate(wcc)],
    }

@router_2026.get("/calendar")
def get_calendar():
    completed = set(int(k) for k in _state.state["race_results"].keys())
    return {
        "season": 2026,
        "races": [
            {
                **race,
                "status": "completed" if race["round"] in completed else
                          ("next" if race["round"] == min((r["round"] for r in CALENDAR_2026 if r["round"] not in completed), default=1)
                           else "upcoming"),
                "circuit_type": CIRCUIT_TYPES.get(race["circuit"], "permanent"),
            }
            for race in CALENDAR_2026
        ],
    }

@router_2026.get("/driver/{driver_code}")
def get_driver_profile(driver_code: str):
    driver_code = driver_code.upper()
    
    # Clean 3-letter lookup
    cfg = DRIVERS.get(driver_code)

    season_results = []
    for round_num, race_data in _state.state["race_results"].items():
        for res in race_data["results"]:
            if res["driver"] == driver_code:
                season_results.append({
                    "round":         int(round_num),
                    "race":          race_data["race_name"],
                    "finish_pos":    res["finish_pos"],
                    "dnf":           res.get("dnf", False),
                    "fastest_lap":   res.get("fastest_lap", False),
                    "points":        FIA_POINTS.get(res["finish_pos"], 0) if not res.get("dnf") else 0,
                })

    total_pts = sum(r["points"] for r in season_results)
    wins      = sum(1 for r in season_results if r["finish_pos"] == 1)
    podiums   = sum(1 for r in season_results if r["finish_pos"] <= 3)
    testing   = TESTING_PACE_2026.get(driver_code)

    return {
        "driver_code":    driver_code,
        "name":           cfg["name"] if cfg else driver_code,
        "team_2026":      cfg["team_2026"] if cfg else "Unknown",
        "debut_year":     cfg["debut_year"] if cfg else None,
        "career_seasons": len(cfg["active_years"]) if cfg else 0,
        "testing_2026":   testing,
        "season_2026": {
            "races_entered": len(season_results),
            "points":        total_pts,
            "wins":          wins,
            "podiums":       podiums,
            "results":       season_results,
        },
        "wdc_position": next((i+1 for i, (d, _) in enumerate(_state.get_wdc_standings()) if d == driver_code), None),
    }

@router_2026.get("/pace-scaling")
def get_pace_scaling():
    return {
        "description": "Multiplicative corrections applied to the phase_2 model's output.",
        "team_scaling": {
            team: {
                "factor":         factor,
                "interpretation": (
                    f"Car is {abs(1-factor)*100:.2f}% "
                    f"{'faster' if factor < 1 else 'slower'} than 2025 baseline"
                ),
            }
            for team, factor in sorted(TEAM_PACE_SCALING_2026.items(), key=lambda x: x[1])
        },
        "note": "Team scaling × driver delta = total prediction correction. See /driver-deltas for intra-team breakdown.",
    }


@router_2026.get("/driver-deltas")
def get_driver_deltas():
    """
    Returns per-driver pace deltas — the intra-team skill component of predictions.

    Each driver has a multiplicative delta applied ON TOP of their team's scaling factor.
    A delta < 1.0 means the driver is faster than their team's baseline (good).
    A delta > 1.0 means slower.

    Confidence starts at 0% (testing-only prior) and reaches 100% after 5 races.
    Low-confidence deltas should be treated as weak priors, not strong signals.
    """
    rows = []
    for driver_code, delta in sorted(DRIVER_PACE_DELTA_2026.items(), key=lambda x: x[1]):
        from phase_3.core.drivers_config import DRIVERS
        cfg        = DRIVERS.get(driver_code, {})
        team       = cfg.get("team_2026", "Unknown")
        confidence = DRIVER_DELTA_CONFIDENCE.get(driver_code, 0.0)
        n_obs      = DRIVER_DELTA_OBSERVATIONS.get(driver_code, 0)
        team_scale = TEAM_PACE_SCALING_2026.get(team, 1.0)

        rows.append({
            "driver":          driver_code,
            "name":            cfg.get("name", driver_code),
            "team":            team,
            "driver_delta":    delta,
            "team_scale":      team_scale,
            "combined_factor": round(team_scale * delta, 5),
            "direction":       "faster" if delta < 1.0 else "slower",
            "vs_baseline_pct": round((1 - delta) * 100, 3),
            "confidence":      confidence,
            "observations":    n_obs,
            "data_source":     "race data" if n_obs > 0 else "pre-season testing prior",
        })

    return {
        "season": 2026,
        "description": (
            "Per-driver multiplicative delta applied on top of team scaling. "
            "combined_factor = team_scale × driver_delta = total prediction multiplier."
        ),
        "drivers": rows,
    }

@router_2026.get("/grid")
def get_2026_grid():
    grid = []
    for team_name, driver_codes in TEAMS_2026.items():
        for dc in driver_codes:
            # Clean 3-letter lookup
            cfg     = DRIVERS.get(dc.upper(), {})
            testing = TESTING_PACE_2026.get(dc.upper())
            
            grid.append({
                "driver_code":  dc.upper(),
                "name":         cfg.get("name", dc),
                "team":         team_name,
                "debut_year":   cfg.get("debut_year"),
                "is_rookie":    dc.upper() == "LIN",
                "testing_pace": testing,
            })
    grid.sort(key=lambda x: (x["testing_pace"]["best_lap"] if x["testing_pace"] else 9999))
    return {"season": 2026, "total_drivers": len(grid), "grid": grid}

# =============================================================================
# HELPERS
# =============================================================================

def _build_stint_summary(starting_compound, pit_stops, total_laps):
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