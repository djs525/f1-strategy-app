"""
phase_3/core/pace_anchor.py
=============================
Hardcoded 2026 Bahrain pre-season testing pace data.
This is the primary calibration source for the 2026 scaling factors.

Also contains derive_strategy_features_2026() — a clean re-export
of the phase_2 feature derivation logic for use by phase_3 routes.
"""

# Best lap times from combined Bahrain testing (Feb 11-20, 2026)
# Source: Published timing sheets. Missing entries = no representative lap set.
TESTING_PACE_2026 = {
    "LEC": {"best_lap": 91.992, "testing_rank": 1,  "team": "Ferrari"},
    "ANT": {"best_lap": 92.803, "testing_rank": 2,  "team": "Mercedes"},
    "PIA": {"best_lap": 92.861, "testing_rank": 3,  "team": "McLaren"},
    "NOR": {"best_lap": 92.871, "testing_rank": 4,  "team": "McLaren"},
    "VER": {"best_lap": 93.109, "testing_rank": 5,  "team": "Red Bull"},
    "RUS": {"best_lap": 93.197, "testing_rank": 6,  "team": "Mercedes"},
    "HAM": {"best_lap": 93.408, "testing_rank": 7,  "team": "Ferrari"},
    "GAS": {"best_lap": 93.421, "testing_rank": 8,  "team": "Alpine"},
    "BEA": {"best_lap": 93.487, "testing_rank": 9,  "team": "Haas"},
    "BOR": {"best_lap": 93.755, "testing_rank": 10, "team": "Audi"},
    "COL": {"best_lap": 93.818, "testing_rank": 11, "team": "Alpine"},
    "HUL": {"best_lap": 93.987, "testing_rank": 12, "team": "Audi"},
    "LIN": {"best_lap": 94.149, "testing_rank": 13, "team": "Racing Bulls"},
    "OCO": {"best_lap": 94.201, "testing_rank": 14, "team": "Haas"},
    "HAD": {"best_lap": 94.260, "testing_rank": 15, "team": "Red Bull"},
    "SAI": {"best_lap": 94.342, "testing_rank": 16, "team": "Williams"},
    "LAW": {"best_lap": 94.532, "testing_rank": 17, "team": "Racing Bulls"},
    # Missing from published composite: ALB, ALO, STR, PER, BOT (Cadillac/Aston Martin issues)
    # These are assigned estimated values based on team tier
    "ALB": {"best_lap": 94.700, "testing_rank": 18, "team": "Williams",     "estimated": True},
    "ALO": {"best_lap": 94.800, "testing_rank": 19, "team": "Aston Martin", "estimated": True},
    "STR": {"best_lap": 95.200, "testing_rank": 20, "team": "Aston Martin", "estimated": True,
            "note": "Battery issues in testing — very limited running"},
    "PER": {"best_lap": 95.500, "testing_rank": 21, "team": "Cadillac",     "estimated": True},
    "BOT": {"best_lap": 95.700, "testing_rank": 22, "team": "Cadillac",     "estimated": True},
}

COMPOUND_TO_COL = {
    "SOFT":         "laps_on_soft",
    "MEDIUM":       "laps_on_medium",
    "HARD":         "laps_on_hard",
    "INTERMEDIATE": "laps_on_intermediate",
    "WET":          "laps_on_wet",
}

COMPOUND_COLS = [
    "laps_on_soft", "laps_on_medium", "laps_on_hard",
    "laps_on_intermediate", "laps_on_wet",
]

PIT_TIMING_COLS = [
    "first_pit_lap_pct", "second_pit_lap_pct", "third_pit_lap_pct",
]


# Compound hardness weights for the degradation score.
# Softer compounds degrade faster per lap, so they carry more weight
# when computing how hard tyres were stressed across the race.
# Soft=3 (degrades quickly), Hard=1 (most durable), Wet=1 (low temp = low deg)
COMPOUND_HARDNESS = {
    "SOFT":         3.0,
    "MEDIUM":       2.0,
    "HARD":         1.0,
    "INTERMEDIATE": 1.5,
    "WET":          1.0,
}


def derive_strategy_features_2026(
    starting_compound: str,
    pit_stops: list[dict],   # [{"lap": int, "compound": str}]
    total_laps: int,
) -> dict:
    """
    Derives the 8 strategy features expected by the phase_2 model's num_cols.
    This is a clean reimplementation of phase_2's derive_strategy_features()
    that accepts plain dicts instead of Pydantic PitStop objects.
    """
    compound_laps    = {col: 0 for col in COMPOUND_COLS}
    prev_lap         = 1
    current_compound = starting_compound.upper()

    for ps in pit_stops:
        laps_in_stint = ps["lap"] - prev_lap
        col = COMPOUND_TO_COL[current_compound]
        compound_laps[col] += laps_in_stint
        prev_lap         = ps["lap"]
        current_compound = ps["compound"].upper()

    final_laps = total_laps - prev_lap + 1
    compound_laps[COMPOUND_TO_COL[current_compound]] += final_laps

    pit_laps = [ps["lap"] for ps in pit_stops]
    return {
        **compound_laps,
        "first_pit_lap_pct":  pit_laps[0] / total_laps if len(pit_laps) > 0 else 0.0,
        "second_pit_lap_pct": pit_laps[1] / total_laps if len(pit_laps) > 1 else 0.0,
        "third_pit_lap_pct":  pit_laps[2] / total_laps if len(pit_laps) > 2 else 0.0,
    }


def derive_degradation_features(
    starting_compound: str,
    pit_stops: list[dict],   # [{"lap": int, "compound": str}]
    total_laps: int,
) -> dict:
    """
    Derives the 3 tyre degradation features.

    avg_tyre_age:
        The weighted average number of laps each tyre was on the car.
        A 1-stop race where the driver ran 30 laps on soft then 27 on hard
        has avg_tyre_age = (30 + 27) / 2 = 28.5 laps.
        Higher = tyres pushed further = more performance loss late in stints.

    max_stint_laps:
        The single longest stint in the race. This is the primary signal
        for "did the driver push a tyre way past its optimal window?"
        A 45-lap soft stint at Bahrain is catastrophic. A 45-lap hard
        stint at Monaco is perfectly fine. The model learns this nuance
        from the interaction with compound_cols and circuit embeddings.

    deg_score:
        max_stint_laps × compound_hardness of that stint's compound.
        Penalises running soft tyres for too long much more than hard tyres.
        A 30-lap soft (30 × 3 = 90) is far more damaging than a 30-lap hard
        (30 × 1 = 30). This is the key feature that punishes bad strategies.
    """
    # Build list of (compound, stint_length) tuples
    stints = []
    prev_lap         = 1
    current_compound = starting_compound.upper()

    for ps in pit_stops:
        stint_laps = ps["lap"] - prev_lap
        stints.append((current_compound, stint_laps))
        prev_lap         = ps["lap"]
        current_compound = ps["compound"].upper()

    # Final stint
    final_laps = total_laps - prev_lap + 1
    stints.append((current_compound, final_laps))

    if not stints:
        return {"avg_tyre_age": 0.0, "max_stint_laps": 0.0, "deg_score": 0.0}

    stint_lengths = [s[1] for s in stints]
    avg_tyre_age  = sum(stint_lengths) / len(stint_lengths)
    max_idx       = stint_lengths.index(max(stint_lengths))
    max_compound, max_laps = stints[max_idx]

    deg_score = max_laps * COMPOUND_HARDNESS.get(max_compound, 1.0)

    return {
        "avg_tyre_age":   round(avg_tyre_age, 2),
        "max_stint_laps": float(max_laps),
        "deg_score":      round(deg_score, 2),
    }