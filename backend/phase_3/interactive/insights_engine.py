"""
phase_3/interactive/insights_engine.py
========================================
Generates model-backed narrative insights after the user inputs real
race results. This is the "brain" of the interactive system.

WHAT IT DOES
------------
When the user tells the system "Hamilton won in Australia", the engine:

  1. Compares actual result vs model's pre-race prediction
  2. Calculates the over/under-performance delta per driver
  3. Triggers a Bayesian update of team pace scaling factors
  4. Re-projects the remaining championship season
  5. Generates a human-readable insight report

THEORY — Why post-race updating matters
-----------------------------------------
A model trained on 2018-2025 data has a prior belief about each team's
pace. But 2026 is a regulation-change year — the hierarchy reshuffles.
By updating scaling factors after each real race, we implement a simple
form of "online learning": the model's predictions improve over the season
as more real data arrives.

This is analogous to how a Kalman filter works: you have a prior estimate,
you get a noisy measurement, you update your estimate weighted by how
much you trust the measurement vs the prior.
"""

import sys
import os
import json
from datetime import datetime
from typing import Optional

# Resolve project root (f1/) regardless of where uvicorn is launched from
_HERE         = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, "../.."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from phase_3.adapters.model_adapter import (
    predict_2026,
    update_pace_scaling,
    update_driver_delta,
    TEAM_PACE_SCALING_2026,
    DRIVER_PACE_DELTA_2026,
    DRIVER_DELTA_CONFIDENCE,
    DRIVER_DELTA_OBSERVATIONS,
    CADILLAC_RAMP_UP,
)
from phase_3.core.roster_2026 import ROSTER_2026, TEAMS_2026
from phase_3.core.pace_anchor import TESTING_PACE_2026

# FIA points system (1st through 10th + fastest lap bonus)
FIA_POINTS = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10,
              6: 8,  7: 6,  8: 4,  9: 2,  10: 1}
FASTEST_LAP_BONUS = 1  # Awarded to P1-P10 driver with fastest lap


class ChampionshipState:
    """
    Tracks the live WDC and WCC standings as real results are entered.
    Persists to a JSON file so state survives between sessions.

    THEORY — State persistence:
        The interactive system needs to remember results from previous
        sessions (e.g. you enter R1 today, come back tomorrow for R2).
        We use a simple JSON file — not a database — because the dataset
        is small (24 races × 22 drivers = 528 rows maximum) and JSON is
        human-readable, meaning you can manually edit it if needed.
    """

    def __init__(self, state_path: str = "phase_3/data/championship_state.json"):
        self.state_path = state_path
        self.state      = self._load()

    def _load(self) -> dict:
        try:
            with open(self.state_path, "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return {
                "season": 2026,
                "last_updated": None,
                "races_completed": 0,
                "wdc": {d: 0 for d in ROSTER_2026},
                "wcc": {t: 0 for t in TEAMS_2026},
                "race_results": {},        # round_num → result dict
                "prediction_log": {},      # round_num → pre-race predictions
                "pace_scaling_history": {} # round_num → scaling factors snapshot
            }

    def save(self):
        import os
        os.makedirs(os.path.dirname(self.state_path), exist_ok=True)
        self.state["last_updated"] = datetime.now().isoformat()
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2)

    def record_result(self, round_num: int, race_name: str, results: list[dict]):
        """
        Store a completed race result and update WDC/WCC standings.

        results: list of dicts ordered P1→P22, each with:
            { "driver": "HAM", "team": "Ferrari",
              "finish_pos": 1, "fastest_lap": False, "dnf": False }
        """
        self.state["race_results"][str(round_num)] = {
            "race_name": race_name,
            "results": results,
            "entered_at": datetime.now().isoformat(),
        }
        self.state["races_completed"] = max(
            self.state["races_completed"], round_num
        )

        # Award points
        for res in results:
            driver = res["driver"]
            team   = res["team"]
            pos    = res["finish_pos"]
            points = FIA_POINTS.get(pos, 0)
            if res.get("fastest_lap") and 1 <= pos <= 10:
                points += FASTEST_LAP_BONUS

            if driver in self.state["wdc"]:
                self.state["wdc"][driver] += points
            if team in self.state["wcc"]:
                self.state["wcc"][team] += points

        self.save()

    def get_wdc_standings(self) -> list[tuple]:
        """Returns [(driver_code, points)] sorted descending."""
        return sorted(self.state["wdc"].items(), key=lambda x: x[1], reverse=True)

    def get_wcc_standings(self) -> list[tuple]:
        """Returns [(team_name, points)] sorted descending."""
        return sorted(self.state["wcc"].items(), key=lambda x: x[1], reverse=True)


# Singleton state — shared across the FastAPI app and CLI
_state = ChampionshipState()


# =============================================================================
# CORE INSIGHT GENERATION
# =============================================================================

def generate_post_race_insights(
    round_num: int,
    race_name: str,
    circuit_id: str,
    actual_results: list[dict],
    pre_race_predictions: Optional[dict] = None,
) -> dict:
    """
    Called after the user inputs a completed race result.
    Returns a structured insight report.

    Parameters
    ----------
    round_num           : Race number (1-24)
    race_name           : e.g. "Australian Grand Prix"
    circuit_id          : e.g. "albert_park"
    actual_results      : List of finish results P1→P22
    pre_race_predictions: Dict of {driver_code: predicted_finish_pos}
                          (generated by the season loop before the race)
    """
    insights = {
        "race":           race_name,
        "round":          round_num,
        "generated_at":   datetime.now().isoformat(),
        "driver_insights": [],
        "team_insights":   [],
        "championship":    {},
        "pace_updates":    [],
        "season_outlook":  {},
    }

    # ---- 1. Record result and update standings ---- #
    _state.record_result(round_num, race_name, actual_results)

    # ---- 2. Per-driver analysis ---- #
    for res in actual_results:
        driver   = res["driver"]
        team     = res["team"]
        actual_p = res["finish_pos"]
        is_dnf   = res.get("dnf", False)

        # Compare vs pre-race prediction if available
        pred_p = None
        delta  = None
        narrative = ""

        if pre_race_predictions and driver in pre_race_predictions:
            pred_p = pre_race_predictions[driver]
            delta  = pred_p - actual_p  # Positive = overperformed vs prediction

            if is_dnf:
                narrative = f"DNF — {res.get('dnf_reason', 'mechanical/incident')}. Model had predicted P{pred_p}."
            elif delta >= 5:
                narrative = (
                    f"🔥 Massively overperformed — finished P{actual_p}, "
                    f"model predicted P{pred_p} (+{delta} positions). "
                    f"Either significant car development or exceptional race craft."
                )
            elif delta >= 2:
                narrative = (
                    f"✅ Outperformed prediction by {delta} positions "
                    f"(P{actual_p} vs predicted P{pred_p})."
                )
            elif delta <= -3:
                narrative = (
                    f"⚠️ Underperformed — P{actual_p} vs predicted P{pred_p} "
                    f"({abs(delta)} positions lost). "
                    f"Possible: strategy issue, traffic, incident, or car problem."
                )
            else:
                narrative = (
                    f"Model accuracy: predicted P{pred_p}, finished P{actual_p} "
                    f"(delta: {'+' if delta >= 0 else ''}{delta})."
                )
        else:
            narrative = f"Finished P{actual_p}{'  DNF' if is_dnf else ''}."

        # Special narratives for story drivers
        if team == "Cadillac F1 Team":
            ramp   = CADILLAC_RAMP_UP.get(round_num, 0.0)
            narrative += (
                f" Cadillac race {round_num}/8 of expected ramp-up period "
                f"(current position penalty applied: +{ramp:.1f} pos)."
            )

        # Only Lindblad is a true 2026 rookie — HAD/ANT/BOR/BEA/COL all
        # completed a full 2025 season and are established F1 drivers by now.
        if driver in ("LIN",) and not is_dnf:
            if actual_p <= 10:
                narrative += f" 🌟 Rookie in the points — a strong result for a first-year driver."

        insights["driver_insights"].append({
            "driver":             driver,
            "team":               team,
            "finish_position":    actual_p,
            "predicted_position": pred_p,
            "delta":              delta,
            "dnf":                is_dnf,
            "fastest_lap":        res.get("fastest_lap", False),
            "narrative":          narrative,
        })

    # ---- 3. Team-level analysis ---- #
    team_finishes = {}
    for res in actual_results:
        t = res["team"]
        if t not in team_finishes:
            team_finishes[t] = []
        if not res.get("dnf"):
            team_finishes[t].append(res["finish_pos"])

    for team, finishes in team_finishes.items():
        if not finishes:
            team_note = "Both cars DNF — zero points."
        else:
            avg_finish = sum(finishes) / len(finishes)
            pts        = sum(FIA_POINTS.get(p, 0) for p in finishes)
            team_note  = (
                f"Average finish: P{avg_finish:.1f} | "
                f"Points this race: {pts} | "
                f"Season total: {_state.state['wcc'].get(team, 0)}"
            )
        insights["team_insights"].append({
            "team": team,
            "race_points": sum(FIA_POINTS.get(p, 0) for p in finishes),
            "note": team_note,
        })

    # ---- 4. Bayesian pace scaling updates ---- #
    # We update BOTH team scaling factors AND per-driver deltas based on
    # actual vs predicted performance this race.
    #
    # THEORY — Two-level correction:
    #   Team scale  → corrects for how fast the CAR is vs the model's baseline
    #   Driver delta → corrects for how this DRIVER performs vs their team baseline
    #
    #   By separating these, we avoid conflating car performance with driver skill.
    #   If Ferrari finishes P1 and P8, the team scale updates on both results,
    #   but the driver deltas pull LEC toward faster and HAM toward slower
    #   (or vice versa). Over the season, this produces an accurate intra-team
    #   skill picture independent of car pace.

    # Build driver → predicted_position map for delta updates below
    def _get_pred_pos(pred_val):
        if isinstance(pred_val, dict):
            return pred_val.get("predicted_position")
        return pred_val  # plain int/float from older callers

    # Team-level updates
    for team, finishes in team_finishes.items():
        if not finishes:
            continue
        best_finish = min(finishes)
        if pre_race_predictions:
            team_drivers = [r["driver"] for r in actual_results if r["team"] == team]
            team_preds   = [_get_pred_pos(pre_race_predictions.get(d))
                            for d in team_drivers
                            if pre_race_predictions.get(d) is not None]
            team_preds   = [p for p in team_preds if p is not None]
            if team_preds:
                best_pred  = min(team_preds)
                old_scale  = TEAM_PACE_SCALING_2026.get(team, 1.0)
                pace_ratio = best_finish / max(best_pred, 1)
                new_scale  = update_pace_scaling(team, pace_ratio, 1.0)
                if abs(new_scale - old_scale) > 0.001:
                    insights["pace_updates"].append({
                        "type":      "team",
                        "team":      team,
                        "old_scale": old_scale,
                        "new_scale": new_scale,
                        "direction": "faster" if new_scale < old_scale else "slower",
                        "note": (
                            f"Team recalibrated: {team} running "
                            f"{'faster' if new_scale < old_scale else 'slower'} "
                            f"than prior suggested "
                            f"({old_scale:.4f} → {new_scale:.4f})."
                        ),
                    })

    # Per-driver delta updates
    for res in actual_results:
        driver   = res["driver"]
        team     = res["team"]
        is_dnf   = res.get("dnf", False)
        if is_dnf or driver not in DRIVER_PACE_DELTA_2026:
            continue

        actual_pos = res["finish_pos"]
        if not pre_race_predictions or driver not in pre_race_predictions:
            continue

        pred_pos = _get_pred_pos(pre_race_predictions[driver])
        if not pred_pos or pred_pos <= 0:
            continue

        # Position ratio as pace proxy — finishing higher than predicted
        # implies faster than the model expected, so observed_delta < 1.0
        old_delta  = DRIVER_PACE_DELTA_2026.get(driver, 1.0)
        new_delta  = update_driver_delta(driver, float(actual_pos), float(pred_pos))
        confidence = DRIVER_DELTA_CONFIDENCE.get(driver, 0.0)
        n_obs      = DRIVER_DELTA_OBSERVATIONS.get(driver, 0)

        if abs(new_delta - old_delta) > 0.00015:
            insights["pace_updates"].append({
                "type":         "driver",
                "driver":       driver,
                "team":         team,
                "old_delta":    old_delta,
                "new_delta":    new_delta,
                "confidence":   confidence,
                "observations": n_obs,
                "direction":    "faster" if new_delta < old_delta else "slower",
                "note": (
                    f"Driver delta updated: {driver} ({team}) "
                    f"{'faster' if new_delta < old_delta else 'slower'} "
                    f"than team baseline "
                    f"({old_delta:.5f} → {new_delta:.5f}, "
                    f"confidence {confidence:.0%} after "
                    f"{n_obs} race{'s' if n_obs != 1 else ''})."
                ),
            })



    # ---- 5. Updated championship standings ---- #
    wdc = _state.get_wdc_standings()
    wcc = _state.get_wcc_standings()

    insights["championship"] = {
        "wdc_top5": wdc[:5],
        "wcc_top5": wcc[:5],
        "races_completed": round_num,
        "races_remaining": 24 - round_num,
    }

    # ---- 6. Season outlook ---- #
    insights["season_outlook"] = _generate_season_outlook(wdc, wcc, round_num)

    return insights


def _generate_season_outlook(
    wdc: list, wcc: list, races_done: int
) -> dict:
    """
    Projects championship probabilities based on current standings and
    remaining races.

    THEORY — Points projection:
        Maximum remaining points = (24 - races_done) × 26
        (25 for win + 1 fastest lap, every remaining race)

        A driver is mathematically eliminated when:
            their_points + max_remaining < leader_points

        We also compute a simple probabilistic projection based on
        current points-per-race average. This isn't a full Monte Carlo
        simulation (that comes in the season loop) but gives a quick
        "who is likely to win" read after each real race result.
    """
    if not wdc or races_done == 0:
        return {"note": "No races completed yet."}

    max_remaining  = (24 - races_done) * 26
    leader, leader_pts = wdc[0]

    outlook = {
        "max_remaining_points": max_remaining,
        "leader": leader,
        "leader_points": leader_pts,
        "contenders": [],
        "eliminated": [],
    }

    for driver, pts in wdc:
        gap = leader_pts - pts
        if driver == leader:
            ppr = pts / max(races_done, 1)
            proj = round(pts + ppr * (24 - races_done))
            outlook["contenders"].append({
                "driver": driver,
                "points": pts,
                "gap_to_leader": 0,
                "projected_final": proj,
                "status": "🏆 Leader",
            })
        elif pts + max_remaining < leader_pts:
            outlook["eliminated"].append({
                "driver": driver,
                "points": pts,
                "note": "Mathematically eliminated from title",
            })
        else:
            ppr  = pts / max(races_done, 1)
            proj = round(pts + ppr * (24 - races_done))
            outlook["contenders"].append({
                "driver": driver,
                "points": pts,
                "gap_to_leader": gap,
                "projected_final": proj,
                "status": "🔥 Title contender" if gap <= max_remaining * 0.4 else "In the fight",
            })

    return outlook


def format_insights_report(insights: dict) -> str:
    """
    Formats the insights dict into a readable CLI/terminal report.
    The same dict is also returned via the API for frontend display.
    """
    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"  🏁  RACE INSIGHTS: {insights['race'].upper()}")
    lines.append(f"  Round {insights['round']}/24 — Generated {insights['generated_at'][:10]}")
    lines.append(f"{'='*65}")

    # Driver highlights
    lines.append("\n📊 DRIVER PERFORMANCE vs MODEL PREDICTIONS")
    lines.append("-" * 65)
    for d in sorted(insights["driver_insights"], key=lambda x: x["finish_position"]):
        pos_str = f"P{d['finish_position']:<3}" if not d["dnf"] else "DNF "
        lines.append(f"  {pos_str} {d['driver']:<5} ({d['team']:<22})  {d['narrative']}")

    # Pace updates
    if insights["pace_updates"]:
        lines.append("\n🔧 MODEL RECALIBRATION (Pace Scaling Updates)")
        lines.append("-" * 65)
        team_updates   = [u for u in insights["pace_updates"] if u.get("type") == "team"]
        driver_updates = [u for u in insights["pace_updates"] if u.get("type") == "driver"]
        if team_updates:
            lines.append("  Car pace (team scaling):")
            for u in team_updates:
                lines.append(f"    {u['note']}")
        if driver_updates:
            lines.append("  Driver deltas (intra-team skill):")
            for u in driver_updates:
                lines.append(f"    {u['note']}")

    # Championship
    champ = insights["championship"]
    lines.append(f"\n🏆 CHAMPIONSHIP AFTER ROUND {champ['races_completed']}")
    lines.append("-" * 65)
    lines.append("  WDC Top 5:")
    for i, (driver, pts) in enumerate(champ["wdc_top5"], 1):
        lines.append(f"    P{i}. {driver:<6} {pts} pts")
    lines.append("  WCC Top 5:")
    for i, (team, pts) in enumerate(champ["wcc_top5"], 1):
        lines.append(f"    P{i}. {team:<25} {pts} pts")

    # Season outlook
    outlook = insights["season_outlook"]
    if "contenders" in outlook:
        lines.append(f"\n📈 SEASON OUTLOOK ({outlook['races_remaining']} races remaining)")
        lines.append("-" * 65)
        lines.append(f"  Max remaining points: {outlook['max_remaining_points']}")
        for c in outlook["contenders"][:3]:
            lines.append(
                f"  {c['status']} {c['driver']}: {c['points']} pts "
                f"(gap: {c['gap_to_leader']}) → projected {c['projected_final']} pts"
            )
        if outlook["eliminated"]:
            elim_names = [e["driver"] for e in outlook["eliminated"]]
            lines.append(f"  ❌ Mathematically eliminated: {', '.join(elim_names)}")

    lines.append(f"\n{'='*65}\n")
    return "\n".join(lines)