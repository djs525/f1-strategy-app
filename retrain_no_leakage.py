"""
retrain_no_leakage.py
=====================
Retrains the F1 lap-time predictor with three fixes applied:

FIX 1 — Leakage removal
    Dropped from num_cols: best_position, worst_position,
    avg_position, avg_position_vs_grid.
    These are post-race outcomes, not pre-race inputs.

FIX 2 — Outlier filtering
    Rows where total_laps_completed < 10 are dropped before training.
    The 2021 Belgian GP (3 laps behind safety car) and similar
    red-flag/DNF races produce 200s+ average lap times that have
    nothing to do with strategy and blow up the loss function.

FIX 3 — Circuit-relative target (delta from circuit mean)
    Instead of predicting raw lap time (range: 69s–135s),
    the model predicts delta = avg_lap_time - circuit_mean_lap_time.
    This collapses the target range from ~66s down to roughly ±5s,
    making the task tractable without leaky features.

    At inference time, the circuit mean is added back:
        predicted_lap_time = model_output + circuit_mean

    The circuit_means dict is stored in the preprocessors joblib
    so model_adapter.py and main.py can perform the reconstruction.

    WHY THIS WORKS BETTER:
        The circuit embedding already captures "this is a fast/slow
        circuit". Before this fix, 95% of the model's job was just
        learning that Spa is slower than Austria. Now it focuses
        entirely on the 3s of within-circuit variance that actually
        depends on driver, team, strategy, and conditions.

HOW TO RUN
----------
    cd /path/to/f1-strategy-app
    python retrain_no_leakage.py

Outputs (overwrites existing files):
    data/trained_models/f1_preprocessors.joblib
    data/trained_models/best_lap_time_predictor.pth
"""

import os
import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ─── PATHS ────────────────────────────────────────────────────────────────────
_HERE     = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_HERE, "data/feature_data/features_dataset_with_targets.csv")
OUT_PREP  = os.path.join(_HERE, "data/trained_models/f1_preprocessors.joblib")
OUT_MODEL = os.path.join(_HERE, "data/trained_models/best_lap_time_predictor.pth")

# ─── HYPERPARAMETERS ──────────────────────────────────────────────────────────
EPOCHS       = 150
BATCH_SIZE   = 256
LR           = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE     = 20
DEVICE       = torch.device("cpu")

# ─── FEATURES ─────────────────────────────────────────────────────────────────
LEAKAGE_COLS = [
    "best_position", "worst_position",
    "avg_position",  "avg_position_vs_grid",
]

SAFE_NUM_COLS = [
    "grid_position",
    "num_pit_stops",
    "avg_pit_duration",
    "total_laps_completed",
    "first_pit_lap_pct",
    "second_pit_lap_pct",
    "third_pit_lap_pct",
    "laps_on_soft",
    "laps_on_medium",
    "laps_on_hard",
    "laps_on_intermediate",
    "laps_on_wet",
    "pit_stops_per_lap",
    "avg_tyre_age",
    "max_stint_laps",
    "deg_score",
    # FIX 3: fuel load proxy features.
    # The model has no direct fuel signal. These two features give it a
    # proxy for how much heavy-fuel running occurred in the first stint.
    #
    # stint1_fuel_weight: fraction of the race spent in the opening stint.
    #   A high value (e.g. 0.7 for a 1-stop) means the driver spent most
    #   of the race on a heavy, degrading fuel load. A low value (e.g. 0.25
    #   for a 3-stop) means short stints with fresh tyres AND lighter fuel.
    #   This lets the model learn that 3-stop strategies aren't always faster
    #   on average because the early laps carry ~40kg more fuel.
    #
    # num_pit_stops_norm: pit stops scaled 0-1 over max 3 stops.
    #   Redundant with num_pit_stops but on a normalised scale that avoids
    #   the model conflating "1 stop" and "3 stops" as linearly equivalent
    #   to their fuel/tyre interaction effects.
    "stint1_fuel_weight",
    "num_pit_stops_norm",
]

RAW_TARGET = "avg_lap_time_circuit"
DELTA_COL  = "_lap_time_delta"        # computed — delta from circuit mean


# ─── MODEL ARCHITECTURE (unchanged from original) ─────────────────────────────
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


# ─── MAIN ─────────────────────────────────────────────────────────────────────
def train():
    print("=" * 62)
    print("  F1 Lap Time Predictor — Leakage-Free + Delta Retraining")
    print("=" * 62)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    print(f"\n[1/7] Loading data from {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    print(f"      {len(df)} rows loaded")

    # ── 2. Filter outliers ────────────────────────────────────────────────────
    print(f"\n[2/7] Filtering outliers")
    before = len(df)
    df = df.dropna(subset=[RAW_TARGET])
    df = df[df["total_laps_completed"] >= 10].copy()
    removed = before - len(df)
    print(f"      Removed {removed} rows (DNF/SC races with < 10 laps completed)")
    print(f"      {len(df)} rows remaining")
    print(f"      Target range after filtering: "
          f"{df[RAW_TARGET].min():.2f}s – {df[RAW_TARGET].max():.2f}s")

    # ── 3. Compute circuit means + delta target ───────────────────────────────
    print(f"\n[3/7] Computing circuit-relative target")
    circuit_means = df.groupby("gp_name")[RAW_TARGET].mean().to_dict()
    df[DELTA_COL] = df[RAW_TARGET] - df["gp_name"].map(circuit_means)

    raw_std   = df[RAW_TARGET].std()
    delta_std = df[DELTA_COL].std()
    print(f"      Raw target std   : {raw_std:.3f}s")
    print(f"      Delta target std : {delta_std:.3f}s  "
          f"({raw_std/delta_std:.1f}x easier for the model)")
    print(f"      Delta range      : {df[DELTA_COL].min():.3f}s – "
          f"{df[DELTA_COL].max():.3f}s")

    # FIX 3: Derive fuel load proxy features.
    # stint1_fuel_weight = first_pit_lap_pct (fraction of race in opening stint).
    # A high value means the driver spent most of the race on heavy fuel.
    # For 0-stop races the whole race is the first stint so weight = 1.0.
    # num_pit_stops_norm normalises stops to 0–1 range over max 3 stops.
    print(f"\n      [Fix 3] Computing fuel load proxy features")
    df["stint1_fuel_weight"] = df["first_pit_lap_pct"].clip(0.0, 1.0)
    df.loc[df["num_pit_stops"] == 0, "stint1_fuel_weight"] = 1.0
    df["num_pit_stops_norm"] = (df["num_pit_stops"] / 3.0).clip(0.0, 1.0)
    print(f"      stint1_fuel_weight: "
          f"{df['stint1_fuel_weight'].min():.3f} – {df['stint1_fuel_weight'].max():.3f}")
    print(f"      num_pit_stops_norm: "
          f"{df['num_pit_stops_norm'].min():.3f} – {df['num_pit_stops_norm'].max():.3f}")

    # ── 4. Encode categoricals ────────────────────────────────────────────────
    print(f"\n[4/7] Fitting label encoders")
    le_gp      = LabelEncoder().fit(df["gp_name"])
    le_driver  = LabelEncoder().fit(df["driver_code"])
    le_team    = LabelEncoder().fit(df["team_name"])
    le_year    = LabelEncoder().fit(df["Year"])
    le_weather = LabelEncoder().fit(df["weather_code"])

    gp_idx      = le_gp.transform(df["gp_name"])
    driver_idx  = le_driver.transform(df["driver_code"])
    team_idx    = le_team.transform(df["team_name"])
    year_idx    = le_year.transform(df["Year"])
    weather_idx = le_weather.transform(df["weather_code"])

    print(f"      GPs: {len(le_gp.classes_)}, Drivers: {len(le_driver.classes_)}, "
          f"Teams: {len(le_team.classes_)}, Years: {len(le_year.classes_)}, "
          f"Weather: {len(le_weather.classes_)}")

    # ── 5. Build + scale numerical features ───────────────────────────────────
    print(f"\n[5/7] Building numerical features "
          f"({len(SAFE_NUM_COLS)} cols — leakage removed)")
    num_df = df[SAFE_NUM_COLS].copy()
    for col in SAFE_NUM_COLS:
        if num_df[col].isna().any():
            fill = num_df[col].mean()
            num_df[col] = num_df[col].fillna(fill)
            print(f"      NaN-filled: {col} → {fill:.3f}")

    scaler       = StandardScaler()
    num_scaled   = scaler.fit_transform(num_df)
    avg_pit_mean = float(df["avg_pit_duration"].mean())

    # ── 6. Train / val split ──────────────────────────────────────────────────
    y       = df[DELTA_COL].values.astype(np.float32)
    indices = np.arange(len(df))
    train_idx, val_idx = train_test_split(indices, test_size=0.15, random_state=42)

    # Keep gp_name for each val row so we can reconstruct absolute times
    val_gp_names = df["gp_name"].values[val_idx]

    def make_tensors(idx):
        return (
            torch.tensor(gp_idx[idx],      dtype=torch.long),
            torch.tensor(driver_idx[idx],  dtype=torch.long),
            torch.tensor(team_idx[idx],    dtype=torch.long),
            torch.tensor(year_idx[idx],    dtype=torch.long),
            torch.tensor(weather_idx[idx], dtype=torch.long),
            torch.tensor(num_scaled[idx],  dtype=torch.float32),
            torch.tensor(y[idx],           dtype=torch.float32).unsqueeze(1),
        )

    train_dl = DataLoader(TensorDataset(*make_tensors(train_idx)),
                          batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(TensorDataset(*make_tensors(val_idx)),
                          batch_size=BATCH_SIZE, shuffle=False)

    print(f"      Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")

    # ── 7. Train ──────────────────────────────────────────────────────────────
    print(f"\n[6/7] Training")
    model = F1LapTimePredictor(
        num_gps                = len(le_gp.classes_),
        num_drivers            = len(le_driver.classes_),
        num_teams              = len(le_team.classes_),
        num_numerical_features = len(SAFE_NUM_COLS),
        num_years              = len(le_year.classes_),
        num_weather_conditions = len(le_weather.classes_),
    ).to(DEVICE)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        for batch in train_dl:
            gp_b, dr_b, tm_b, yr_b, wx_b, num_b, tgt_b = [t.to(DEVICE) for t in batch]
            optimizer.zero_grad()
            pred = model(gp_b, dr_b, tm_b, yr_b, wx_b, num_b)
            loss = criterion(pred, tgt_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * len(gp_b)
        train_loss /= len(train_idx)

        model.eval()
        val_loss = val_mae = 0.0
        with torch.no_grad():
            for batch in val_dl:
                gp_b, dr_b, tm_b, yr_b, wx_b, num_b, tgt_b = [t.to(DEVICE) for t in batch]
                pred = model(gp_b, dr_b, tm_b, yr_b, wx_b, num_b)
                val_loss += criterion(pred, tgt_b).item() * len(gp_b)
                val_mae  += (pred - tgt_b).abs().sum().item()
        val_loss /= len(val_idx)
        val_mae  /= len(val_idx)

        scheduler.step()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state    = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"      Epoch {epoch:3d}/{EPOCHS} | "
                  f"train={train_loss:.4f} | val={val_loss:.4f} | "
                  f"val_MAE(delta)={val_mae:.3f}s | "
                  f"lr={scheduler.get_last_lr()[0]:.2e}")

        if no_improve >= PATIENCE:
            print(f"      Early stopping at epoch {epoch} "
                  f"(no improvement for {PATIENCE} epochs)")
            break

    model.load_state_dict(best_state)
    model.eval()

    # Final metrics — report both delta MAE and reconstructed absolute MAE
    all_deltas_pred, all_deltas_true = [], []
    with torch.no_grad():
        for batch in val_dl:
            gp_b, dr_b, tm_b, yr_b, wx_b, num_b, tgt_b = [t.to(DEVICE) for t in batch]
            pred = model(gp_b, dr_b, tm_b, yr_b, wx_b, num_b)
            all_deltas_pred.extend(pred.squeeze().tolist())
            all_deltas_true.extend(tgt_b.squeeze().tolist())

    delta_mae = np.mean(np.abs(
        np.array(all_deltas_pred) - np.array(all_deltas_true)
    ))

    # Reconstruct absolute times to show real-world accuracy
    circuit_means_arr = np.array([
        circuit_means.get(gp, df[RAW_TARGET].mean()) for gp in val_gp_names
    ])
    abs_mae = np.mean(np.abs(
        (np.array(all_deltas_pred) + circuit_means_arr) -
        (np.array(all_deltas_true) + circuit_means_arr)
    ))  # abs_mae == delta_mae by definition — shown separately for clarity

    print(f"\n      ✓ Val MAE (delta)    : {delta_mae:.4f}s "
          f"← within-circuit accuracy")
    print(f"      ✓ Val MAE (absolute) : {abs_mae:.4f}s "
          f"← same number, circuit mean added back")

    # ── 8. Save ───────────────────────────────────────────────────────────────
    print(f"\n[7/7] Saving artifacts")
    os.makedirs(os.path.dirname(OUT_PREP), exist_ok=True)

    preprocessors = {
        # Label encoders
        "le_gp":            le_gp,
        "le_driver":        le_driver,
        "le_team":          le_team,
        "le_year":          le_year,
        "le_weather":       le_weather,
        # Scaler
        "scaler":           scaler,
        # Feature lists
        "num_cols":         SAFE_NUM_COLS,
        "compound_cols":    [c for c in SAFE_NUM_COLS if c.startswith("laps_on_")],
        "pit_timing_cols":  ["first_pit_lap_pct", "second_pit_lap_pct",
                             "third_pit_lap_pct"],
        "degradation_cols": ["avg_tyre_age", "max_stint_laps", "deg_score"],
        # Delta reconstruction — inference code MUST add circuit mean back
        "circuit_means":    circuit_means,
        "global_mean":      float(df[RAW_TARGET].mean()),
        "target_is_delta":  True,
        # Misc
        "avg_pit_mean":     avg_pit_mean,
        "pit_laps_mean":    float(df[["first_pit_lap", "second_pit_lap",
                                       "third_pit_lap"]].stack().mean()),
        "leakage_removed":  LEAKAGE_COLS,
    }

    joblib.dump(preprocessors, OUT_PREP)
    torch.save(model.state_dict(), OUT_MODEL)

    print(f"      ✓ Preprocessors → {OUT_PREP}")
    print(f"      ✓ Model weights  → {OUT_MODEL}")
    print(f"\n{'=' * 62}")
    print(f"  Done.  MAE = {delta_mae:.3f}s within-circuit")
    print(f"  Restart uvicorn to pick up the new artifacts.")
    print(f"{'=' * 62}\n")


if __name__ == "__main__":
    train()