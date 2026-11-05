"""Étape 6 : un LSTM PyTorch qui prévoit les 24 heures du lendemain, comparé à LightGBM."""
import copy
import time

import mlflow
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from eval_lib import bootstrap_wape, bootstrap_wape_diff, load_dataset, metrics

torch.manual_seed(42)
np.random.seed(42)
HIST_DAYS, BATCH, MAX_EPOCHS, PATIENCE, LR = 7, 256, 40, 5, 1e-3

# ---------- 1. Les données sous forme de tableaux [zone, heure] ----------
df = load_dataset().sort_values(["pu_zone", "hour"])
zones = np.sort(df["pu_zone"].unique())
hours = np.sort(df["hour"].unique())
Z, H = len(zones), len(hours)
D = H // 24
demand = df["n_pickups"].to_numpy(float).reshape(Z, H)          # possible car la grille est complète
baseline = df["mean_same_slot_4w"].to_numpy(float).reshape(Z, H)

# Une ligne par jour (heure 0 de la première zone) pour le calendrier et la période
first = df[df["pu_zone"] == zones[0]].iloc[::24]
dow = first["day_of_week"].to_numpy(int) - 1                     # 0 = lundi
month = first["month"].to_numpy(float)
day_feat = np.concatenate([np.eye(7)[dow],                       # jour de la semaine (one-hot)
                           first[["is_holiday"]].to_numpy(float),
                           np.sin(2 * np.pi * month / 12)[:, None],
                           np.cos(2 * np.pi * month / 12)[:, None]], axis=1)   # [D, 10]
day_split = first["split"].to_numpy()
days = {s: np.where((day_split == s) & (np.arange(D) >= HIST_DAYS))[0] for s in ("train", "val")}

# Normalisation par zone, calculée sur l'ENTRAÎNEMENT seulement
train_pos = (days["train"][:, None] * 24 + np.arange(24)).ravel()
scale = demand[:, train_pos].mean(axis=1)
y_mean = demand[:, train_pos].mean()

hod = np.arange(HIST_DAYS * 24) % 24                             # heure de la journée de chaque pas
hour_enc = np.stack([np.sin(2 * np.pi * hod / 24), np.cos(2 * np.pi * hod / 24)], axis=1)


def build(day_idx):
    """Un exemple par (zone, jour) : 168 heures d'historique -> 24 heures à prévoir."""
    z = np.repeat(np.arange(Z), len(day_idx))
    d = np.tile(day_idx, Z)
    hist_pos = (d[:, None] - HIST_DAYS) * 24 + np.arange(HIST_DAYS * 24)   # [N, 168]
    tgt_pos = d[:, None] * 24 + np.arange(24)                              # [N, 24]
    s = scale[z][:, None]
    hist = demand[z[:, None], hist_pos] / s
    x_hist = np.concatenate([hist[..., None],
                             np.broadcast_to(hour_enc, (len(z), HIST_DAYS * 24, 2))], axis=2)
    base = baseline[z[:, None], tgt_pos] / s
    y = demand[z[:, None], tgt_pos]
    f = lambda a: torch.tensor(np.ascontiguousarray(a), dtype=torch.float32)
    ds = TensorDataset(f(x_hist), f(base), f(day_feat[d]), torch.tensor(z), f(y), f(scale[z]))
    return ds, z, d


train_ds, _, _ = build(days["train"])
val_ds, val_z, val_d = build(days["val"])
print(f"Exemples : {len(train_ds):,} en entraînement, {len(val_ds):,} en validation")


# ---------- 2. Le modèle ----------
class DemandLSTM(nn.Module):
    def __init__(self, n_zones, hidden=64, zone_dim=8, day_dim=10):
        super().__init__()
        self.zone_emb = nn.Embedding(n_zones, zone_dim)
        self.lstm = nn.LSTM(input_size=3, hidden_size=hidden, num_layers=2,
                            batch_first=True, dropout=0.1)
        self.head = nn.Sequential(
            nn.Linear(hidden + zone_dim + day_dim + 24, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, 24))

    def forward(self, x_hist, base, day, zone):
        _, (h, _) = self.lstm(x_hist)              # h[-1] : le résumé des 168 heures lues
        ctx = torch.cat([h[-1], self.zone_emb(zone), day, base], dim=1)
        return base + self.head(ctx)               # la référence, plus une correction apprise


model = DemandLSTM(Z)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
n_params = sum(p.numel() for p in model.parameters())
print(f"Paramètres du modèle : {n_params:,}")


def loss_fn(out, y, s):
    """Erreur absolue sur la demande réelle (alignée sur la WAPE), divisée par la demande moyenne."""
    return (out * s[:, None] - y).abs().mean() / y_mean


def predict(ds):
    model.eval()                                   # désactive le dropout
    preds = []
    with torch.no_grad():                          # pas de gradients en prévision
        for xh, b, dy, z, _, s in DataLoader(ds, batch_size=1024):
            preds.append((model(xh, b, dy, z) * s[:, None]).clamp(min=0))
    model.train()
    return torch.cat(preds).numpy()


# ---------- 3. Entraînement, avec arrêt anticipé et suivi MLflow ----------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("pytorch")
val_y = val_ds.tensors[4].numpy()
best_wape, best_state, best_epoch, waited = float("inf"), None, 0, 0

with mlflow.start_run(run_name="lstm_7j_residuel"):
    mlflow.log_params({"hist_days": HIST_DAYS, "batch": BATCH, "lr": LR, "hidden": 64,
                       "num_layers": 2, "zone_dim": 8, "dropout": 0.1, "n_params": n_params})
    loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    for epoch in range(1, MAX_EPOCHS + 1):
        t0, total = time.time(), 0.0
        for xh, b, dy, z, y, s in loader:
            loss = loss_fn(model(xh, b, dy, z), y, s)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(y)
        train_loss = total / len(train_ds)
        val_wape = metrics(val_y.ravel(), predict(val_ds).ravel())["WAPE"]
        mlflow.log_metrics({"train_loss": train_loss, "val_WAPE_epoch": val_wape}, step=epoch)
        print(f"Époque {epoch:2d} : perte {train_loss:.4f}, WAPE validation {val_wape:.4f}, "
              f"{time.time() - t0:.0f} s")
        if val_wape < best_wape:
            best_wape, best_state, best_epoch, waited = val_wape, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            waited += 1
            if waited >= PATIENCE:
                print(f"Arrêt anticipé : pas de progrès depuis {PATIENCE} époques")
                break

    model.load_state_dict(best_state)
    torch.save(best_state, "models/lstm.pt")
    mlflow.log_artifact("models/lstm.pt")

    # ---------- 4. Comparaison avec la référence et LightGBM ----------
    pred = predict(val_ds)
    res = pd.DataFrame({
        "pu_zone": np.repeat(zones[val_z], 24).astype(int),
        "hour": hours[(val_d[:, None] * 24 + np.arange(24)).ravel()],
        "pred_lstm": pred.ravel(),
    })
    lgbm = pd.read_parquet("data/processed/val_predictions_lgbm.parquet").rename(columns={"pred": "pred_lgbm"})
    lgbm["pu_zone"] = lgbm["pu_zone"].astype(int)
    for frame in (res, lgbm):
        frame["hour"] = pd.to_datetime(frame["hour"]).astype("datetime64[ns]")
    val = lgbm.merge(res, on=["pu_zone", "hour"], how="inner")
    assert len(val) == len(res) == len(lgbm), "les prévisions ne correspondent pas aux mêmes lignes"
    val["pred_moyenne_lgbm_lstm"] = (val["pred_lgbm"] + val["pred_lstm"]) / 2

    rows = []
    for name, col in [("référence : moyenne 4 semaines", "mean_same_slot_4w"),
                      ("LightGBM (Poisson, sans météo)", "pred_lgbm"),
                      ("LSTM", "pred_lstm"),
                      ("moyenne LightGBM + LSTM", "pred_moyenne_lgbm_lstm")]:
        m = metrics(val["n_pickups"], val[col])
        lo, hi = bootstrap_wape(val, col)
        rows.append({"modèle": name, **{k: round(v, 3) for k, v in m.items()},
                     "WAPE IC 95 %": f"[{lo:.3f} ; {hi:.3f}]"})
    table = pd.DataFrame(rows)
    print(f"\nMeilleure époque : {best_epoch}")
    print("\n=== Validation (juillet à septembre 2025) ===")
    print(table.to_string(index=False))
    table.to_csv("reports/lstm_val.csv", index=False)

    for a, b in [("pred_lstm", "mean_same_slot_4w"), ("pred_lstm", "pred_lgbm"),
                 ("pred_moyenne_lgbm_lstm", "pred_lgbm")]:
        lo, hi = bootstrap_wape_diff(val, a, b)
        print(f"WAPE({a}) - WAPE({b}) : IC 95 % [{lo:+.3f} ; {hi:+.3f}]")
        mlflow.log_metrics({f"diff_{a}_vs_{b}_low": lo, f"diff_{a}_vs_{b}_high": hi})

    m = metrics(val["n_pickups"], val["pred_lstm"])
    mlflow.log_metrics({f"val_{k}": v for k, v in m.items()} | {"best_epoch": best_epoch})
    val.to_parquet("data/processed/val_predictions_all.parquet", index=False)