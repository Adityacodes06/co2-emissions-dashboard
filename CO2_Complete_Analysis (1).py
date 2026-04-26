import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor,
    ExtraTreesRegressor,
    StackingRegressor
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: Load Dataset & Exploratory Data Analysis

df = pd.read_csv("data/CO2 Emissions_Canada.csv")

print("=" * 55)
print("  STEP 1: DATASET OVERVIEW")
print("=" * 55)
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nColumns:", list(df.columns))

print("\nDataset Info:")
df.info()

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# ============================================================
# STEP 2: Physics-Informed Feature Engineering
# ============================================================
print("\n" + "=" * 55)
print("  STEP 2: PHYSICS-INFORMED FEATURE ENGINEERING")
print("=" * 55)

# 1. Displacement per Cylinder
#    Larger displacement/cylinder → more fuel burned per stroke → higher CO2
df["Displacement_per_Cylinder"] = df["Engine Size(L)"] / df["Cylinders"]

# 2. Thermal Load Index (upgraded from basic Knocking Index)
#    Approximates heat output from combustion
#    Engine Size × Cylinders × Combined Fuel Consumption / 100
df["Thermal_Load_Index"] = (
    df["Engine Size(L)"] *
    df["Cylinders"] *
    df["Fuel Consumption Comb (L/100 km)"]
) / 100

# 3. City-Highway Consumption Spread
#    High spread = engine is inefficient at highway speeds (aerodynamic drag)
df["City_Hwy_Spread"] = (
    df["Fuel Consumption City (L/100 km)"] -
    df["Fuel Consumption Hwy (L/100 km)"]
)

# 4. City vs Combined Ratio (stop-start inefficiency indicator)
#    Ratio close to 1 → consistent fuel use across cycles
df["City_Comb_Ratio"] = (
    df["Fuel Consumption City (L/100 km)"] /
    df["Fuel Consumption Comb (L/100 km)"]
)

# 5. Volumetric Efficiency Proxy
#    Displacement per cylinder normalized by combined fuel consumption
#    Higher = more efficient combustion per unit displacement
df["Volumetric_Efficiency_Proxy"] = (
    df["Displacement_per_Cylinder"] /
    df["Fuel Consumption Comb (L/100 km)"]
)

# 6. Harmonic Fuel Mean
#    Harmonic mean of city/highway consumption (more physically meaningful
#    than arithmetic mean — penalizes extreme values correctly)
df["Harmonic_Fuel_Mean"] = (
    2 / (
        1 / df["Fuel Consumption City (L/100 km)"] +
        1 / df["Fuel Consumption Hwy (L/100 km)"]
    )
)

# 7. Stoichiometric CO2 Estimate
#    Each fuel type emits a different amount of CO2 per litre burned:
#    X (regular gasoline) ~2289 g/L
#    Z (premium gasoline) ~2289 g/L
#    D (diesel)           ~2640 g/L  ← heavier carbon content
#    E (ethanol blend)    ~1594 g/L  ← lower carbon content
fuel_co2_factor = {"X": 2289, "Z": 2289, "D": 2640, "E": 1594, "N": 1900}
df["Stoich_CO2_Estimate"] = (
    df["Fuel Consumption Comb (L/100 km)"] *
    df["Fuel Type"].map(fuel_co2_factor).fillna(2289) / 100
)

physics_cols = [
    "Displacement_per_Cylinder", "Thermal_Load_Index",
    "City_Hwy_Spread", "City_Comb_Ratio",
    "Volumetric_Efficiency_Proxy", "Harmonic_Fuel_Mean",
    "Stoich_CO2_Estimate"
]

print("\nPhysics Features Summary:")
print(df[physics_cols].describe().round(3))

print("\nCorrelation of Physics Features with CO2 Emissions:")
for col in physics_cols:
    corr = df[col].corr(df["CO2 Emissions(g/km)"])
    print(f"  {col:<35} {corr:+.4f}")

# ============================================================
# STEP 3: Feature Selection & Smart Encoding
# ============================================================
print("\n" + "=" * 55)
print("  STEP 3: FEATURE SELECTION & ENCODING")
print("=" * 55)

X = df.drop("CO2 Emissions(g/km)", axis=1)
y = df["CO2 Emissions(g/km)"]

# Target Encoding for high-cardinality columns (Make, Model)
# Maps each category to the mean CO2 of that group
# Much better than one-hot for high-cardinality — avoids sparse dimensions
high_card_cols = ["Make", "Model"]
for col in high_card_cols:
    means = y.groupby(df[col]).mean()
    X[col + "_TargetEnc"] = df[col].map(means)
X = X.drop(columns=high_card_cols)

# One-Hot Encoding for low-cardinality columns
low_card_cols = ["Vehicle Class", "Transmission", "Fuel Type"]
X_encoded = pd.get_dummies(X, columns=low_card_cols, drop_first=True)

print(f"\nShape after encoding : {X_encoded.shape}")
print(f"Target encoded cols  : {high_card_cols}")
print(f"One-hot encoded cols : {low_card_cols}")

# ============================================================
# STEP 4: Train-Test Split
# ============================================================
print("\n" + "=" * 55)
print("  STEP 4: TRAIN-TEST SPLIT")
print("=" * 55)

X_train, X_test, y_train, y_test = train_test_split(
    X_encoded, y, test_size=0.2, random_state=42
)
print(f"Training set : {X_train.shape}")
print(f"Testing set  : {X_test.shape}")

# ============================================================
# STEP 5: Model Training & Evaluation Helper
# ============================================================
results = {}

def evaluate_model(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae  = mean_absolute_error(y_te, preds)
    rmse = np.sqrt(mean_squared_error(y_te, preds))
    r2   = r2_score(y_te, preds)
    mape = np.mean(np.abs((y_te - preds) / y_te)) * 100
    rae  = np.sum(np.abs(y_te - preds)) / np.sum(np.abs(y_te - np.mean(y_te)))
    results[name] = {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape, "RAE": rae, "preds": preds}
    print(f"\n  {'='*40}")
    print(f"  {name}")
    print(f"  {'='*40}")
    print(f"  MAE  : {mae:.4f} g/km")
    print(f"  RMSE : {rmse:.4f} g/km")
    print(f"  R²   : {r2:.6f}")
    print(f"  MAPE : {mape:.4f} %")
    print(f"  RAE  : {rae:.6f}")
    return model, preds

# ============================================================
# STEP 6: Model 1 — Linear Regression (Original Baseline)
# ============================================================
print("\n" + "=" * 55)
print("  STEP 5-7: MODEL TRAINING")
print("=" * 55)

lr_model, lr_preds = evaluate_model(
    "Linear Regression (Baseline)",
    LinearRegression(),
    X_train, y_train, X_test, y_test
)

# ============================================================
# STEP 7: Model 2 — Random Forest (Upgraded Baseline)
# ============================================================
rf_model, rf_preds = evaluate_model(
    "Random Forest",
    RandomForestRegressor(n_estimators=200, max_depth=12,
                          min_samples_leaf=2, random_state=42, n_jobs=-1),
    X_train, y_train, X_test, y_test
)

# ============================================================
# STEP 8: Model 3 — Stacking Ensemble
# ============================================================
# Architecture:
#   Base Learner 1: Random Forest     — captures non-linear feature interactions
#   Base Learner 2: Gradient Boosting — sequential error correction
#   Base Learner 3: Extra Trees       — high variance, low bias (complementary to GB)
#   Meta-Learner  : ElasticNet        — learns optimal blend of base predictions
#                                       Ridge removes multicollinearity between base learners
#                                       Lasso allows automatic base learner selection

stacking_model = StackingRegressor(
    estimators=[
        ("random_forest",      RandomForestRegressor(n_estimators=200, max_depth=12,
                                                      min_samples_leaf=2, random_state=42, n_jobs=-1)),
        ("gradient_boosting",  GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                          max_depth=5, subsample=0.8, random_state=42)),
        ("extra_trees",        ExtraTreesRegressor(n_estimators=200, max_depth=12,
                                                    min_samples_leaf=2, random_state=42, n_jobs=-1))
    ],
    final_estimator=ElasticNet(alpha=0.01, l1_ratio=0.5),
    cv=5,           # 5-fold CV generates out-of-fold predictions for meta-learner
    passthrough=False,
    n_jobs=-1
)

stack_model, stack_preds = evaluate_model(
    "Stacking Ensemble (RF + GB + ET → ElasticNet)",
    stacking_model,
    X_train, y_train, X_test, y_test
)

# ============================================================
# STEP 9: Final Comparison Table
# ============================================================
print("\n" + "=" * 80)
print("  FINAL MODEL COMPARISON")
print("=" * 80)
print(f"  {'Model':<42} {'R²':>8}  {'MAE':>7}  {'RMSE':>7}  {'MAPE':>8}  {'RAE':>7}")
print("-" * 80)
for name, m in results.items():
    print(f"  {name:<42} {m['R2']:>8.5f}  {m['MAE']:>7.3f}  {m['RMSE']:>7.3f}  {m['MAPE']:>7.3f}%  {m['RAE']:>7.4f}")
print("=" * 80)

lr_r2    = results["Linear Regression (Baseline)"]["R2"]
rf_r2    = results["Random Forest"]["R2"]
stack_r2 = results["Stacking Ensemble (RF + GB + ET → ElasticNet)"]["R2"]
rf_mae   = results["Random Forest"]["MAE"]
stack_mae = results["Stacking Ensemble (RF + GB + ET → ElasticNet)"]["MAE"]

r2_gain  = (stack_r2 - rf_r2) / (1 - rf_r2) * 100
mae_gain = (rf_mae - stack_mae) / rf_mae * 100

print(f"\n  Stacking vs RF:")
print(f"  → R² gain        : {r2_gain:+.2f}% of remaining error closed")
print(f"  → MAE reduction  : {mae_gain:+.2f}%")
print(f"\n  Physics features : {len(physics_cols)}")
print(f"  Base learners    : 3 (RF, GradBoost, ExtraTrees)")
print(f"  Meta-learner     : ElasticNet")
print(f"  CV folds         : 5")
print("=" * 60)


# STEP 10: Visualizations & Comparison Graphs

import os
import matplotlib.gridspec as gridspec

os.makedirs("graphs", exist_ok=True)
print("\n[VIZ] Generating plots → saved to graphs/ folder")

# --- Figure 1: Full Analysis Dashboard ---
fig, axes = plt.subplots(2, 3, figsize=(18, 11))
fig.suptitle("CO2 Emissions Prediction — Complete Analysis Dashboard",
             fontsize=15, fontweight="bold", y=1.01)

# Plot 1: Model Comparison (R²)
ax = axes[0, 0]
names  = ["Linear\nRegression", "Random\nForest", "Stacking\nEnsemble"]
r2s    = [results[k]["R2"] for k in results]
colors = ["#95A5A6", "#5B8DB8", "#2ECC71"]
bars   = ax.bar(names, r2s, color=colors, edgecolor="white", linewidth=1.5)
ax.set_ylim(min(r2s) - 0.05, 1.0)
ax.set_ylabel("R² Score", fontsize=11)
ax.set_title("Model Comparison: R²", fontsize=12, fontweight="bold")
for bar, score in zip(bars, r2s):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
            f"{score:.5f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Plot 2: Model Comparison (MAE)
ax = axes[0, 1]
maes = [results[k]["MAE"] for k in results]
bars = ax.bar(names, maes, color=colors, edgecolor="white", linewidth=1.5)
ax.set_ylabel("MAE (g/km)", fontsize=11)
ax.set_title("Model Comparison: MAE", fontsize=12, fontweight="bold")
for bar, score in zip(bars, maes):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
            f"{score:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.grid(axis="y", alpha=0.3)

# Plot 3: Actual vs Predicted — Stacking
ax = axes[0, 2]
ax.scatter(y_test, stack_preds, alpha=0.3, color="#2ECC71", s=10, edgecolors="none")
lims = [y_test.min(), y_test.max()]
ax.plot(lims, lims, "r--", linewidth=2, label="Perfect")
ax.set_xlabel("Actual CO2 (g/km)", fontsize=10)
ax.set_ylabel("Predicted CO2 (g/km)", fontsize=10)
ax.set_title("Actual vs Predicted\n(Stacking Ensemble)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.text(0.05, 0.92, f"R² = {stack_r2:.5f}", transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.grid(alpha=0.3)

# Plot 4: Residuals — Stacking
ax = axes[1, 0]
residuals = y_test.values - stack_preds
ax.hist(residuals, bins=50, color="#2ECC71", edgecolor="white", linewidth=0.5, alpha=0.85)
ax.axvline(0, color="red", linestyle="--", linewidth=2)
ax.set_xlabel("Residual (Actual − Predicted)", fontsize=10)
ax.set_ylabel("Frequency", fontsize=10)
ax.set_title("Residuals Distribution\n(Stacking Ensemble)", fontsize=12, fontweight="bold")
ax.text(0.65, 0.90, f"Mean: {residuals.mean():.2f}\nStd: {residuals.std():.2f}",
        transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.grid(alpha=0.3)

# Plot 5: Physics Features Correlation
ax = axes[1, 1]
corr_vals = df[physics_cols + ["CO2 Emissions(g/km)"]].corr()["CO2 Emissions(g/km)"].drop("CO2 Emissions(g/km)").sort_values()
colors_c  = ["#E74C3C" if v > 0 else "#3498DB" for v in corr_vals.values]
ax.barh(corr_vals.index, corr_vals.values, color=colors_c, edgecolor="white")
ax.set_xlabel("Correlation with CO2 Emissions", fontsize=10)
ax.set_title("Physics Features vs CO2\n(Correlation)", fontsize=12, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8)
ax.grid(axis="x", alpha=0.3)

# Plot 6: Actual vs Predicted — Linear Regression (original baseline for comparison)
ax = axes[1, 2]
ax.scatter(y_test, lr_preds, alpha=0.3, color="#95A5A6", s=10, edgecolors="none")
ax.plot(lims, lims, "r--", linewidth=2, label="Perfect")
ax.set_xlabel("Actual CO2 (g/km)", fontsize=10)
ax.set_ylabel("Predicted CO2 (g/km)", fontsize=10)
ax.set_title("Actual vs Predicted\n(Linear Regression)", fontsize=12, fontweight="bold")
ax.legend(fontsize=9)
ax.text(0.05, 0.92, f"R² = {lr_r2:.5f}", transform=ax.transAxes, fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("graphs/CO2_Analysis_Dashboard.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → graphs/CO2_Analysis_Dashboard.png")

# --- Figure 2: Feature Importance ---
rf_from_stack = stacking_model.named_estimators_["random_forest"]
fi = pd.Series(
    rf_from_stack.feature_importances_,
    index=X_encoded.columns
).sort_values(ascending=False)[:15]

colors_fi = [
    "#E74C3C" if any(p in name for p in [
        "Thermal", "Displacement", "City_Hwy", "City_Comb",
        "Volumetric", "Harmonic", "Stoich"
    ]) else "#5B8DB8"
    for name in fi.index
]
plt.figure(figsize=(10, 6))
sns.barplot(x=fi.values, y=fi.index, palette=colors_fi)
plt.title("Top 15 Feature Importances (RF Base Learner)\nRed = Physics-Informed | Blue = Original",
          fontsize=12, fontweight="bold")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("graphs/CO2_Feature_Importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("  → graphs/CO2_Feature_Importance.png")

# --- Figure 3: Full Model Comparison Dashboard ---
all_models = {
    "Linear\nRegression": lr_model,
    "Random\nForest":     rf_model,
    "Gradient\nBoosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05,
                                                     max_depth=5, subsample=0.8, random_state=42),
    "Extra\nTrees":       ExtraTreesRegressor(n_estimators=200, max_depth=12,
                                               min_samples_leaf=2, random_state=42, n_jobs=-1),
    "Stacking\nEnsemble": stacking_model,
}
palette    = ["#95A5A6", "#3498DB", "#E8A838", "#9B59B6", "#2ECC71"]
all_results = {}
all_preds   = []
for name, mdl in all_models.items():
    if name not in ["Linear\nRegression", "Random\nForest", "Stacking\nEnsemble"]:
        mdl.fit(X_train, y_train)
    p = mdl.predict(X_test)
    all_preds.append(p)
    all_results[name] = {
        "R2":   r2_score(y_test, p),
        "MAE":  mean_absolute_error(y_test, p),
        "RMSE": np.sqrt(mean_squared_error(y_test, p)),
        "MAPE": np.mean(np.abs((y_test - p) / y_test)) * 100,
        "RAE":  np.sum(np.abs(y_test - p)) / np.sum(np.abs(y_test - np.mean(y_test))),
    }

m_names = list(all_results.keys())
r2s     = [all_results[n]["R2"]   for n in m_names]
maes    = [all_results[n]["MAE"]  for n in m_names]
rmses   = [all_results[n]["RMSE"] for n in m_names]
mapes   = [all_results[n]["MAPE"] for n in m_names]
raes    = [all_results[n]["RAE"]  for n in m_names]
STACK   = "Stacking\nEnsemble"
edge    = ["#27AE60" if n==STACK else "none" for n in m_names]

fig_c = plt.figure(figsize=(18, 13))
fig_c.patch.set_facecolor("#F8F9FA")
gs    = gridspec.GridSpec(3, 3, figure=fig_c, hspace=0.45, wspace=0.35)
fig_c.suptitle("Model Comparison Dashboard — CO2 Emissions Prediction",
               fontsize=16, fontweight="bold", color="#2C3E50", y=1.01)

def style_ax(ax, ttl):
    ax.set_facecolor("#FFFFFF")
    ax.set_title(ttl, fontsize=11, fontweight="bold", color="#2C3E50", pad=8)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.spines[["top","right"]].set_visible(False)
    for lbl in ax.get_xticklabels(): lbl.set_fontsize(8.5)

# R² bar
ax1 = fig_c.add_subplot(gs[0, 0])
bars = ax1.bar(m_names, r2s, color=palette, edgecolor=edge, linewidth=2.5, width=0.6)
ax1.set_ylim(min(r2s)-0.03, 1.0); ax1.set_ylabel("R² Score", fontsize=10)
for b, v in zip(bars, r2s):
    ax1.text(b.get_x()+b.get_width()/2, b.get_height()+0.0005,
             f"{v:.5f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
style_ax(ax1, "R² Score (Higher = Better)")

# MAE bar
ax2 = fig_c.add_subplot(gs[0, 1])
bars = ax2.bar(m_names, maes, color=palette, edgecolor=edge, linewidth=2.5, width=0.6)
ax2.set_ylabel("MAE (g/km)", fontsize=10)
for b, v in zip(bars, maes):
    ax2.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
style_ax(ax2, "Mean Absolute Error (Lower = Better)")

# RMSE bar
ax3 = fig_c.add_subplot(gs[0, 2])
bars = ax3.bar(m_names, rmses, color=palette, edgecolor=edge, linewidth=2.5, width=0.6)
ax3.set_ylabel("RMSE (g/km)", fontsize=10)
for b, v in zip(bars, rmses):
    ax3.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
             f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
style_ax(ax3, "Root Mean Squared Error (Lower = Better)")

# Actual vs Predicted overlay
ax4 = fig_c.add_subplot(gs[1, :2])
ax4.set_facecolor("#FFFFFF")
lims = [y_test.min(), y_test.max()]
ax4.plot(lims, lims, "k--", linewidth=1.5, alpha=0.6, label="Perfect", zorder=5)
for n, p, c in zip(m_names, all_preds, palette):
    lw = 3 if n==STACK else 1.5
    ax4.scatter(y_test, p, alpha=0.55 if n==STACK else 0.2,
                color=c, s=8 if n==STACK else 5,
                label=n.replace("\n"," "), edgecolors="none",
                zorder=4 if n==STACK else 3)
ax4.set_xlabel("Actual CO2 (g/km)", fontsize=10)
ax4.set_ylabel("Predicted CO2 (g/km)", fontsize=10)
ax4.set_title("Actual vs Predicted — All Models", fontsize=11, fontweight="bold", color="#2C3E50", pad=8)
ax4.legend(loc="upper left", fontsize=8, framealpha=0.9)
ax4.spines[["top","right"]].set_visible(False); ax4.grid(alpha=0.2)

# Residuals boxplot
ax5 = fig_c.add_subplot(gs[1, 2])
res_all = [y_test.values - p for p in all_preds]
bp = ax5.boxplot(res_all, tick_labels=[n.replace("\n"," ") for n in m_names],
                 patch_artist=True,
                 medianprops=dict(color="black", linewidth=2),
                 flierprops=dict(marker="o", markersize=2, alpha=0.4))
for patch, color in zip(bp["boxes"], palette):
    patch.set_facecolor(color); patch.set_alpha(0.7)
ax5.axhline(0, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
ax5.set_ylabel("Residual (g/km)", fontsize=10)
ax5.set_title("Residuals Boxplot", fontsize=11, fontweight="bold", color="#2C3E50", pad=8)
ax5.tick_params(axis="x", labelsize=7)
ax5.spines[["top","right"]].set_visible(False); ax5.grid(axis="y", alpha=0.2)

# KDE error distribution
ax6 = fig_c.add_subplot(gs[2, :])
ax6.set_facecolor("#FFFFFF")
for n, p, c in zip(m_names, all_preds, palette):
    residuals = y_test.values - p
    sns.kdeplot(residuals, ax=ax6, color=c, linewidth=3 if n==STACK else 1.5,
                linestyle="-" if n==STACK else "--", label=n.replace("\n"," "))
ax6.axvline(0, color="black", linewidth=1, alpha=0.5)
ax6.set_xlabel("Prediction Error (g/km)", fontsize=10)
ax6.set_ylabel("Density", fontsize=10)
ax6.set_title("Error Distribution — All Models  |  Narrower & taller = better",
              fontsize=11, fontweight="bold", color="#2C3E50", pad=8)
ax6.legend(fontsize=9, loc="upper right", framealpha=0.9)
ax6.set_xlim(-30, 30)
ax6.spines[["top","right"]].set_visible(False); ax6.grid(alpha=0.2)

plt.savefig("graphs/graph_model_comparison.png", dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
plt.show()
print("  → graphs/graph_model_comparison.png")

# --- Figure 4: Per-Model Deep Dive ---
fig4, axes4 = plt.subplots(2, 3, figsize=(18, 11))
fig4.patch.set_facecolor("#F8F9FA")
fig4.suptitle("Per-Model Deep Dive — Actual vs Predicted",
              fontsize=15, fontweight="bold", color="#2C3E50", y=1.01)
for idx, (n, p, c) in enumerate(zip(m_names, all_preds, palette)):
    row, col = divmod(idx, 3)
    ax = axes4[row][col]
    ax.set_facecolor("#FFFFFF")
    ax.scatter(y_test, p, alpha=0.3, color=c, s=8, edgecolors="none")
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", linewidth=2)
    r2v = all_results[n]["R2"]; mae_v = all_results[n]["MAE"]; rmse_v = all_results[n]["RMSE"]
    mape_v = all_results[n]["MAPE"]; rae_v = all_results[n]["RAE"]
    ax.set_title(n.replace("\n"," "), fontsize=11, fontweight="bold",
                 color="#27AE60" if n==STACK else "#2C3E50")
    ax.set_xlabel("Actual CO2 (g/km)", fontsize=9)
    ax.set_ylabel("Predicted CO2 (g/km)", fontsize=9)
    ax.text(0.04, 0.93, f"R²={r2v:.5f}\nMAE={mae_v:.3f}\nRMSE={rmse_v:.3f}\nMAPE={mape_v:.2f}%\nRAE={rae_v:.3f}",
            transform=ax.transAxes, fontsize=8.5,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.85), va="top")
    ax.spines[["top","right"]].set_visible(False); ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("graphs/graph_per_model_deepdive.png", dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
plt.show()
print("  → graphs/graph_per_model_deepdive.png")

# --- Figure 5: Scorecard Heatmap ---
fig5, ax_h = plt.subplots(figsize=(11, 5))
fig5.patch.set_facecolor("#F8F9FA"); ax_h.set_facecolor("#F8F9FA")
short   = [n.replace("\n"," ") for n in m_names]
data_h  = np.array([[r2s[i], maes[i], rmses[i], mapes[i], raes[i]] for i in range(len(m_names))])
norm_h  = data_h.copy().astype(float)
norm_h[:, 0] = (data_h[:,0]-data_h[:,0].min())/(data_h[:,0].max()-data_h[:,0].min())
norm_h[:, 1] = 1-(data_h[:,1]-data_h[:,1].min())/(data_h[:,1].max()-data_h[:,1].min())
norm_h[:, 2] = 1-(data_h[:,2]-data_h[:,2].min())/(data_h[:,2].max()-data_h[:,2].min())
norm_h[:, 3] = 1-(data_h[:,3]-data_h[:,3].min())/(data_h[:,3].max()-data_h[:,3].min())
norm_h[:, 4] = 1-(data_h[:,4]-data_h[:,4].min())/(data_h[:,4].max()-data_h[:,4].min())
im = ax_h.imshow(norm_h.T, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
ax_h.set_xticks(range(len(short))); ax_h.set_xticklabels(short, fontsize=11)
ax_h.set_yticks([0,1,2,3,4]); ax_h.set_yticklabels(["R² ↑","MAE ↓","RMSE ↓","MAPE ↓","RAE ↓"], fontsize=11, fontweight="bold")
for i in range(len(m_names)):
    for j, (v, raw) in enumerate(zip(norm_h[i], data_h[i])):
        if j == 0:
            text_val = f"{raw:.5f}"
        elif j == 3:
            text_val = f"{raw:.2f}%"
        else:
            text_val = f"{raw:.3f}"
        ax_h.text(i, j, text_val,
                  ha="center", va="center", fontsize=9, fontweight="bold", color="black")
ax_h.set_title("Model Scorecard Heatmap  |  Green = Best", fontsize=13, fontweight="bold", color="#2C3E50", pad=12)
plt.colorbar(im, ax=ax_h, label="Normalized Score (1 = Best)")
plt.tight_layout()
plt.savefig("graphs/graph_scorecard_heatmap.png", dpi=150, bbox_inches="tight", facecolor="#F8F9FA")
plt.show()
print("  → graphs/graph_scorecard_heatmap.png")

print("\n[DONE] All steps complete.")
print("\nFolder structure:")
print("  CO2_project/")
print("  ├── CO2_Complete_Analysis.py")
print("  ├── CO2 Emissions_Canada.csv")
print("  ├── Data_Description.csv")
print("  └── graphs/")
print("      ├── CO2_Analysis_Dashboard.png")
print("      ├── CO2_Feature_Importance.png")
print("      ├── graph_model_comparison.png")
print("      ├── graph_per_model_deepdive.png")
print("      └── graph_scorecard_heatmap.png")