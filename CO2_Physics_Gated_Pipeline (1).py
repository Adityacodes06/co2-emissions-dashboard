"""
=============================================================================
  PHYSICS-GATED PREDICTION PIPELINE FOR VEHICLE CO2
=============================================================================
  Novel Contributions:
    1. Extended Metrics: MAPE, RAE, Max Error added for regulatory contexts.
    2. Physics-Residual ML Base (PRNN-style): ML only learns the residual
       error between complete combustion stoichiometry and reality.
    3. Multi-Layer OOD Detection:
       - Layer 1: Feature Space OOD (Isolation Forest)
       - Layer 2: Physics Consistency OOD (Combustion Efficiency vs population)
    4. Adaptive Physics-Gated Pipeline:
       Dynamically routes predictions & fall back to pure physics bounds 
       based on OOD severity.
=============================================================================
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, max_error
try:
    from category_encoders import TargetEncoder
except ImportError:
    TargetEncoder = None
    print("[WARN] category_encoders not installed. Target encoding will be skipped or done manually.")

warnings.filterwarnings("ignore")
os.makedirs("graphs", exist_ok=True)

print("=" * 65)
print("  STEP 1: DATASET OVERVIEW & PHYSICS FEATURES")
print("=" * 65)

df = pd.read_csv("data/CO2 Emissions_Canada.csv")

df["_FuelType_raw"] = df["Fuel Type"].copy()
df["_VehicleClass_raw"] = df["Vehicle Class"].copy()

df["Displacement_per_Cylinder"] = df["Engine Size(L)"] / df["Cylinders"]
df["Thermal_Load_Index"] = (df["Engine Size(L)"] * df["Cylinders"] * df["Fuel Consumption Comb (L/100 km)"]) / 100
df["City_Hwy_Spread"] = df["Fuel Consumption City (L/100 km)"] - df["Fuel Consumption Hwy (L/100 km)"]
df["City_Comb_Ratio"] = df["Fuel Consumption City (L/100 km)"] / df["Fuel Consumption Comb (L/100 km)"]
df["Volumetric_Efficiency_Proxy"] = df["Displacement_per_Cylinder"] / df["Fuel Consumption Comb (L/100 km)"]
df["Harmonic_Fuel_Mean"] = 2 / (1 / df["Fuel Consumption City (L/100 km)"] + 1 / df["Fuel Consumption Hwy (L/100 km)"])

FUEL_CO2_FACTOR = {"X": 2289, "Z": 2289, "D": 2640, "E": 1594, "N": 1900}
df["Stoich_CO2_Estimate"] = df["Fuel Consumption Comb (L/100 km)"] * df["Fuel Type"].map(FUEL_CO2_FACTOR).fillna(2289) / 100

print(f"Added 7 Physics-informed features.")

print("\n" + "=" * 65)
print("  STEP 2: ENCODING & SPLIT")
print("=" * 65)

y = df["CO2 Emissions(g/km)"]
X_raw = df.drop(columns=["CO2 Emissions(g/km)", "_FuelType_raw", "_VehicleClass_raw"])

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.2, random_state=42
)

high_card_cols = ["Make", "Model"]
if TargetEncoder is not None:
    te = TargetEncoder(cols=high_card_cols, smoothing=10)
    X_train_enc = te.fit_transform(X_train_raw, y_train)
    X_test_enc  = te.transform(X_test_raw)
else:
    X_train_enc, X_test_enc = X_train_raw.copy(), X_test_raw.copy()
    for col in high_card_cols:
        means = y_train.groupby(X_train_raw[col]).mean()
        X_train_enc[col + "_TE"] = X_train_raw[col].map(means)
        X_test_enc[col + "_TE"]  = X_test_raw[col].map(means).fillna(means.mean())
    X_train_enc.drop(columns=high_card_cols, inplace=True)
    X_test_enc.drop(columns=high_card_cols, inplace=True)

low_card_cols = ["Vehicle Class", "Transmission", "Fuel Type"]
X_train_final = pd.get_dummies(X_train_enc, columns=low_card_cols, drop_first=True)
X_test_final  = pd.get_dummies(X_test_enc,  columns=low_card_cols, drop_first=True)
X_train_final, X_test_final = X_train_final.align(X_test_final, join="left", axis=1, fill_value=0)

stoich_train = X_train_final["Stoich_CO2_Estimate"].values
stoich_test  = X_test_final["Stoich_CO2_Estimate"].values

y_train_residual = y_train.values - stoich_train

print("\n" + "=" * 65)
print("  STEP 3: EXTENDED EVALUATION METRICS")
print("=" * 65)

def mape_score(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

def rae_score(y_true, y_pred):
    return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))

def evaluate_predictions(name, y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mape_score(y_true, y_pred)
    rae  = rae_score(y_true, y_pred)
    max_err = max_error(y_true, y_pred)

    print(f"\n  [ {name} ]")
    print(f"  R2        : {r2:.5f}")
    print(f"  MAE       : {mae:.3f} g/km")
    print(f"  RMSE      : {rmse:.3f} g/km")
    print(f"  MAPE      : {mape:.2f}%     <-- (Normalized relative error format)")
    print(f"  RAE       : {rae:.3f}       <-- (< 1 means better than mean baseline)")
    print(f"  Max Error : {max_err:.1f} g/km    <-- (Worst-case compliance error)")
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE": mape, "RAE": rae, "Max_Error": max_err}

print("  Metrics configured: R2, MAE, RMSE + MAPE, RAE, Max Error.")

print("\n" + "=" * 65)
print("  STEP 4: PHYSICS-RESIDUAL NEURAL NETWORK (PRNN) APPROACH")
print("=" * 65)
print("  The model natively learns to predict the residual difference")
print("  between complete stoichiometric combustion and empirical reality.")

prnn_residual_model = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32),
                         activation='relu', solver='adam',
                         max_iter=300, random_state=42))
])

prnn_residual_model.fit(X_train_final, y_train_residual)

prnn_residual_preds = prnn_residual_model.predict(X_test_final)
prnn_final_preds = stoich_test + prnn_residual_preds
prnn_final_preds = np.maximum(prnn_final_preds, stoich_test)

evaluate_predictions("PRNN Base (ML Residual + Physics Bound)", y_test.values, prnn_final_preds)

print("\n" + "=" * 65)
print("  STEP 5: MULTI-LAYER OUT-OF-DISTRIBUTION (OOD) DETECTION")
print("=" * 65)

iso_forest = IsolationForest(contamination=0.03, random_state=42)
iso_forest.fit(X_train_final)
ood_layer1_scores = -iso_forest.score_samples(X_test_final) 
ood_layer1_norm = (ood_layer1_scores - ood_layer1_scores.min()) / (ood_layer1_scores.max() - ood_layer1_scores.min())

comb_eff_train = X_train_final["Displacement_per_Cylinder"] / X_train_final["Fuel Consumption Comb (L/100 km)"]
comb_eff_test  = X_test_final["Displacement_per_Cylinder"] / X_test_final["Fuel Consumption Comb (L/100 km)"]

ce_mean = comb_eff_train.mean()
ce_std  = comb_eff_train.std()
ood_layer2_norm = np.clip(np.abs(comb_eff_test - ce_mean) / (ce_std * 3), 0, 1)

ood_severity_score = 0.5 * ood_layer1_norm + 0.5 * ood_layer2_norm

print(f"  OOD Multi-Layer Scorers calibrated.")
print(f"  Max severity on test set: {ood_severity_score.max():.3f}")

print("\n" + "=" * 65)
print("  STEP 6: PHYSICS-GATED PREDICTION PIPELINE")
print("=" * 65)

gated_preds = np.zeros_like(prnn_final_preds)
trust_tiers = []

for i in range(len(X_test_final)):
    sev = ood_severity_score.iloc[i] if hasattr(ood_severity_score, 'iloc') else ood_severity_score[i]
    stoich_base = stoich_test[i]
    prnn_pred = prnn_final_preds[i]
    
    if sev < 0.25:
        gated_preds[i] = prnn_pred
        trust_tiers.append("1 - High Confidence")
    elif sev < 0.50:
        gated_preds[i] = prnn_pred
        trust_tiers.append("2 - Moderate (Flagged)")
    elif sev < 0.75:
        gated_preds[i] = (0.7 * stoich_base) + (0.3 * prnn_pred)
        trust_tiers.append("3 - Low (Physics Anchored)")
    else:
        gated_preds[i] = stoich_base
        trust_tiers.append("4 - OOD (Pure Physics Baseline)")

test_results = pd.DataFrame({
    "Actual": y_test.values,
    "Stoich_Bound": stoich_test,
    "PRNN_Pred": prnn_final_preds,
    "Gated_Predicted": gated_preds,
    "OOD_Score": ood_severity_score.values if hasattr(ood_severity_score, 'values') else ood_severity_score,
    "Trust_Tier": trust_tiers
})

evaluate_predictions("Full Physics-Gated Pipeline Output", test_results["Actual"], test_results["Gated_Predicted"])

print("\n  Breakdown of predictions by Confidence Tier:")
tier_counts = test_results["Trust_Tier"].value_counts().sort_index()
for tier, count in tier_counts.items():
    print(f"    {tier:<35} : {count} vehicles")

print("\n" + "=" * 65)
print("  STEP 7: VISUALISATIONS (Novel OOD + Error Analysis)")
print("=" * 65)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Physics-Gated Pipeline & OOD Analysis", fontsize=14, fontweight='bold')

physics_error = test_results["Actual"] - test_results["Stoich_Bound"]
residual_error = test_results["Actual"] - test_results["Gated_Predicted"]

sns.kdeplot(physics_error, fill=True, alpha=0.3, label='Physics Error (Actual - Stoich)', ax=axes[0], color='#E74C3C')
sns.kdeplot(residual_error, fill=True, alpha=0.5, label='Pipeline Error (Actual - Gated Pred)', ax=axes[0], color='#2ECC71')
axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
axes[0].set_title("Decomposed Error Distribution", fontsize=12)
axes[0].set_xlabel("Error Magnitude (g/km)", fontsize=10)
axes[0].set_ylabel("Density", fontsize=10)
axes[0].legend()
axes[0].set_xlim(-40, 150)

test_results["Abs_Error"] = np.abs(test_results["Actual"] - test_results["Gated_Predicted"])
trust_palette = {"1 - High Confidence": "#2ECC71", 
                 "2 - Moderate (Flagged)": "#F1C40F", 
                 "3 - Low (Physics Anchored)": "#E67E22", 
                 "4 - OOD (Pure Physics Baseline)": "#E74C3C"}

sns.scatterplot(x="OOD_Score", y="Abs_Error", hue="Trust_Tier", 
                palette=trust_palette, data=test_results, alpha=0.7, ax=axes[1], s=40)
axes[1].set_title("Absolute Error vs OOD Severity Score", fontsize=12)
axes[1].set_xlabel("OOD Severity Score (0 to 1)", fontsize=10)
axes[1].set_ylabel("Absolute Prediction Error (g/km)", fontsize=10)
axes[1].legend(title="Prediction Trust Tier", bbox_to_anchor=(1.05, 1), loc='upper left')
axes[1].set_ylim(0, test_results["Abs_Error"].max() * 1.05)

plt.tight_layout()
plt.subplots_adjust(top=0.9)
plt.savefig("graphs/Physics_Gated_Pipeline_Analysis.png", dpi=150, bbox_inches="tight")
print("  -> Saved 'graphs/Physics_Gated_Pipeline_Analysis.png'")

print("\n[DONE] Physics-Gated Pipeline executed successfully. Original codebase untouched.")