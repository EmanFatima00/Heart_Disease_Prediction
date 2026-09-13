"""
Heart Disease Prediction — Advanced ML Pipeline
Author: Eman Fatima | BS-AI @ PAF-IAST | ML Intern @ ProSensia
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_auc_score, roc_curve,
                             f1_score, precision_score, recall_score)
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

# ── Paths (works on any OS) ───────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(SCRIPT_DIR, "heart_disease_uci.csv")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 60)
print("  HEART DISEASE PREDICTION — ADVANCED ML PIPELINE")
print("=" * 60)

# ── 1. Load Data ──────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
print(f"\n✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# ── 2. Target Engineering ─────────────────────────────────────
df['target'] = (df['num'] > 0).astype(int)
df.drop(columns=['num', 'id', 'dataset'], inplace=True, errors='ignore')

print(f"\n📊 Target distribution:")
print(f"   No Disease: {(df['target']==0).sum()} ({(df['target']==0).mean()*100:.1f}%)")
print(f"   Disease:    {(df['target']==1).sum()} ({(df['target']==1).mean()*100:.1f}%)")

# ── 3. Feature Engineering ────────────────────────────────────
df['age_group']  = pd.cut(df['age'], bins=[0,40,55,65,100],
                           labels=['Young','Middle','Senior','Elderly'])
df['bp_category'] = pd.cut(df['trestbps'].fillna(df['trestbps'].median()),
                            bins=[0,120,130,140,300],
                            labels=['Normal','Elevated','High1','High2'])
df['chol_risk']  = (df['chol'] > 240).astype(int)
df['hr_reserve'] = df['thalch'] - df['age']
print("\n✅ Feature engineering complete — 4 new features added")

# ── 4. Preprocessing ──────────────────────────────────────────
categorical_cols = df.select_dtypes(include=['object','category']).columns.tolist()
for col in categorical_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype in ['float64','int64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

print("✅ Missing values handled")

# ── 5. Split & SMOTE ──────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']
feature_names = X.columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

X_train = X_train.fillna(X_train.median(numeric_only=True))
X_test  = X_test.fillna(X_train.median(numeric_only=True))

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
print(f"✅ SMOTE applied — training samples: {len(X_train)} → {len(X_train_res)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_test_scaled  = scaler.transform(X_test)

# ── 6. Train Models ───────────────────────────────────────────
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=0.1, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=200, max_depth=8,
                                                   min_samples_split=5, random_state=42),
    'XGBoost':             XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                                          subsample=0.8, colsample_bytree=0.8,
                                          eval_metric='logloss', random_state=42),
    'Gradient Boosting':   GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                       learning_rate=0.05, random_state=42),
    'SVM':                 SVC(kernel='rbf', probability=True, C=1.0, random_state=42),
}

print("\n" + "=" * 60)
print("  MODEL COMPARISON")
print("=" * 60)

results = {}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train_scaled, y_train_res)
    preds    = model.predict(X_test_scaled)
    proba    = model.predict_proba(X_test_scaled)[:, 1]
    cv_scores = cross_val_score(model, X_train_scaled, y_train_res,
                                cv=cv, scoring='accuracy')
    results[name] = {
        'model':    model,
        'accuracy': accuracy_score(y_test, preds),
        'f1':       f1_score(y_test, preds, average='weighted'),
        'roc_auc':  roc_auc_score(y_test, proba),
        'cv_mean':  cv_scores.mean(),
        'cv_std':   cv_scores.std(),
        'preds':    preds,
        'proba':    proba,
    }
    print(f"\n📌 {name}")
    print(f"   Accuracy : {results[name]['accuracy']*100:.2f}%")
    print(f"   F1 Score : {results[name]['f1']:.4f}")
    print(f"   ROC-AUC  : {results[name]['roc_auc']:.4f}")
    print(f"   CV Score : {results[name]['cv_mean']*100:.2f}% ± {results[name]['cv_std']*100:.2f}%")

# ── 7. Best Model ─────────────────────────────────────────────
best_name = max(results, key=lambda x: results[x]['roc_auc'])
best      = results[best_name]
print(f"\n🏆 Best Model: {best_name}")
print(f"   Accuracy : {best['accuracy']*100:.2f}%")
print(f"   ROC-AUC  : {best['roc_auc']:.4f}")
print(f"\n{classification_report(y_test, best['preds'], target_names=['No Disease','Disease'])}")

# ── 8. Ensemble ───────────────────────────────────────────────
print("🔗 Training Ensemble Voting Classifier...")
ensemble = VotingClassifier(
    estimators=[
        ('rf',  RandomForestClassifier(n_estimators=200, random_state=42)),
        ('xgb', XGBClassifier(n_estimators=200, eval_metric='logloss', random_state=42)),
        ('gb',  GradientBoostingClassifier(n_estimators=200, random_state=42)),
    ], voting='soft')
ensemble.fit(X_train_scaled, y_train_res)
ens_preds = ensemble.predict(X_test_scaled)
ens_proba = ensemble.predict_proba(X_test_scaled)[:, 1]
ens_acc   = accuracy_score(y_test, ens_preds)
ens_auc   = roc_auc_score(y_test, ens_proba)
print(f"   Ensemble Accuracy : {ens_acc*100:.2f}%")
print(f"   Ensemble ROC-AUC  : {ens_auc:.4f}")

# Select final model
if ens_auc > best['roc_auc']:
    final_model, final_name  = ensemble, "Ensemble (RF+XGB+GB)"
    final_preds, final_proba = ens_preds, ens_proba
    final_acc,   final_auc   = ens_acc,   ens_auc
else:
    final_model, final_name  = best['model'], best_name
    final_preds, final_proba = best['preds'], best['proba']
    final_acc,   final_auc   = best['accuracy'], best['roc_auc']

print(f"\n✅ Final Model : {final_name}")
print(f"   Accuracy    : {final_acc*100:.2f}%")
print(f"   ROC-AUC     : {final_auc:.4f}")

# ── 9. Save ───────────────────────────────────────────────────
joblib.dump(final_model,   os.path.join(OUTPUT_DIR, "heart_disease_model.pkl"))
joblib.dump(scaler,        os.path.join(OUTPUT_DIR, "scaler.pkl"))
joblib.dump(feature_names, os.path.join(OUTPUT_DIR, "feature_names.pkl"))
print("\n✅ Model, scaler and feature names saved to outputs/")

# ── 10. Visualizations ────────────────────────────────────────
print("📊 Generating visualizations...")

# A. Model Comparison
names_  = list(results.keys()) + ["Ensemble"]
accs_   = [results[n]['accuracy']*100 for n in results] + [ens_acc*100]
aucs_   = [results[n]['roc_auc']      for n in results] + [ens_auc]
f1s_    = [results[n]['f1']           for n in results] + [f1_score(y_test, ens_preds, average='weighted')]
colors_ = ['#6366F1','#059669','#D97706','#0891B2','#DC2626','#7C3AED']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Model Performance Comparison", fontsize=14, fontweight='bold')
for ax, vals, title, xlim in zip(
        axes, [accs_, aucs_, f1s_],
        ["Accuracy (%)","ROC-AUC","F1 Score"],
        [(60,100),(0.6,1.0),(0.6,1.0)]):
    ax.barh(names_, vals, color=colors_)
    ax.set_title(title); ax.set_xlim(*xlim)
    for i, v in enumerate(vals):
        lbl = f"{v:.1f}%" if title=="Accuracy (%)" else f"{v:.3f}"
        ax.text(v+0.2 if title=="Accuracy (%)" else v+0.003, i, lbl, va='center', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "model_comparison.png"), dpi=150, bbox_inches='tight')
plt.close()

# B. ROC Curves
plt.figure(figsize=(9,6))
for name, res in results.items():
    fpr, tpr, _ = roc_curve(y_test, res['proba'])
    plt.plot(fpr, tpr, label=f"{name} (AUC={res['roc_auc']:.3f})", linewidth=2)
fpr_e, tpr_e, _ = roc_curve(y_test, ens_proba)
plt.plot(fpr_e, tpr_e, label=f"Ensemble (AUC={ens_auc:.3f})",
         linewidth=2.5, linestyle='--', color='black')
plt.plot([0,1],[0,1],'gray',linestyle=':')
plt.title("ROC Curves — All Models", fontsize=13, fontweight='bold')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.legend(loc='lower right', fontsize=9); plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "roc_curves.png"), dpi=150, bbox_inches='tight')
plt.close()

# C. Confusion Matrix
plt.figure(figsize=(6,5))
cm = confusion_matrix(y_test, final_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Disease','Disease'],
            yticklabels=['No Disease','Disease'])
plt.title(f"Confusion Matrix — {final_name}", fontweight='bold')
plt.ylabel("Actual"); plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "confusion_matrix.png"), dpi=150, bbox_inches='tight')
plt.close()

# D. Feature Importance
rf_model = results['Random Forest']['model']
importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
plt.figure(figsize=(9,7))
colors_fi = ['#6366F1' if v > importances.median() else '#CBD5E1' for v in importances]
importances.plot(kind='barh', color=colors_fi)
plt.title("Feature Importance — Random Forest", fontsize=13, fontweight='bold')
plt.xlabel("Importance Score"); plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "feature_importance.png"), dpi=150, bbox_inches='tight')
plt.close()

# E. Correlation Heatmap
plt.figure(figsize=(12,9))
numeric_df = df.select_dtypes(include=[np.number])
mask = np.triu(np.ones_like(numeric_df.corr(), dtype=bool))
sns.heatmap(numeric_df.corr(), mask=mask, annot=True, fmt='.2f',
            cmap='coolwarm', center=0, linewidths=0.5, annot_kws={'size':8})
plt.title("Feature Correlation Heatmap", fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_heatmap.png"), dpi=150, bbox_inches='tight')
plt.close()

# F. Age & Cholesterol Distribution
fig, axes = plt.subplots(1, 2, figsize=(12,5))
sns.histplot(data=df, x='age', hue='target', bins=20, kde=True,
             ax=axes[0], palette=['#059669','#DC2626'])
axes[0].set_title("Age Distribution by Heart Disease Status")
axes[0].legend(title='Heart Disease', labels=['No','Yes'])
sns.boxplot(data=df, x='target', y='chol', ax=axes[1],
            palette=['#059669','#DC2626'])
axes[1].set_title("Cholesterol by Heart Disease Status")
axes[1].set_xticklabels(['No Disease','Disease'])
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "age_chol_distribution.png"), dpi=150, bbox_inches='tight')
plt.close()

print("✅ All visualizations saved to outputs/")
print("\n" + "=" * 60)
print("  PIPELINE COMPLETE")
print(f"  Final Model  : {final_name}")
print(f"  Accuracy     : {final_acc*100:.2f}%")
print(f"  ROC-AUC      : {final_auc:.4f}")
print("=" * 60)