import sys
import subprocess

# Auto-install missing packages (works in VS Code, Jupyter, and Google Colab)
required = ['scikit-learn', 'pandas', 'numpy', 'matplotlib', 'seaborn']
for pkg in required:
    module = pkg.replace('-', '_').split('[')[0]
    try:
        __import__(module)
    except ImportError:
        print(f'Installing {pkg}...')
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
        print(f'  Done.')

# Core libraries
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn — preprocessing
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

# Scikit-learn — models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier

# Scikit-learn — metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix,
    classification_report, ConfusionMatrixDisplay
)

# Plot style
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
COLORS = ['#2563EB', '#16A34A', '#DC2626', '#D97706']
MODELS_LABEL = ['Logistic Regression', 'SVM (RBF)', 'Decision Tree', 'Random Forest']

print('✅ All libraries imported successfully!')


# ---

# Load UCI Heart Disease Dataset (Cleveland subset)
url = 'https://raw.githubusercontent.com/dsrscientist/dataset1/master/heart.csv'

try:
    df = pd.read_csv(url)
    print('✅ Dataset loaded from GitHub mirror.')
except:
    # Fallback: generate synthetic data matching UCI schema
    print('⚠️  URL failed — generating synthetic UCI-schema dataset...')
    np.random.seed(42)
    n = 303
    df = pd.DataFrame({
        'age':      np.random.randint(29, 77, n),
        'sex':      np.random.randint(0, 2, n),
        'cp':       np.random.randint(0, 4, n),
        'trestbps': np.random.randint(94, 200, n),
        'chol':     np.random.randint(126, 564, n),
        'fbs':      np.random.randint(0, 2, n),
        'restecg':  np.random.randint(0, 3, n),
        'thalach':  np.random.randint(71, 202, n),
        'exang':    np.random.randint(0, 2, n),
        'oldpeak':  np.round(np.random.uniform(0, 6.2, n), 1),
        'slope':    np.random.randint(0, 3, n),
        'ca':       np.random.randint(0, 4, n),
        'thal':     np.random.randint(0, 4, n),
        'target':   np.random.randint(0, 2, n),
    })
    print('✅ Synthetic dataset generated.')

# Rename target column if needed
if 'condition' in df.columns:
    df.rename(columns={'condition': 'target'}, inplace=True)

print(f'\n📊 Dataset Shape: {df.shape}')
print(f'\n📋 Column Names: {list(df.columns)}')
df.head(10)

# ---

print('='*55)
print('        DATASET BASIC INFO')
print('='*55)
print(f'  Rows             : {df.shape[0]}')
print(f'  Columns          : {df.shape[1]}')
print(f'  Missing Values   : {df.isnull().sum().sum()}')
print(f'  Duplicate Rows   : {df.duplicated().sum()}')
print(f'  Disease +ve (1)  : {(df["target"]==1).sum()} ({(df["target"]==1).mean()*100:.1f}%)')
print(f'  Disease -ve (0)  : {(df["target"]==0).sum()} ({(df["target"]==0).mean()*100:.1f}%)')
print('='*55)
print('\nData Types & Missing Values:')
print(df.dtypes.to_frame('dtype').join(df.isnull().sum().to_frame('nulls')))

# ---

print('\n📊 Statistical Summary:')
df.describe().round(2)

# ---

# ── EDA Figure 1: Class Distribution + Age Histogram ─────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
fig.suptitle('Figure 1 — Dataset Overview', fontsize=14, fontweight='bold', y=1.01)

# Pie chart
counts = df['target'].value_counts()
axes[0].pie(counts, labels=['Heart Disease (1)', 'No Disease (0)'],
            colors=['#F87171', '#93C5FD'], autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11})
axes[0].set_title('Target Class Distribution', fontweight='bold')

# Age histogram by class
for label, color, name in zip([0,1], ['#93C5FD','#F87171'], ['No Disease','Heart Disease']):
    axes[1].hist(df[df['target']==label]['age'], bins=20, alpha=0.65,
                 color=color, edgecolor='white', label=name)
axes[1].set_xlabel('Age (years)'); axes[1].set_ylabel('Count')
axes[1].set_title('Patient Age Distribution by Class', fontweight='bold')
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_overview.png', dpi=150, bbox_inches='tight')
plt.show()

# ---

# ── EDA Figure 2: Correlation Heatmap ────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 9))
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
            center=0, linewidths=0.5, ax=ax, cbar_kws={'shrink': 0.8})
ax.set_title('Figure 2 — Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('eda_correlation.png', dpi=150, bbox_inches='tight')
plt.show()

# ---

# ── EDA Figure 3: Feature Distributions (key features) ──────────────────
key_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle('Figure 3 — Key Feature Distributions by Target Class', fontsize=13, fontweight='bold')

for ax, feat in zip(axes, key_features):
    for label, color, name in zip([0,1],['#93C5FD','#F87171'],['No Disease','Heart Disease']):
        ax.hist(df[df['target']==label][feat], bins=18, alpha=0.65,
                color=color, edgecolor='white', label=name, density=True)
    ax.set_title(feat, fontweight='bold')
    ax.set_xlabel('Value'); ax.set_ylabel('Density')
    ax.grid(axis='y', alpha=0.3)
axes[0].legend(fontsize=8)
plt.tight_layout()
plt.savefig('eda_distributions.png', dpi=150, bbox_inches='tight')
plt.show()

# ---

# ── EDA Figure 4: Boxplots of continuous features vs Target ──────────────
fig, axes = plt.subplots(1, 5, figsize=(18, 4))
fig.suptitle('Figure 4 — Boxplots: Feature vs Target Class', fontsize=13, fontweight='bold')

for ax, feat in zip(axes, key_features):
    data_0 = df[df['target']==0][feat]
    data_1 = df[df['target']==1][feat]
    bp = ax.boxplot([data_0, data_1], patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    bp['boxes'][0].set_facecolor('#93C5FD')
    bp['boxes'][1].set_facecolor('#F87171')
    ax.set_xticklabels(['No Disease', 'Disease'], fontsize=9)
    ax.set_title(feat, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('eda_boxplots.png', dpi=150, bbox_inches='tight')
plt.show()

# ---

# ── 4.1  Handle Missing Values ──────────────────────────────────────────
print('Missing values before cleaning:')
print(df.isnull().sum())

# Impute continuous columns with median, categorical with mode
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object' or df[col].nunique() <= 5:
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            df[col].fillna(df[col].median(), inplace=True)

# ── 4.2  Remove Duplicates ───────────────────────────────────────────────
before = len(df)
df.drop_duplicates(inplace=True)
print(f'\nRemoved {before - len(df)} duplicate rows.')

# ── 4.3  Features & Target ───────────────────────────────────────────────
X = df.drop('target', axis=1)
y = df['target']

feature_names = list(X.columns)
print(f'\nFeatures ({len(feature_names)}): {feature_names}')
print(f'Target distribution:\n{y.value_counts()}')

# ---

# ── 4.4  Train / Test Split ──────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 4.5  Feature Scaling (z-score standardisation) ──────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f'Training set : {X_train_sc.shape}')
print(f'Test set     : {X_test_sc.shape}')
print('\n✅ Standardisation applied.')

# ---

# ── 4.6  Principal Component Analysis (PCA) ──────────────────────────────
pca_full = PCA()
pca_full.fit(X_train_sc)

ev = pca_full.explained_variance_ratio_
cum_ev = np.cumsum(ev)
n_components_95 = np.argmax(cum_ev >= 0.95) + 1
print(f'Components needed to explain 95% variance: {n_components_95}')

# Apply PCA
pca = PCA(n_components=n_components_95, random_state=42)
X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca  = pca.transform(X_test_sc)

# ── Figure 5: PCA Explained Variance ─────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(10, 4))
ax1.bar(range(1, len(ev)+1), ev*100, color='#3B82F6', alpha=0.8, label='Individual')
ax2 = ax1.twinx()
ax2.plot(range(1, len(cum_ev)+1), cum_ev*100, 'o-', color='#DC2626',
         lw=2, markersize=5, label='Cumulative')
ax2.axhline(95, color='gray', linestyle='--', lw=1, alpha=0.8)
ax2.text(len(ev)*0.85, 96, '95% threshold', ha='right', fontsize=9, color='gray')
ax1.set_xlabel('Principal Component'); ax1.set_ylabel('Individual Variance (%)', color='#3B82F6')
ax2.set_ylabel('Cumulative Variance (%)', color='#DC2626')
ax1.set_title('Figure 5 — PCA: Explained Variance by Component', fontsize=13, fontweight='bold')
lines1, labs1 = ax1.get_legend_handles_labels()
lines2, labs2 = ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2, labs1+labs2, loc='center right')
ax1.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('pca_variance.png', dpi=150, bbox_inches='tight')
plt.show()

# ---

# Define classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42, C=1.0),
    'SVM (RBF)':           SVC(kernel='rbf', probability=True, random_state=42, C=1.0, gamma='scale'),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, max_depth=None,
                                                   random_state=42, n_jobs=-1),
}

# Train on standardised (non-PCA) data for easier interpretability
trained_models = {}
for name, clf in classifiers.items():
    clf.fit(X_train_sc, y_train)
    trained_models[name] = clf
    print(f'✅ {name} trained.')

print('\n🎯 All models trained successfully!')

# ---

# ── Compute all metrics ───────────────────────────────────────────────────
results = []
for name, clf in trained_models.items():
    y_pred = clf.predict(X_test_sc)
    y_prob = clf.predict_proba(X_test_sc)[:, 1]
    results.append({
        'Model':     name,
        'Accuracy':  round(accuracy_score(y_test, y_pred), 4),
        'Precision': round(precision_score(y_test, y_pred), 4),
        'Recall':    round(recall_score(y_test, y_pred), 4),
        'F1 Score':  round(f1_score(y_test, y_pred), 4),
        'ROC-AUC':   round(roc_auc_score(y_test, y_prob), 4),
    })

results_df = pd.DataFrame(results).set_index('Model')

print('='*70)
print('                   MODEL PERFORMANCE COMPARISON')
print('='*70)
print(results_df.to_string())
print('='*70)
print(f'\n🏆 Best Model: {results_df["Accuracy"].idxmax()}')
print(f'   Accuracy  : {results_df["Accuracy"].max()}')
print(f'   ROC-AUC   : {results_df.loc[results_df["Accuracy"].idxmax(), "ROC-AUC"]}')

# ---

# ── Figure 6: Grouped Bar Chart ───────────────────────────────────────────
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'ROC-AUC']
model_names = results_df.index.tolist()
x = np.arange(len(model_names))
w = 0.15
offsets = np.linspace(-2, 2, len(metrics)) * w
metric_colors = ['#2563EB', '#16A34A', '#DC2626', '#D97706', '#7C3AED']

fig, ax = plt.subplots(figsize=(13, 5))
for i, (metric, color) in enumerate(zip(metrics, metric_colors)):
    vals = results_df[metric].values
    bars = ax.bar(x + offsets[i], vals, w, label=metric, color=color, alpha=0.85, edgecolor='white')
    for bar in bars:
        ax.annotate(f'{bar.get_height():.2f}',
                    xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                    xytext=(0, 2), textcoords='offset points',
                    ha='center', va='bottom', fontsize=7)

ax.set_xticks(x); ax.set_xticklabels(model_names, fontsize=11)
ax.set_ylim(0.65, 1.03); ax.set_ylabel('Score')
ax.set_title('Figure 6 — Model Performance Comparison (All Metrics)', fontsize=13, fontweight='bold')
ax.legend(loc='lower right', ncol=3, fontsize=9)
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

# ---

# ── Figure 7: ROC Curves ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, clf), color in zip(trained_models.items(), COLORS):
    y_prob = clf.predict_proba(X_test_sc)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob)
    ax.plot(fpr, tpr, color=color, lw=2, label=f'{name} (AUC = {auc:.3f})')

ax.plot([0,1],[0,1],'k--', lw=1, label='Random Classifier (AUC = 0.500)')
ax.fill_between([0,1],[0,1],[0,1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Figure 7 — ROC Curves for All Classifiers', fontsize=13, fontweight='bold')
ax.legend(fontsize=10, loc='lower right')
ax.grid(linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ---

# ── Figure 8: Confusion Matrices (all 4 models) ───────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))
fig.suptitle('Figure 8 — Confusion Matrices for All Models', fontsize=13, fontweight='bold')

for ax, (name, clf), color in zip(axes, trained_models.items(), COLORS):
    y_pred = clf.predict(X_test_sc)
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['No Disease','Disease'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontweight='bold', fontsize=10)
    ax.set_xlabel('Predicted', fontsize=9)
    ax.set_ylabel('True', fontsize=9)

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.show()

# ---

# ── Detailed Classification Reports ──────────────────────────────────────
for name, clf in trained_models.items():
    y_pred = clf.predict(X_test_sc)
    print(f'\n{"-"*50}')
    print(f' Classification Report — {name}')
    print(f'{"-"*50}')
    print(classification_report(y_test, y_pred, target_names=['No Disease','Heart Disease']))

# ---

# ── Figure 9: Random Forest Feature Importance ───────────────────────────
rf_model = trained_models['Random Forest']
importances = rf_model.feature_importances_
idx = np.argsort(importances)

fig, ax = plt.subplots(figsize=(9, 5.5))
colors_grad = plt.cm.Blues(np.linspace(0.3, 0.9, len(idx)))
bars = ax.barh([feature_names[i] for i in idx], importances[idx], color=colors_grad)
for bar, val in zip(bars, importances[idx]):
    ax.text(val + 0.002, bar.get_y() + bar.get_height()/2,
            f'{val:.4f}', va='center', fontsize=9)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Figure 9 — Random Forest Feature Importance', fontsize=13, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print('\nTop 5 Most Important Features:')
top5 = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:5]
for rank, (feat, score) in enumerate(top5, 1):
    print(f'  {rank}. {feat:12s} → {score:.4f}')

# ---

# ── Figure 10: Decision Tree Visualisation ────────────────────────────────
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    trained_models['Decision Tree'],
    feature_names=feature_names,
    class_names=['No Disease', 'Heart Disease'],
    filled=True, rounded=True, max_depth=3,
    fontsize=9, ax=ax,
    impurity=True, proportion=False
)
ax.set_title('Figure 10 — Decision Tree Structure (Max Depth = 3)',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('decision_tree.png', dpi=120, bbox_inches='tight')
plt.show()

# ---

# ── Figure 11: Learning Curves ────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), sharey=True)
fig.suptitle('Figure 11 — Learning Curves (CV Accuracy vs Training Size)',
             fontsize=13, fontweight='bold')

for ax, (name, clf), color in zip(axes, trained_models.items(), COLORS):
    train_sizes, tr_sc, te_sc = learning_curve(
        clf, X_train_sc, y_train, cv=5,
        train_sizes=np.linspace(0.15, 1.0, 10),
        scoring='accuracy', n_jobs=-1
    )
    tr_mean, tr_std = tr_sc.mean(1), tr_sc.std(1)
    te_mean, te_std = te_sc.mean(1), te_sc.std(1)

    ax.plot(train_sizes, tr_mean, 'o-', color=color, lw=2, label='Train', markersize=4)
    ax.plot(train_sizes, te_mean, 's--', color=color, lw=2, label='CV Val', alpha=0.7, markersize=4)
    ax.fill_between(train_sizes, te_mean-te_std, te_mean+te_std, alpha=0.15, color=color)
    ax.set_title(name, fontweight='bold', fontsize=10)
    ax.set_xlabel('Training Size')
    ax.set_ylim(0.5, 1.05)
    ax.grid(linestyle='--', alpha=0.4)
    ax.legend(fontsize=8)

axes[0].set_ylabel('Accuracy')
plt.tight_layout()
plt.savefig('learning_curves.png', dpi=150, bbox_inches='tight')
plt.show()

# ---

# ── Figure 12: 10-Fold Cross-Validation Boxplot ───────────────────────────
cv_results = {}
print('Running 10-Fold Cross-Validation...\n')
for name, clf in trained_models.items():
    scores = cross_val_score(clf, X_train_sc, y_train, cv=10, scoring='accuracy', n_jobs=-1)
    cv_results[name] = scores
    print(f'  {name:22s} | Mean: {scores.mean():.4f} ± {scores.std():.4f} '
          f'| Min: {scores.min():.4f} | Max: {scores.max():.4f}')

# Boxplot
fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(list(cv_results.values()), patch_artist=True,
                medianprops=dict(color='black', lw=2.5), notch=False)
for patch, color in zip(bp['boxes'], COLORS):
    patch.set_facecolor(color); patch.set_alpha(0.75)
ax.set_xticklabels(list(cv_results.keys()), fontsize=11)
ax.set_ylabel('10-Fold CV Accuracy', fontsize=12)
ax.set_title('Figure 12 — 10-Fold Cross-Validation Accuracy Distribution',
             fontsize=13, fontweight='bold')
for i, (name, scores) in enumerate(cv_results.items()):
    ax.text(i+1, scores.max()+0.005, f'μ={scores.mean():.3f}',
            ha='center', fontsize=9, fontweight='bold')
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('cv_boxplot.png', dpi=150, bbox_inches='tight')
plt.show()

# ---

print('Running GridSearchCV for Random Forest (this may take ~1-2 min)...\n')

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth':    [None, 5, 10],
    'min_samples_split': [2, 5],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=0
)
grid_search.fit(X_train_sc, y_train)

best_rf = grid_search.best_estimator_
y_pred_best = best_rf.predict(X_test_sc)

print(f'Best Parameters  : {grid_search.best_params_}')
print(f'Best CV Accuracy : {grid_search.best_score_:.4f}')
print(f'Test Accuracy    : {accuracy_score(y_test, y_pred_best):.4f}')
print(f'ROC-AUC          : {roc_auc_score(y_test, best_rf.predict_proba(X_test_sc)[:,1]):.4f}')

# ---

def predict_heart_disease(patient_data: dict, model=None, scaler=scaler):
    """
    Predict heart disease risk for a new patient.

    Parameters
    ----------
    patient_data : dict with keys matching feature_names
    model        : trained classifier (default: best Random Forest)
    scaler       : fitted StandardScaler

    Returns
    -------
    dict with prediction, probability, and risk level
    """
    if model is None:
        model = best_rf

    # Build input array in correct feature order
    input_arr = np.array([[patient_data.get(f, 0) for f in feature_names]])
    input_sc  = scaler.transform(input_arr)

    pred  = model.predict(input_sc)[0]
    prob  = model.predict_proba(input_sc)[0][1]
    risk  = 'HIGH RISK 🔴' if prob >= 0.6 else ('MODERATE RISK 🟡' if prob >= 0.4 else 'LOW RISK 🟢')

    return {'Prediction': 'Heart Disease' if pred == 1 else 'No Heart Disease',
            'Probability': round(prob, 4), 'Risk Level': risk}


# ── Example: Patient 1 (high risk profile) ───────────────────────────────
patient1 = {
    'age': 63, 'sex': 1, 'cp': 3, 'trestbps': 145, 'chol': 233,
    'fbs': 1, 'restecg': 0, 'thalach': 150, 'exang': 0,
    'oldpeak': 2.3, 'slope': 0, 'ca': 0, 'thal': 1
}

# ── Example: Patient 2 (low risk profile) ────────────────────────────────
patient2 = {
    'age': 41, 'sex': 0, 'cp': 1, 'trestbps': 120, 'chol': 182,
    'fbs': 0, 'restecg': 1, 'thalach': 170, 'exang': 0,
    'oldpeak': 0.0, 'slope': 2, 'ca': 0, 'thal': 2
}

print('='*50)
print(' PATIENT PREDICTION RESULTS')
print('='*50)
for i, patient in enumerate([patient1, patient2], 1):
    result = predict_heart_disease(patient)
    print(f'\n  Patient {i}:')
    for k, v in result.items():
        print(f'    {k:15s}: {v}')
print('\n' + '='*50)

# ---

# ── Figure 13: Summary Dashboard ─────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.patch.set_facecolor('#F8FAFC')
fig.suptitle('Heart Disease Risk Prediction System — Summary Dashboard\n'
             'Kashish Mohammad | Vridhi Vazirani | Jaaswanth Chikkala | Mahadev | Mohith\n'
             'Woxsen University — B.Tech CSE (Data Science) | 2026',
             fontsize=12, fontweight='bold', y=0.98)

# ── Subplot 1: Accuracy bar ───────────────────────────────────────────────
ax1 = fig.add_subplot(2, 3, 1)
accs = results_df['Accuracy'].values
bars = ax1.bar(range(len(accs)), accs, color=COLORS, edgecolor='white', width=0.6)
ax1.set_xticks(range(len(accs)))
ax1.set_xticklabels(['LR','SVM','DT','RF'], fontsize=10)
ax1.set_ylim(0.6, 1.0); ax1.set_ylabel('Accuracy'); ax1.set_title('Model Accuracy', fontweight='bold')
for bar, v in zip(bars, accs):
    ax1.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
ax1.grid(axis='y', alpha=0.3)

# ── Subplot 2: ROC AUC bar ────────────────────────────────────────────────
ax2 = fig.add_subplot(2, 3, 2)
aucs = results_df['ROC-AUC'].values
bars2 = ax2.bar(range(len(aucs)), aucs, color=COLORS, edgecolor='white', width=0.6)
ax2.set_xticks(range(len(aucs)))
ax2.set_xticklabels(['LR','SVM','DT','RF'], fontsize=10)
ax2.set_ylim(0.6, 1.0); ax2.set_ylabel('AUC'); ax2.set_title('ROC-AUC Score', fontweight='bold')
for bar, v in zip(bars2, aucs):
    ax2.text(bar.get_x()+bar.get_width()/2, v+0.005, f'{v:.2f}', ha='center', fontsize=10, fontweight='bold')
ax2.grid(axis='y', alpha=0.3)

# ── Subplot 3: Radar / Spider Chart ──────────────────────────────────────
ax3 = fig.add_subplot(2, 3, 3, polar=True)
cats = ['Accuracy','Precision','Recall','F1','AUC']
N = len(cats)
angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]
ax3.set_theta_offset(np.pi / 2); ax3.set_theta_direction(-1)
ax3.set_xticks(angles[:-1]); ax3.set_xticklabels(cats, fontsize=9)
ax3.set_ylim(0.6, 1.0)
for (name, row), color in zip(results_df.iterrows(), COLORS):
    vals = row[['Accuracy','Precision','Recall','F1 Score','ROC-AUC']].tolist()
    vals += vals[:1]
    ax3.plot(angles, vals, 'o-', color=color, lw=1.8, label=name, markersize=4)
    ax3.fill(angles, vals, color=color, alpha=0.05)
ax3.set_title('Radar Chart', fontweight='bold', pad=15)
ax3.legend(loc='upper right', bbox_to_anchor=(1.35, 1.15), fontsize=7)

# ── Subplot 4: RF Confusion Matrix ───────────────────────────────────────
ax4 = fig.add_subplot(2, 3, 4)
y_pred_rf = trained_models['Random Forest'].predict(X_test_sc)
cm = confusion_matrix(y_test, y_pred_rf)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['No Disease','Disease'],
            yticklabels=['No Disease','Disease'],
            cbar=False, linewidths=1)
ax4.set_xlabel('Predicted', fontsize=10); ax4.set_ylabel('True', fontsize=10)
ax4.set_title('RF Confusion Matrix', fontweight='bold')

# ── Subplot 5: Top Feature Importances ───────────────────────────────────
ax5 = fig.add_subplot(2, 3, 5)
top_n = 8
sorted_idx = np.argsort(importances)[-top_n:]
ax5.barh([feature_names[i] for i in sorted_idx],
          importances[sorted_idx],
          color=plt.cm.Blues(np.linspace(0.4, 0.9, top_n)))
ax5.set_xlabel('Importance'); ax5.set_title(f'Top {top_n} Features (RF)', fontweight='bold')
ax5.grid(axis='x', alpha=0.3)

# ── Subplot 6: Metrics Table ──────────────────────────────────────────────
ax6 = fig.add_subplot(2, 3, 6)
ax6.axis('off')
table_data = results_df.reset_index().values
col_labels = ['Model','Acc','Prec','Rec','F1','AUC']
tbl = ax6.table(cellText=[[row[0]]+[f'{v:.2f}' for v in row[1:]] for row in table_data],
                colLabels=col_labels, cellLoc='center', loc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9)
tbl.scale(1.1, 1.6)
# Highlight header and best row
for j in range(len(col_labels)):
    tbl[(0, j)].set_facecolor('#1E40AF'); tbl[(0, j)].set_text_props(color='white', fontweight='bold')
for j in range(len(col_labels)):
    tbl[(4, j)].set_facecolor('#DCFCE7')  # highlight RF row
ax6.set_title('Performance Summary Table', fontweight='bold')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('summary_dashboard.png', dpi=150, bbox_inches='tight', facecolor='#F8FAFC')
plt.show()
print('\n✅ Summary dashboard saved!')

# ---

import pickle, os

# Save best model
with open('best_rf_model.pkl', 'wb') as f:
    pickle.dump(best_rf, f)

# Save scaler
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Save results CSV
results_df.to_csv('model_results.csv')

print('✅ Saved files:')
for fname in ['best_rf_model.pkl', 'scaler.pkl', 'model_results.csv']:
    size = os.path.getsize(fname)
    print(f'   {fname:25s} ({size:,} bytes)')

print('\n🎓 Project Complete!')
print('\nGroup Members:')
for m in ['Kashish Mohammad','Vridhi Vazirani','Jaaswanth Chikkala','Mahadev','Mohith']:
    print(f'   • {m}')
print('\nWoxsen University — B.Tech CSE (Data Science) | 2026')