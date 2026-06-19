"""
Sleep Quality Prediction - Dataset Generation & Model Training
Generates a realistic synthetic sleep dataset and trains multiple ML models.
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, ConfusionMatrixDisplay)
import joblib
import os
import json
import warnings
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# ─────────────────────────────────────────────
# 1. GENERATE REALISTIC SYNTHETIC DATASET
# ─────────────────────────────────────────────
def generate_dataset(n=2000):
    print("📊 Generating synthetic sleep dataset...")

    data = []
    for _ in range(n):
        # Core features with realistic distributions
        sleep_duration    = np.clip(np.random.normal(7.0, 1.5), 3, 12)
        stress_level      = int(np.clip(np.random.normal(5, 2.5), 0, 10))
        exercise_min      = np.clip(np.random.exponential(35), 0, 180)
        screen_time       = np.clip(np.random.exponential(60), 0, 300)
        caffeine          = np.random.choice(['None','Low','Moderate','High'],
                                             p=[0.25, 0.35, 0.25, 0.15])
        mood              = np.random.choice(['Happy','Neutral','Sad','Anxious'],
                                             p=[0.30, 0.35, 0.15, 0.20])
        interruptions     = np.random.choice(['Yes','No'], p=[0.35, 0.65])
        bedtime_hour      = np.clip(np.random.normal(23, 2), 20, 28) % 24  # 8pm-4am range
        wake_hour         = (bedtime_hour + sleep_duration) % 24

        # ── Build sleep score (0-100) from features ──
        score = 50.0

        # Sleep duration contribution (optimal 7-9 hrs)
        if 7 <= sleep_duration <= 9:
            score += 15
        elif 6 <= sleep_duration < 7 or 9 < sleep_duration <= 10:
            score += 5
        else:
            score -= 10

        # Stress contribution
        score -= stress_level * 2.5

        # Exercise contribution
        if exercise_min >= 30:
            score += 12
        elif exercise_min >= 15:
            score += 6
        else:
            score -= 5

        # Screen time contribution
        if screen_time <= 30:
            score += 10
        elif screen_time <= 60:
            score += 3
        elif screen_time <= 120:
            score -= 5
        else:
            score -= 12

        # Caffeine contribution
        caffeine_map = {'None': 8, 'Low': 3, 'Moderate': -5, 'High': -12}
        score += caffeine_map[caffeine]

        # Mood contribution
        mood_map = {'Happy': 8, 'Neutral': 2, 'Sad': -6, 'Anxious': -10}
        score += mood_map[mood]

        # Interruptions contribution
        score += (-12 if interruptions == 'Yes' else 6)

        # Bedtime consistency (closer to midnight = slightly better)
        optimal_bed = 23.0
        bed_diff = abs(bedtime_hour - optimal_bed)
        if bed_diff > 12:
            bed_diff = 24 - bed_diff
        score -= bed_diff * 1.5

        # Add realistic noise
        score += np.random.normal(0, 5)
        score = float(np.clip(score, 0, 100))

        # ── Classify sleep quality ──
        if score >= 65:
            quality = 'Good'
        elif score >= 40:
            quality = 'Average'
        else:
            quality = 'Poor'

        data.append({
            'sleep_duration': round(sleep_duration, 1),
            'bedtime_hour': round(bedtime_hour, 1),
            'wake_hour': round(wake_hour, 1),
            'caffeine_intake': caffeine,
            'exercise_duration': round(exercise_min, 0),
            'screen_time_before_bed': round(screen_time, 0),
            'stress_level': stress_level,
            'mood_before_sleep': mood,
            'sleep_interruptions': interruptions,
            'sleep_score': round(score, 1),
            'sleep_quality': quality
        })

    df = pd.DataFrame(data)
    os.makedirs("data", exist_ok=True)
    df.to_csv("data/sleep_data.csv", index=False)
    print(f"   ✅ Dataset generated: {len(df)} records")
    print(f"   Distribution: {df['sleep_quality'].value_counts().to_dict()}")
    return df


# ─────────────────────────────────────────────
# 2. PREPROCESSING
# ─────────────────────────────────────────────
def preprocess(df):
    print("\n🔧 Preprocessing data...")
    df = df.copy()

    # Drop missing (none expected but just in case)
    df.dropna(inplace=True)

    # Encode categoricals
    caffeine_map = {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3}
    mood_map     = {'Happy': 3, 'Neutral': 2, 'Sad': 1, 'Anxious': 0}
    interrupt_map = {'No': 0, 'Yes': 1}
    quality_map  = {'Poor': 0, 'Average': 1, 'Good': 2}

    df['caffeine_enc']      = df['caffeine_intake'].map(caffeine_map)
    df['mood_enc']          = df['mood_before_sleep'].map(mood_map)
    df['interruptions_enc'] = df['sleep_interruptions'].map(interrupt_map)
    df['quality_enc']       = df['sleep_quality'].map(quality_map)

    feature_cols = [
        'sleep_duration', 'bedtime_hour', 'wake_hour',
        'caffeine_enc', 'exercise_duration', 'screen_time_before_bed',
        'stress_level', 'mood_enc', 'interruptions_enc'
    ]

    X = df[feature_cols]
    y = df['quality_enc']

    # Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_cols)

    print(f"   ✅ Features: {feature_cols}")
    return X_scaled, y, scaler, feature_cols, quality_map


# ─────────────────────────────────────────────
# 3. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────
def train_models(X, y, feature_cols):
    print("\n🤖 Training ML models...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'Decision Tree':       DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
        'SVM':                 SVC(probability=True, random_state=RANDOM_STATE)
    }

    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc    = accuracy_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')

        results[name] = {
            'model':     model,
            'accuracy':  acc,
            'cv_mean':   cv_scores.mean(),
            'cv_std':    cv_scores.std(),
            'y_test':    y_test,
            'y_pred':    y_pred,
            'report':    classification_report(y_test, y_pred,
                                               target_names=['Poor','Average','Good'],
                                               output_dict=True)
        }
        print(f"   {name:22s}  Acc={acc:.4f}  CV={cv_scores.mean():.4f}±{cv_scores.std():.4f}")

    # Pick best model (by CV mean)
    best_name = max(results, key=lambda k: results[k]['cv_mean'])
    print(f"\n   🏆 Best model: {best_name}")

    # Feature importance (Random Forest)
    rf = results['Random Forest']['model']
    importances = pd.Series(rf.feature_importances_, index=feature_cols).sort_values(ascending=False)

    return results, best_name, X_test, y_test, importances


# ─────────────────────────────────────────────
# 4. VISUALIZATIONS (EDA + Model)
# ─────────────────────────────────────────────
def make_eda_plots(df):
    print("\n📈 Generating EDA visualizations...")
    os.makedirs('static/images', exist_ok=True)
    palette = {'Good': '#6C63FF', 'Average': '#48CAE4', 'Poor': '#FF6B6B'}

    # ── Plot 1: Quality distribution
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df['sleep_quality'].value_counts()
    bars = ax.bar(counts.index, counts.values,
                  color=[palette[q] for q in counts.index], width=0.5, edgecolor='white', linewidth=2)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()+10,
                str(val), ha='center', va='bottom', fontweight='bold', color='#333')
    ax.set_title('Sleep Quality Distribution', fontsize=14, fontweight='bold', color='#2d1b69', pad=15)
    ax.set_xlabel('Sleep Quality', color='#555')
    ax.set_ylabel('Count', color='#555')
    ax.tick_params(colors='#555')
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('#f8f9ff')
    ax.set_facecolor('#f8f9ff')
    plt.tight_layout()
    plt.savefig('static/images/quality_dist.png', dpi=100, bbox_inches='tight')
    plt.close()

    # ── Plot 2: Sleep Duration vs Quality (boxplot)
    fig, ax = plt.subplots(figsize=(7, 4))
    order = ['Poor','Average','Good']
    data_plot = [df[df['sleep_quality']==q]['sleep_duration'].values for q in order]
    bp = ax.boxplot(data_plot, patch_artist=True, labels=order,
                    medianprops={'color':'white','linewidth':2})
    for patch, q in zip(bp['boxes'], order):
        patch.set_facecolor(palette[q])
        patch.set_alpha(0.85)
    ax.set_title('Sleep Duration by Quality Category', fontsize=14, fontweight='bold', color='#2d1b69', pad=15)
    ax.set_xlabel('Sleep Quality', color='#555')
    ax.set_ylabel('Sleep Duration (hours)', color='#555')
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('#f8f9ff')
    ax.set_facecolor('#f8f9ff')
    plt.tight_layout()
    plt.savefig('static/images/duration_quality.png', dpi=100, bbox_inches='tight')
    plt.close()

    # ── Plot 3: Stress Level vs Sleep Quality (violin)
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = [palette[q] for q in order]
    parts = ax.violinplot([df[df['sleep_quality']==q]['stress_level'].values for q in order],
                          positions=[1,2,3], showmeans=True, showmedians=False)
    for pc, col in zip(parts['bodies'], colors):
        pc.set_facecolor(col)
        pc.set_alpha(0.8)
    parts['cmeans'].set_color('#333')
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(order)
    ax.set_title('Stress Level by Sleep Quality', fontsize=14, fontweight='bold', color='#2d1b69', pad=15)
    ax.set_xlabel('Sleep Quality', color='#555')
    ax.set_ylabel('Stress Level (0-10)', color='#555')
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('#f8f9ff')
    ax.set_facecolor('#f8f9ff')
    plt.tight_layout()
    plt.savefig('static/images/stress_quality.png', dpi=100, bbox_inches='tight')
    plt.close()

    # ── Plot 4: Correlation heatmap
    num_cols = ['sleep_duration','exercise_duration','screen_time_before_bed',
                'stress_level','sleep_score']
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(7, 5))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='coolwarm',
                ax=ax, square=True, linewidths=1,
                cbar_kws={'shrink': 0.8})
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', color='#2d1b69', pad=15)
    fig.patch.set_facecolor('#f8f9ff')
    plt.tight_layout()
    plt.savefig('static/images/correlation.png', dpi=100, bbox_inches='tight')
    plt.close()

    # ── Plot 5: Screen time vs Sleep Score scatter
    fig, ax = plt.subplots(figsize=(7, 4))
    for q in order:
        subset = df[df['sleep_quality'] == q]
        ax.scatter(subset['screen_time_before_bed'], subset['sleep_score'],
                   alpha=0.4, label=q, color=palette[q], s=20)
    ax.set_xlabel('Screen Time Before Bed (min)', color='#555')
    ax.set_ylabel('Sleep Score', color='#555')
    ax.set_title('Screen Time vs Sleep Score', fontsize=14, fontweight='bold', color='#2d1b69', pad=15)
    ax.legend(title='Quality')
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('#f8f9ff')
    ax.set_facecolor('#f8f9ff')
    plt.tight_layout()
    plt.savefig('static/images/screentime_score.png', dpi=100, bbox_inches='tight')
    plt.close()

    # ── Plot 6: Exercise vs Sleep Score
    fig, ax = plt.subplots(figsize=(7, 4))
    for q in order:
        subset = df[df['sleep_quality'] == q]
        ax.scatter(subset['exercise_duration'], subset['sleep_score'],
                   alpha=0.4, label=q, color=palette[q], s=20)
    ax.set_xlabel('Exercise Duration (min)', color='#555')
    ax.set_ylabel('Sleep Score', color='#555')
    ax.set_title('Exercise Duration vs Sleep Score', fontsize=14, fontweight='bold', color='#2d1b69', pad=15)
    ax.legend(title='Quality')
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('#f8f9ff')
    ax.set_facecolor('#f8f9ff')
    plt.tight_layout()
    plt.savefig('static/images/exercise_score.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("   ✅ EDA plots saved")


def make_model_plots(results, best_name, importances):
    print("\n📊 Generating model performance plots...")

    # ── Accuracy comparison bar chart
    fig, ax = plt.subplots(figsize=(8, 4))
    names  = list(results.keys())
    accs   = [results[n]['accuracy'] for n in names]
    cvs    = [results[n]['cv_mean'] for n in names]
    x = np.arange(len(names))
    w = 0.35
    b1 = ax.bar(x - w/2, accs, w, label='Test Acc', color='#6C63FF', alpha=0.85)
    b2 = ax.bar(x + w/2, cvs,  w, label='CV Mean',  color='#48CAE4', alpha=0.85)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.002,
                    f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha='right')
    ax.set_ylim(0, 1.1)
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold', color='#2d1b69', pad=15)
    ax.legend()
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('#f8f9ff')
    ax.set_facecolor('#f8f9ff')
    plt.tight_layout()
    plt.savefig('static/images/model_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()

    # ── Confusion matrix for best model
    r = results[best_name]
    fig, ax = plt.subplots(figsize=(6, 5))
    cm = confusion_matrix(r['y_test'], r['y_pred'])
    disp = ConfusionMatrixDisplay(cm, display_labels=['Poor','Average','Good'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    ax.set_title(f'Confusion Matrix – {best_name}', fontsize=13, fontweight='bold', color='#2d1b69', pad=15)
    fig.patch.set_facecolor('#f8f9ff')
    plt.tight_layout()
    plt.savefig('static/images/confusion_matrix.png', dpi=100, bbox_inches='tight')
    plt.close()

    # ── Feature importance
    fig, ax = plt.subplots(figsize=(8, 4))
    colors = ['#6C63FF','#8B5CF6','#48CAE4','#06B6D4','#10B981',
              '#F59E0B','#EF4444','#EC4899','#F97316']
    labels_display = {
        'sleep_duration':'Sleep Duration', 'bedtime_hour':'Bedtime Hour',
        'wake_hour':'Wake Hour', 'caffeine_enc':'Caffeine Intake',
        'exercise_duration':'Exercise Duration', 'screen_time_before_bed':'Screen Time',
        'stress_level':'Stress Level', 'mood_enc':'Mood', 'interruptions_enc':'Interruptions'
    }
    imp_display = importances.rename(labels_display)
    bars = ax.barh(imp_display.index[::-1], imp_display.values[::-1],
                   color=colors[:len(imp_display)], edgecolor='white', linewidth=0.5)
    ax.set_title('Feature Importance (Random Forest)', fontsize=14, fontweight='bold', color='#2d1b69', pad=15)
    ax.set_xlabel('Importance Score', color='#555')
    ax.spines[['top','right']].set_visible(False)
    fig.patch.set_facecolor('#f8f9ff')
    ax.set_facecolor('#f8f9ff')
    plt.tight_layout()
    plt.savefig('static/images/feature_importance.png', dpi=100, bbox_inches='tight')
    plt.close()

    print("   ✅ Model plots saved")


# ─────────────────────────────────────────────
# 5. SAVE ARTIFACTS
# ─────────────────────────────────────────────
def save_artifacts(results, best_name, scaler, feature_cols, importances):
    print("\n💾 Saving model artifacts...")
    best_model = results[best_name]['model']
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, 'models/sleep_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')

    meta = {
        'best_model_name': best_name,
        'feature_cols': feature_cols,
        'quality_labels': {0: 'Poor', 1: 'Average', 2: 'Good'},
        'model_results': {
            name: {
                'accuracy': round(r['accuracy'], 4),
                'cv_mean':  round(r['cv_mean'], 4),
                'cv_std':   round(r['cv_std'], 4)
            }
            for name, r in results.items()
        },
        'feature_importances': {k: round(float(v), 4) for k, v in importances.items()}
    }
    with open('models/meta.json', 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"   ✅ Best model saved: {best_name}")
    print(f"   ✅ Scaler saved")
    print(f"   ✅ Metadata saved")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == '__main__':
    print("=" * 55)
    print("  Sleep Quality Predictor — Training Pipeline")
    print("=" * 55)

    df = generate_dataset(2000)
    X, y, scaler, feature_cols, quality_map = preprocess(df)
    results, best_name, X_test, y_test, importances = train_models(X, y, feature_cols)
    make_eda_plots(df)
    make_model_plots(results, best_name, importances)
    save_artifacts(results, best_name, scaler, feature_cols, importances)

    print("\n✅ Training pipeline complete!")
    print(f"   Best model : {best_name}")
    print(f"   Test acc   : {results[best_name]['accuracy']:.4f}")