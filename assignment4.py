import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from scipy import stats
from scipy.stats import ttest_rel
from ucimlrepo import fetch_ucirepo
import warnings
from datetime import datetime

warnings.filterwarnings('ignore')

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
N_CV_FOLDS = 5
N_RUNS = 5
ALPHA = 0.05

plt.rcParams.update({'figure.dpi': 300, 'font.size': 9, 'font.family': 'sans-serif',
                     'lines.linewidth': 1.5, 'savefig.dpi': 300, 'savefig.bbox': 'tight'})

COLORS = {'blue': '#0C5DA5', 'red': '#C8102E', 'green': '#00843D', 
          'orange': '#FF6F00', 'purple': '#6A1B9A'}


def compute_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    se = stats.sem(data)
    ci = se * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean, mean - ci, mean + ci


def paired_ttest_bonferroni(baseline_scores, comparison_dict, alpha=0.05):

    n_comparisons = len(comparison_dict)
    corrected_alpha = alpha / n_comparisons
    
    results = {}
    for name, scores in comparison_dict.items():
        if len(baseline_scores) != len(scores):
            continue
        t_stat, p_value = ttest_rel(baseline_scores, scores)
        results[name] = {
            'p_value': p_value,
            'significant': p_value < corrected_alpha,
            'better': np.mean(scores) > np.mean(baseline_scores)
        }
    return results


class RFAnalyzer:    
    def __init__(self, X, y, name):
        self.name = name
        self.X = X
        self.y = y.values.ravel() if hasattr(y, 'values') else y
        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=RANDOM_STATE, stratify=self.y)
        
        self.cv = StratifiedKFold(n_splits=N_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        self.results = {}
        self.best_params = {}
        
        n_classes = len(np.unique(self.y))
        self.scoring = 'f1' if n_classes == 2 else 'f1_weighted'
        self.f1_average = 'binary' if n_classes == 2 else 'weighted'
        
        print(f"Train: {len(self.X_train)}, Test: {len(self.X_test)}, "
              f"Features: {self.X.shape[1]}, Classes: {n_classes}")
        print(f"Scoring: {self.scoring}")
    
    def exp1_tree_depth(self):
        print(f"\nExp 1: Tree Depth Impact")
        
        depths = [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, None]
        results = {'depth': [], 'cv_mean': [], 'cv_std': [], 'ci_low': [], 'ci_high': [],
                   'train_mean': [], 'cv_scores': []}
        
        all_scores = {}
        for depth in depths:
            cv_scores, train_scores = [], []
            for run in range(N_RUNS):
                rf = RandomForestClassifier(n_estimators=100, max_depth=depth, 
                                           random_state=RANDOM_STATE + run, n_jobs=-1)
                cv = cross_val_score(rf, self.X_train, self.y_train, cv=self.cv, 
                                   scoring=self.scoring)
                cv_scores.extend(cv)
                
                rf.fit(self.X_train, self.y_train)
                y_train_pred = rf.predict(self.X_train)
                train_scores.append(f1_score(self.y_train, y_train_pred, 
                                           average=self.f1_average, zero_division=0))
            
            mean, ci_low, ci_high = compute_ci(cv_scores)
            results['depth'].append(str(depth))
            results['cv_mean'].append(mean)
            results['cv_std'].append(np.std(cv_scores))
            results['ci_low'].append(ci_low)
            results['ci_high'].append(ci_high)
            results['train_mean'].append(np.mean(train_scores))
            results['cv_scores'].append(cv_scores)
            all_scores[str(depth)] = cv_scores
        
        best_idx = np.argmax(results['cv_mean'])
        self.best_params['max_depth'] = depths[best_idx]
        baseline_scores = all_scores[str(depths[best_idx])]
        comparison = {k: v for k, v in all_scores.items() if k != str(depths[best_idx])}
        results['stats'] = paired_ttest_bonferroni(baseline_scores, comparison, ALPHA)
        
        print(f"  Best: depth={depths[best_idx]}, CV F1={results['cv_mean'][best_idx]:.4f}")
        self.results['depth'] = results
        return results
    
    def exp2_depth_trees(self):
        print(f"\nExp 2: Depth × Trees Relationship")
        
        depths = [3, 7, 15, None]
        n_trees = [10, 50, 100, 200]
        results = {'depth': [], 'n_trees': [], 'cv_mean': [], 'cv_std': []}
        
        for depth in depths:
            for n in n_trees:
                scores = []
                for run in range(N_RUNS):
                    rf = RandomForestClassifier(n_estimators=n, max_depth=depth,
                                               random_state=RANDOM_STATE + run, n_jobs=-1)
                    cv = cross_val_score(rf, self.X_train, self.y_train, cv=self.cv, 
                                       scoring=self.scoring)
                    scores.append(cv.mean())
                
                results['depth'].append(str(depth))
                results['n_trees'].append(n)
                results['cv_mean'].append(np.mean(scores))
                results['cv_std'].append(np.std(scores))
        
        self.results['depth_trees'] = results
        return results
    
    def exp3_bag_size(self):
        print(f"\nExp 3: Bag Size Impact")
        
        bag_sizes = [0.3, 0.5, 0.7, 0.9, None] 
        results = {'bag_size': [], 'cv_mean': [], 'cv_std': [], 'ci_low': [], 'ci_high': []}
        
        for bag in bag_sizes:
            scores = []
            for run in range(N_RUNS):
                rf = RandomForestClassifier(n_estimators=100,
                                           max_depth=self.best_params.get('max_depth'),
                                           max_samples=bag,
                                           random_state=RANDOM_STATE + run, n_jobs=-1)
                cv = cross_val_score(rf, self.X_train, self.y_train, cv=self.cv, 
                                   scoring=self.scoring)
                scores.extend(cv)
            
            mean, ci_low, ci_high = compute_ci(scores)
            bag_label = f"{int(bag*100)}%" if bag else "100%"
            results['bag_size'].append(bag_label)
            results['cv_mean'].append(mean)
            results['cv_std'].append(np.std(scores))
            results['ci_low'].append(ci_low)
            results['ci_high'].append(ci_high)
        
        best_idx = np.argmax(results['cv_mean'])
        self.best_params['max_samples'] = bag_sizes[best_idx]
        print(f"  Best: bag_size={results['bag_size'][best_idx]}, CV F1={results['cv_mean'][best_idx]:.4f}")
        
        self.results['bag_size'] = results
        return results
    
    def exp4_features(self):
        print(f"\nExp 4: Feature Selection Impact")
        
        n_feat = self.X.shape[1]
        features = [1, int(np.log2(n_feat)), int(np.sqrt(n_feat)), 
                   int(n_feat * 0.5), n_feat, None]
        features = sorted(list(set([f for f in features if f is not None and f <= n_feat]))) + [None]
        
        results = {'features': [], 'cv_mean': [], 'cv_std': [], 'ci_low': [], 'ci_high': [], 'cv_scores': []}
        all_scores = {}
        
        for feat in features:
            scores = []
            for run in range(N_RUNS):
                rf = RandomForestClassifier(n_estimators=100, 
                                           max_depth=self.best_params.get('max_depth'),
                                           max_features=feat,
                                           random_state=RANDOM_STATE + run, n_jobs=-1)
                cv = cross_val_score(rf, self.X_train, self.y_train, cv=self.cv, 
                                   scoring=self.scoring)
                scores.extend(cv)
            
            mean, ci_low, ci_high = compute_ci(scores)
            results['features'].append(str(feat))
            results['cv_mean'].append(mean)
            results['cv_std'].append(np.std(scores))
            results['ci_low'].append(ci_low)
            results['ci_high'].append(ci_high)
            results['cv_scores'].append(scores)
            all_scores[str(feat)] = scores
        
        best_idx = np.argmax(results['cv_mean'])
        self.best_params['max_features'] = features[best_idx]
        baseline_scores = all_scores[str(features[best_idx])]
        comparison = {k: v for k, v in all_scores.items() if k != str(features[best_idx])}
        results['stats'] = paired_ttest_bonferroni(baseline_scores, comparison, ALPHA)
        
        print(f"  Best: features={features[best_idx]}, CV F1={results['cv_mean'][best_idx]:.4f}")
        self.results['features'] = results
        return results
    
    def exp5_ensemble_size(self):
        print(f"\nExp 5: Ensemble Size Impact")
        
        n_trees = [1, 5, 10, 25, 50, 100, 200, 500]
        results = {'n_trees': [], 'cv_mean': [], 'cv_std': [], 'ci_low': [], 'ci_high': []}
        
        for n in n_trees:
            scores = []
            for run in range(N_RUNS):
                rf = RandomForestClassifier(n_estimators=n,
                                           max_depth=self.best_params.get('max_depth'),
                                           max_samples=self.best_params.get('max_samples'),
                                           max_features=self.best_params.get('max_features'),
                                           random_state=RANDOM_STATE + run, n_jobs=-1)
                cv = cross_val_score(rf, self.X_train, self.y_train, cv=self.cv, 
                                   scoring=self.scoring)
                scores.extend(cv)
            
            mean, ci_low, ci_high = compute_ci(scores)
            results['n_trees'].append(n)
            results['cv_mean'].append(mean)
            results['cv_std'].append(np.std(scores))
            results['ci_low'].append(ci_low)
            results['ci_high'].append(ci_high)
        
        best_idx = np.argmax(results['cv_mean'])
        self.best_params['n_estimators'] = n_trees[best_idx]
        print(f"  Best: n_trees={n_trees[best_idx]}, CV F1={results['cv_mean'][best_idx]:.4f}")
        
        self.results['ensemble'] = results
        return results
    
    def exp6_interaction(self):
        print(f"\nExp 6: Depth × Features Interaction (Underfit/Overfit)")
        
        n_feat = self.X.shape[1]
        depths = [1, 3, 5, 10, 15, None]  
        features = [1, int(np.sqrt(n_feat)), int(n_feat * 0.5), n_feat, None]
        features = sorted(list(set([f for f in features if f is not None]))) + [None]
        
        results = {'depth': [], 'features': [], 'cv_mean': [], 'train_mean': [], 'gap': []}
        
        for depth in depths:
            for feat in features:
                cv_scores, train_scores = [], []
                for run in range(N_RUNS):
                    rf = RandomForestClassifier(n_estimators=100, max_depth=depth, max_features=feat,
                                               random_state=RANDOM_STATE + run, n_jobs=-1)
                    cv = cross_val_score(rf, self.X_train, self.y_train, cv=self.cv, 
                                       scoring=self.scoring)
                    cv_scores.append(cv.mean())
                    
                    rf.fit(self.X_train, self.y_train)
                    y_train_pred = rf.predict(self.X_train)
                    train_scores.append(f1_score(self.y_train, y_train_pred, 
                                               average=self.f1_average, zero_division=0))
                
                results['depth'].append(str(depth))
                results['features'].append(str(feat))
                results['cv_mean'].append(np.mean(cv_scores))
                results['train_mean'].append(np.mean(train_scores))
                results['gap'].append(np.mean(train_scores) - np.mean(cv_scores))
        
        self.results['interaction'] = results
        return results
    
    def final_evaluation(self):
        print(f"\nFinal Evaluation")
        
        metrics = []
        for run in range(N_RUNS):
            rf = RandomForestClassifier(
                n_estimators=self.best_params.get('n_estimators', 100),
                max_depth=self.best_params.get('max_depth'),
                max_features=self.best_params.get('max_features'),
                random_state=RANDOM_STATE + run, n_jobs=-1)
            
            rf.fit(self.X_train, self.y_train)
            y_pred = rf.predict(self.X_test)
            
            metrics.append({
                'accuracy': accuracy_score(self.y_test, y_pred),
                'precision': precision_score(self.y_test, y_pred, average=self.f1_average, zero_division=0),
                'recall': recall_score(self.y_test, y_pred, average=self.f1_average, zero_division=0),
                'f1': f1_score(self.y_test, y_pred, average=self.f1_average, zero_division=0)
            })
        
        results = {k: (np.mean([m[k] for m in metrics]), np.std([m[k] for m in metrics])) 
                  for k in metrics[0].keys()}
        
        print(f"  F1-Score: {results['f1'][0]:.4f} ± {results['f1'][1]:.4f}")
        print(f"  Accuracy: {results['accuracy'][0]:.4f} ± {results['accuracy'][1]:.4f}")
        print(f"  Precision: {results['precision'][0]:.4f} ± {results['precision'][1]:.4f}")
        print(f"  Recall: {results['recall'][0]:.4f} ± {results['recall'][1]:.4f}")
        
        self.results['final'] = results
        return results
    
    def run_all(self):
        self.exp1_tree_depth()
        self.exp2_depth_trees()
        self.exp3_bag_size()
        self.exp4_features()
        self.exp5_ensemble_size()
        self.exp6_interaction()
        self.final_evaluation()
        return self


def load_breast_cancer():
    try:
        df = pd.read_csv('breastCancer.csv', sep='\t')
        df.columns = df.columns.str.strip('"')
        X = df.drop(['diagnosis', 'id'] if 'id' in df.columns else ['diagnosis'], axis=1)
        y = LabelEncoder().fit_transform(df['diagnosis'])
        for col in X.select_dtypes(exclude=[np.number]).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        return X, y
    except:
        from sklearn.datasets import load_breast_cancer as load_bc
        data = load_bc()
        return pd.DataFrame(data.data, columns=data.feature_names), data.target


def load_diabetes():
    print("Loading CDC Diabetes Health Indicators dataset...")
    cdc_diabetes = fetch_ucirepo(id=891)
    X = cdc_diabetes.data.features
    y = LabelEncoder().fit_transform(cdc_diabetes.data.targets.values.ravel())
    
    from collections import Counter
    class_counts = Counter(y)
    print(f"  Original class distribution: {class_counts}")
    
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    minority_size = class_counts[minority_class]
    
    minority_indices = np.where(y == minority_class)[0]
    majority_indices = np.where(y == majority_class)[0]
    
    np.random.seed(RANDOM_STATE)
    majority_indices_sampled = np.random.choice(majority_indices, 
                                                size=minority_size, 
                                                replace=False)
    
    balanced_indices = np.concatenate([minority_indices, majority_indices_sampled])
    np.random.shuffle(balanced_indices)
    
    X_balanced = X.iloc[balanced_indices]
    y_balanced = y[balanced_indices]
    
    print(f"  Balanced dataset: {len(X_balanced)} samples "
          f"({Counter(y_balanced)[0]} class 0, {Counter(y_balanced)[1]} class 1)")
    
    return X_balanced, y_balanced


def load_letter():
    letter = fetch_ucirepo(id=59)
    return letter.data.features, LabelEncoder().fit_transform(letter.data.targets.values.ravel())

def save_plots_png(analyzers):
    
    print("\nGenerating PNG plots...")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for idx, a in enumerate(analyzers):
        ax = axes[idx]
        res = a.results['depth']
        x = np.arange(len(res['depth']))
        ax.plot(x, res['cv_mean'], 's-', color=COLORS['blue'], linewidth=2, markersize=5)
        ax.fill_between(x, res['ci_low'], res['ci_high'], alpha=0.2, color=COLORS['blue'])
        best_idx = np.argmax(res['cv_mean'])
        ax.scatter(best_idx, res['cv_mean'][best_idx], s=100, c=COLORS['red'], 
                  marker='*', zorder=5, edgecolors='black')
        ax.set_xlabel('Max Depth', fontweight='bold')
        ax.set_ylabel('CV F1 Score', fontweight='bold')
        ax.set_title(f'{a.name}', fontweight='bold')
        ax.set_xticks(x[::2])
        ax.set_xticklabels([res['depth'][i] for i in range(0, len(res['depth']), 2)], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Experiment 1: Tree Depth Impact', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot1_tree_depth.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ plot1_tree_depth.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for idx, a in enumerate(analyzers):
        ax = axes[idx]
        res = a.results['depth']
        x = np.arange(len(res['depth']))
        gap = np.array(res['train_mean']) - np.array(res['cv_mean'])
        colors_list = [COLORS['red'] if g > 0.05 else COLORS['orange'] if g > 0.02 
                      else COLORS['green'] for g in gap]
        ax.bar(x, gap, color=colors_list, alpha=0.7, edgecolor='black', linewidth=0.5)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        ax.set_xlabel('Max Depth', fontweight='bold')
        ax.set_ylabel('Train-CV Gap (F1)', fontweight='bold')
        ax.set_title(f'{a.name}', fontweight='bold')
        ax.set_xticks(x[::2])
        ax.set_xticklabels([res['depth'][i] for i in range(0, len(res['depth']), 2)], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3, axis='y')
    fig.suptitle('Overfitting Analysis by Depth', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot2_overfitting.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ plot2_overfitting.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for idx, a in enumerate(analyzers):
        ax = axes[idx]
        res = a.results['depth_trees']
        depths = sorted(list(set(res['depth'])), key=lambda x: float('inf') if x == 'None' else int(x))
        colors_list = [COLORS['blue'], COLORS['red'], COLORS['green'], COLORS['orange']]
        for d_idx, depth in enumerate(depths):
            mask = [d == depth for d in res['depth']]
            trees = [res['n_trees'][i] for i, m in enumerate(mask) if m]
            means = [res['cv_mean'][i] for i, m in enumerate(mask) if m]
            ax.plot(trees, means, 'o-', label=f'd={depth}', 
                   color=colors_list[d_idx % len(colors_list)], linewidth=2, markersize=5)
        ax.set_xlabel('Number of Trees', fontweight='bold')
        ax.set_ylabel('CV F1 Score', fontweight='bold')
        ax.set_title(f'{a.name}', fontweight='bold')
        ax.set_xscale('log')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Experiment 2: Depth × Trees Relationship', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot3_depth_trees.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ plot3_depth_trees.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for idx, a in enumerate(analyzers):
        ax = axes[idx]
        res = a.results['bag_size']
        x = np.arange(len(res['bag_size']))
        ax.plot(x, res['cv_mean'], 's-', color=COLORS['purple'], linewidth=2, markersize=5)
        ax.fill_between(x, res['ci_low'], res['ci_high'], alpha=0.2, color=COLORS['purple'])
        best_idx = np.argmax(res['cv_mean'])
        ax.scatter(best_idx, res['cv_mean'][best_idx], s=100, c=COLORS['red'], 
                  marker='*', zorder=5, edgecolors='black')
        ax.set_xlabel('Bag Size (Bootstrap %)', fontweight='bold')
        ax.set_ylabel('CV F1 Score', fontweight='bold')
        ax.set_title(f'{a.name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(res['bag_size'], rotation=45, fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Experiment 3: Bag Size Impact', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot4_bag_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ plot4_bag_size.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for idx, a in enumerate(analyzers):
        ax = axes[idx]
        res = a.results['features']
        x = np.arange(len(res['features']))
        ax.plot(x, res['cv_mean'], 's-', color=COLORS['blue'], linewidth=2, markersize=5)
        ax.fill_between(x, res['ci_low'], res['ci_high'], alpha=0.2, color=COLORS['blue'])
        best_idx = np.argmax(res['cv_mean'])
        ax.scatter(best_idx, res['cv_mean'][best_idx], s=100, c=COLORS['red'], 
                  marker='*', zorder=5, edgecolors='black')
        ax.set_xlabel('Max Features', fontweight='bold')
        ax.set_ylabel('CV F1 Score', fontweight='bold')
        ax.set_title(f'{a.name}', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(res['features'], rotation=45, fontsize=7)
        ax.grid(True, alpha=0.3)
    fig.suptitle('Experiment 4: Feature Selection Impact', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot5_feature_selection.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ plot5_feature_selection.png")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for idx, a in enumerate(analyzers):
        ax = axes[idx]
        res = a.results['ensemble']
        ax.plot(res['n_trees'], res['cv_mean'], 's-', color=COLORS['blue'], linewidth=2, markersize=5)
        ax.fill_between(res['n_trees'], res['ci_low'], res['ci_high'], alpha=0.2, color=COLORS['blue'])
        best_idx = np.argmax(res['cv_mean'])
        ax.scatter(res['n_trees'][best_idx], res['cv_mean'][best_idx], 
                  s=100, c=COLORS['red'], marker='*', zorder=5, edgecolors='black')
        ax.set_xlabel('Number of Trees', fontweight='bold')
        ax.set_ylabel('CV F1 Score', fontweight='bold')
        ax.set_title(f'{a.name}', fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
    fig.suptitle('Experiment 5: Ensemble Size Impact', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plot6_ensemble_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ plot6_ensemble_size.png")
    
    for plot_num, a in enumerate(analyzers, start=7):
        res = a.results['interaction']
        depths = sorted(list(set(res['depth'])), key=lambda x: float('inf') if x == 'None' else int(x))
        features = sorted(list(set(res['features'])), key=lambda x: float('inf') if x == 'None' else int(x))
        
        acc_matrix = np.zeros((len(depths), len(features)))
        gap_matrix = np.zeros((len(depths), len(features)))
        
        for i, d in enumerate(depths):
            for j, f in enumerate(features):
                mask = (np.array(res['depth']) == d) & (np.array(res['features']) == f)
                if mask.any():
                    idx = np.where(mask)[0][0]
                    acc_matrix[i, j] = res['cv_mean'][idx]
                    gap_matrix[i, j] = res['gap'][idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        ax1 = axes[0]
        im1 = ax1.imshow(acc_matrix, cmap='RdYlGn', aspect='auto')
        ax1.set_xticks(np.arange(len(features)))
        ax1.set_yticks(np.arange(len(depths)))
        ax1.set_xticklabels(features, fontsize=8)
        ax1.set_yticklabels(depths, fontsize=8)
        ax1.set_xlabel('Max Features', fontweight='bold')
        ax1.set_ylabel('Max Depth', fontweight='bold')
        ax1.set_title('CV Test F1 Score', fontweight='bold')
        plt.colorbar(im1, ax=ax1, fraction=0.046)
        
        ax2 = axes[1]
        im2 = ax2.imshow(gap_matrix, cmap='RdYlGn_r', aspect='auto', vmin=0)
        ax2.set_xticks(np.arange(len(features)))
        ax2.set_yticks(np.arange(len(depths)))
        ax2.set_xticklabels(features, fontsize=8)
        ax2.set_yticklabels(depths, fontsize=8)
        ax2.set_xlabel('Max Features', fontweight='bold')
        ax2.set_ylabel('Max Depth', fontweight='bold')
        ax2.set_title('Overfitting Gap (F1)', fontweight='bold')
        plt.colorbar(im2, ax=ax2, fraction=0.046)
        
        fig.suptitle(f'Experiment 6: {a.name} - Depth × Features Interaction', 
                    fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'plot{plot_num}_interaction_{a.name.replace(" ", "_").lower()}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ plot{plot_num}_interaction_{a.name.replace(' ', '_').lower()}.png")
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    
    names = [a.name for a in analyzers]
    
    ax1 = axes[0]
    f1s = [a.results['final']['f1'][0] for a in analyzers]
    stds = [a.results['final']['f1'][1] for a in analyzers]
    bars = ax1.bar(range(len(names)), f1s, yerr=stds, capsize=5, 
                  color=[COLORS['blue'], COLORS['red'], COLORS['green']], alpha=0.7,
                  edgecolor='black', linewidth=0.8)
    ax1.set_ylabel('Test F1 Score', fontweight='bold')
    ax1.set_title('Final Model Performance', fontweight='bold')
    ax1.set_xticks(range(len(names)))
    ax1.set_xticklabels(names, rotation=15, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    for bar, f1 in zip(bars, f1s):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{f1:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax2 = axes[1]
    x = np.arange(len(names))
    width = 0.25
    f1_scores = [a.results['final']['f1'][0] for a in analyzers]
    precisions = [a.results['final']['precision'][0] for a in analyzers]
    recalls = [a.results['final']['recall'][0] for a in analyzers]
    ax2.bar(x - width, precisions, width, label='Precision', color=COLORS['blue'], 
           alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.bar(x, recalls, width, label='Recall', color=COLORS['green'], 
           alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.bar(x + width, f1_scores, width, label='F1', color=COLORS['orange'], 
           alpha=0.7, edgecolor='black', linewidth=0.5)
    ax2.set_ylabel('Score', fontweight='bold')
    ax2.set_title('Performance Metrics', fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=15, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('plot10_performance_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  ✓ plot10_performance_comparison.png")


def save_results_txt(analyzers):    
    with open('results.txt', 'w') as f:
        f.write("RANDOM FOREST HYPERPARAMETER ANALYSIS - COMPLETE RESULTS\n")
        f.write("Primary Metric: F1 Score\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("STATISTICAL VALIDATION METHODOLOGY\n")
        f.write(f"Cross-Validation: {N_CV_FOLDS}-fold Stratified K-Fold\n")
        f.write(f"Independent Runs: {N_RUNS} per configuration\n")
        f.write(f"Statistical Testing: Paired t-tests with Bonferroni correction\n")
        f.write(f"Significance Level: alpha = {ALPHA}\n")
        f.write(f"Confidence Intervals: 95 percent\n")
        f.write(f"Scoring: F1 (binary) or F1-weighted (multiclass)\n\n")
        
        for a in analyzers:
            f.write(f"{'='*70}\n")
            f.write(f"DATASET: {a.name}\n")
            f.write(f"{'='*70}\n\n")
            
            f.write("Dataset Statistics:\n")
            f.write(f"  Training Samples: {len(a.X_train)}\n")
            f.write(f"  Testing Samples: {len(a.X_test)}\n")
            f.write(f"  Number of Features: {a.X.shape[1]}\n")
            f.write(f"  Number of Classes: {len(np.unique(a.y))}\n")
            f.write(f"  Scoring Method: {a.scoring}\n\n")
            
            f.write("OPTIMAL HYPERPARAMETERS:\n")
            f.write(f"  Best Max Depth: {a.best_params.get('max_depth')}\n")
            f.write(f"  Best Bag Size: {a.best_params.get('max_samples')}\n")
            f.write(f"  Best Max Features: {a.best_params.get('max_features')}\n")
            f.write(f"  Best n_estimators: {a.best_params.get('n_estimators')}\n\n")
            
            f.write("FINAL MODEL PERFORMANCE:\n")
            final = a.results['final']
            f.write(f"  Test F1-Score:  {final['f1'][0]:.4f} +/- {final['f1'][1]:.4f}\n")
            f.write(f"  Test Accuracy:  {final['accuracy'][0]:.4f} +/- {final['accuracy'][1]:.4f}\n")
            f.write(f"  Precision:      {final['precision'][0]:.4f} +/- {final['precision'][1]:.4f}\n")
            f.write(f"  Recall:         {final['recall'][0]:.4f} +/- {final['recall'][1]:.4f}\n\n")
            
            f.write("EXPERIMENT 1: TREE DEPTH IMPACT\n")
            res = a.results['depth']
            f.write(f"{'Depth':<10} {'CV F1':<12} {'CV Std':<12} {'95% CI Lower':<15} {'95% CI Upper':<15} {'Train F1':<12}\n")
            for i in range(len(res['depth'])):
                f.write(f"{res['depth'][i]:<10} {res['cv_mean'][i]:<12.4f} {res['cv_std'][i]:<12.4f} "
                       f"{res['ci_low'][i]:<15.4f} {res['ci_high'][i]:<15.4f} {res['train_mean'][i]:<12.4f}\n")
            f.write(f"\nBest Depth: {a.best_params.get('max_depth')}\n")
            f.write(f"Best CV F1: {max(res['cv_mean']):.4f}\n\n")
            
            if 'stats' in res:
                f.write("Statistical Testing (Paired t-tests with Bonferroni correction):\n")
                f.write(f"Baseline: depth={a.best_params.get('max_depth')}\n")
                f.write(f"{'Comparison':<15} {'p-value':<15} {'Significant':<15} {'Better':<10}\n")
                for depth, stats_dict in res['stats'].items():
                    f.write(f"{depth:<15} {stats_dict['p_value']:<15.6f} "
                           f"{'Yes' if stats_dict['significant'] else 'No':<15} "
                           f"{'Yes' if stats_dict['better'] else 'No':<10}\n")
                f.write("\n")
            
            f.write("EXPERIMENT 2: DEPTH × TREES RELATIONSHIP\n")
            res = a.results['depth_trees']
            f.write(f"{'Depth':<10} {'n_Trees':<12} {'CV F1':<12} {'CV Std':<12}\n")
            for i in range(len(res['depth'])):
                f.write(f"{res['depth'][i]:<10} {res['n_trees'][i]:<12} "
                       f"{res['cv_mean'][i]:<12.4f} {res['cv_std'][i]:<12.4f}\n")
            f.write("\n")
            
            f.write("EXPERIMENT 3: BAG SIZE IMPACT\n")
            res = a.results['bag_size']
            f.write(f"{'Bag Size':<15} {'CV F1':<12} {'CV Std':<12} {'95% CI Lower':<15} {'95% CI Upper':<15}\n")
            for i in range(len(res['bag_size'])):
                f.write(f"{res['bag_size'][i]:<15} {res['cv_mean'][i]:<12.4f} {res['cv_std'][i]:<12.4f} "
                       f"{res['ci_low'][i]:<15.4f} {res['ci_high'][i]:<15.4f}\n")
            f.write(f"\nBest Bag Size: {a.best_params.get('max_samples')}\n")
            f.write(f"Best CV F1: {max(res['cv_mean']):.4f}\n\n")
            
            f.write("EXPERIMENT 4: FEATURE SELECTION IMPACT\n")
            res = a.results['features']
            f.write(f"{'Features':<12} {'CV F1':<12} {'CV Std':<12} {'95% CI Lower':<15} {'95% CI Upper':<15}\n")
            for i in range(len(res['features'])):
                f.write(f"{res['features'][i]:<12} {res['cv_mean'][i]:<12.4f} {res['cv_std'][i]:<12.4f} "
                       f"{res['ci_low'][i]:<15.4f} {res['ci_high'][i]:<15.4f}\n")
            f.write(f"\nBest Features: {a.best_params.get('max_features')}\n")
            f.write(f"Best CV F1: {max(res['cv_mean']):.4f}\n\n")
            
            if 'stats' in res:
                f.write("Statistical Testing (Paired t-tests with Bonferroni correction):\n")
                f.write(f"Baseline: features={a.best_params.get('max_features')}\n")
                f.write(f"{'Comparison':<15} {'p-value':<15} {'Significant':<15} {'Better':<10}\n")
                for feat, stats_dict in res['stats'].items():
                    f.write(f"{feat:<15} {stats_dict['p_value']:<15.6f} "
                           f"{'Yes' if stats_dict['significant'] else 'No':<15} "
                           f"{'Yes' if stats_dict['better'] else 'No':<10}\n")
                f.write("\n")
            
            f.write("EXPERIMENT 5: ENSEMBLE SIZE IMPACT\n")
            res = a.results['ensemble']
            f.write(f"{'n_Trees':<12} {'CV F1':<12} {'CV Std':<12} {'95% CI Lower':<15} {'95% CI Upper':<15}\n")
            for i in range(len(res['n_trees'])):
                f.write(f"{res['n_trees'][i]:<12} {res['cv_mean'][i]:<12.4f} {res['cv_std'][i]:<12.4f} "
                       f"{res['ci_low'][i]:<15.4f} {res['ci_high'][i]:<15.4f}\n")
            f.write(f"\nBest n_estimators: {a.best_params.get('n_estimators')}\n")
            f.write(f"Best CV F1: {max(res['cv_mean']):.4f}\n\n")
            
            f.write("EXPERIMENT 6: DEPTH × FEATURES INTERACTION (Underfit/Overfit Analysis)\n")
            res = a.results['interaction']
            f.write(f"{'Depth':<10} {'Features':<12} {'CV F1':<12} {'Train F1':<12} {'Gap':<12}\n")
            for i in range(len(res['depth'])):
                f.write(f"{res['depth'][i]:<10} {res['features'][i]:<12} "
                       f"{res['cv_mean'][i]:<12.4f} {res['train_mean'][i]:<12.4f} "
                       f"{res['gap'][i]:<12.4f}\n")
            f.write("\n")
        
        f.write("="*70 + "\n")
        f.write("CROSS-DATASET SUMMARY\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"{'Dataset':<25} {'Samples':<10} {'Features':<10} {'Best Depth':<12} "
               f"{'Best Bag':<12} {'Best Feat':<12} {'Best Trees':<12} {'Test F1':<10}\n")
        f.write("-"*110 + "\n")
        
        for a in analyzers:
            bag = a.best_params.get('max_samples')
            bag_str = f"{bag*100:.0f}%" if bag else "100%"
            f.write(f"{a.name:<25} {len(a.y):<10} {a.X.shape[1]:<10} "
                   f"{str(a.best_params.get('max_depth')):<12} "
                   f"{bag_str:<12} {str(a.best_params.get('max_features')):<12} "
                   f"{a.best_params.get('n_estimators'):<12} "
                   f"{a.results['final']['f1'][0]:.4f}\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("END OF REPORT\n")
    


def save_results_csv(analyzers):
    data = []
    for a in analyzers:
        bag = a.best_params.get('max_samples')
        bag_str = f"{bag*100:.0f}%" if bag else "100%"
        data.append({
            'Dataset': a.name,
            'Best_Depth': a.best_params.get('max_depth'),
            'Best_Bag_Size': bag_str,
            'Best_Features': a.best_params.get('max_features'),
            'Best_Trees': a.best_params.get('n_estimators'),
            'Test_F1': f"{a.results['final']['f1'][0]:.4f}",
            'Test_F1_Std': f"{a.results['final']['f1'][1]:.4f}",
            'Test_Accuracy': f"{a.results['final']['accuracy'][0]:.4f}",
            'Precision': f"{a.results['final']['precision'][0]:.4f}",
            'Recall': f"{a.results['final']['recall'][0]:.4f}"
        })
    
    df = pd.DataFrame(data)
    df.to_csv('results.csv', index=False)



def main():
    
    X1, y1 = load_breast_cancer()
    X2, y2 = load_diabetes()
    X3, y3 = load_letter()
    
    a1 = RFAnalyzer(X1, y1, "Breast Cancer").run_all()
    a2 = RFAnalyzer(X2, y2, "CDC Diabetes").run_all()
    a3 = RFAnalyzer(X3, y3, "Letter Recognition").run_all()
    
    analyzers = [a1, a2, a3]
    
    print("GENERATING OUTPUT FILES")
    
    save_plots_png(analyzers)
    save_results_txt(analyzers)
    save_results_csv(analyzers)
    
    print("ANALYSIS COMPLETE")
   
    return analyzers


if __name__ == "__main__":
    results = main()
