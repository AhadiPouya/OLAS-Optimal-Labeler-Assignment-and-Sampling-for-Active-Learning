"""
=============================================================================
Alpha Sensitivity Analysis -- Standalone Runner
=============================================================================
Purpose : Test whether method rankings are stable across alpha values.
Datasets : Heart Statlog and Spambase only (one small, one medium)
Methods  : All 8 (RS+RLA, RS+OLA, ES+RLA, ES+OLA, OLAS, MV, KC+RLA, BADGE+RLA)
Noise    : NM1 only
Alpha    : {0.15, 0.20, 0.25}
Beta     : 0.20 (fixed, same as main experiments)
Reps     : 100 per alpha value per dataset

Note: alpha=0.20 is your default. Running all three fresh here keeps
results consistent (same random seed structure across all alpha values).

Output files saved to OLAS_Results/Results/Sensitivity/
    Heart_Statlog_NM1_alpha015.xlsx
    Heart_Statlog_NM1_alpha020.xlsx
    Heart_Statlog_NM1_alpha025.xlsx
    Spambase_NM1_alpha015.xlsx
    Spambase_NM1_alpha020.xlsx
    Spambase_NM1_alpha025.xlsx
    Sensitivity_Summary.xlsx   <-- final summary table for the paper
=============================================================================
"""

# =============================================================================
# STEP 1 -- Mount Drive
# =============================================================================
# If running in Google Colab, mount Drive first (comment out if running locally):
# from google.colab import drive
# drive.mount('/content/drive')

# =============================================================================
# STEP 2 -- Imports
# =============================================================================
import os
import copy
import time
import random
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split

from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling

warnings.filterwarnings('ignore', category=FutureWarning)

def _append_compat(self, other, ignore_index=False):
    if isinstance(other, dict):
        other = pd.DataFrame([other])
    return pd.concat([self, other], ignore_index=ignore_index)
pd.DataFrame.append = _append_compat

# =============================================================================
# STEP 3 -- Paths
# =============================================================================
DRIVE_BASE    = os.environ.get('OLAS_BASE_DIR', 'OLAS_Results')  # set this to your project folder
DRIVE_DATA    = os.path.join(DRIVE_BASE, 'Data')
DRIVE_RESULTS = os.path.join(DRIVE_BASE, 'Results', 'Sensitivity')
os.makedirs(DRIVE_RESULTS, exist_ok=True)

def data_path(f):   return os.path.join(DRIVE_DATA,    f)
def result_path(f): return os.path.join(DRIVE_RESULTS, f)

# =============================================================================
# STEP 4 -- Parameters
# =============================================================================
BETA          = 0.20
N_CYCLES      = 10
INIT_NUMBER   = 3
REPLICATIONS  = 50
ALPHA_VALUES  = [0.15, 0.20, 0.25]
DATASETS      = ['Heart Statlog', 'Spambase']

METHODS = ['RS+RLA', 'RS+OLA', 'ES+RLA', 'ES+OLA',
           'OLAS', 'MV', 'KC+RLA', 'BADGE+RLA']

RF_PARAMS = dict(random_state=0, n_jobs=-1, n_estimators=50)

# Module-level globals -- set per dataset / alpha combination
alpha      = None
beta       = BETA
n_cycles   = N_CYCLES
epsilon    = None
A          = None
A_boosted  = None
M          = None
Cap        = None
budget     = None
class_list = None

# =============================================================================
# STEP 5 -- Noise model (NM1 only)
# =============================================================================

def make_epsilon_nm1():
    def _eps(a, e):
        e = float(np.clip(e, 0.0, 1.0))
        return e * (1.0 - a)
    return _eps

# =============================================================================
# STEP 6 -- BADGE embedding and k-means++ selection
# =============================================================================

def badge_embedding(X_pool, model):
    probs = model.predict_proba(X_pool)
    yhat  = np.argmax(probs, axis=1)
    n, K  = probs.shape
    d     = X_pool.shape[1]
    embeddings = np.zeros((n, K * d), dtype=np.float32)
    for i in range(n):
        diff           = probs[i].copy()
        diff[yhat[i]] -= 1.0
        embeddings[i]  = np.outer(diff, X_pool[i]).flatten()
    return embeddings


def kmeans_pp_select(embeddings, k):
    n = len(embeddings)
    k = min(k, n)
    idx      = np.random.randint(0, n)
    selected = [idx]
    min_sq   = np.sum((embeddings - embeddings[idx]) ** 2, axis=1)
    for _ in range(1, k):
        total = min_sq.sum()
        if total == 0:
            remaining = [i for i in range(n) if i not in set(selected)]
            if not remaining:
                break
            idx = np.random.choice(remaining)
        else:
            idx = np.random.choice(n, p=min_sq / total)
        selected.append(idx)
        new_sq = np.sum((embeddings - embeddings[idx]) ** 2, axis=1)
        min_sq = np.minimum(min_sq, new_sq)
        min_sq[idx] = 0.0
    return selected


def badge_select(X_pool, model, k):
    return kmeans_pp_select(badge_embedding(X_pool, model), k)

# =============================================================================
# STEP 7 -- k-Center selection
# =============================================================================

def kcenter_select(X_pool_norm, kc_budget, X_labeled_norm):
    n_pool    = len(X_pool_norm)
    kc_budget = min(kc_budget, n_pool)
    if len(X_labeled_norm) > 0:
        nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', n_jobs=-1)
        nn.fit(X_labeled_norm)
        min_dists, _ = nn.kneighbors(X_pool_norm)
        min_dists    = min_dists.flatten().copy()
    else:
        min_dists = np.full(n_pool, np.inf)
    selected = []
    for _ in range(kc_budget):
        idx            = int(np.argmax(min_dists))
        selected.append(idx)
        new_d          = np.linalg.norm(X_pool_norm - X_pool_norm[idx], axis=1)
        min_dists      = np.minimum(min_dists, new_d)
        min_dists[idx] = -np.inf
    return selected

# =============================================================================
# STEP 8 -- Labeler assignment functions
# =============================================================================

def noisy_RLA(y_true, model_entropy_query):
    Acc         = copy.copy(A_boosted)
    n           = len(y_true)
    y_noisy     = []
    total_flip  = 0
    total_noise = 0.0
    for i in range(n):
        rand_idx = np.random.randint(0, len(Acc))
        acc      = Acc.pop(rand_idx)
        noise    = epsilon(acc, model_entropy_query[i])
        total_noise += noise
        if alpha >= noise:
            y_noisy.append(y_true[i])
        else:
            choices = [c for c in class_list if c != y_true[i]]
            y_noisy.append(np.random.choice(choices))
            total_flip += 1
    return y_noisy, total_flip / n, total_noise / n


def noisy_OLA(y_true, model_entropy_query):
    Acc        = copy.copy(A_boosted)
    n          = len(y_true)
    ent_sorted = np.sort(model_entropy_query).tolist()
    len_diff   = len(Acc) - n
    y_noisy     = []
    total_flip  = 0
    total_noise = 0.0
    for i in range(n):
        idx  = ent_sorted.index(model_entropy_query[i])
        ent_sorted.pop(idx)
        acc  = Acc.pop(idx + len_diff)
        noise = epsilon(acc, model_entropy_query[i])
        total_noise += noise
        if alpha >= noise:
            y_noisy.append(y_true[i])
        else:
            choices = [c for c in class_list if c != y_true[i]]
            y_noisy.append(np.random.choice(choices))
            total_flip += 1
    return y_noisy, total_flip / n, total_noise / n


def OA(U, U_entropy, y, beta_param):
    df = pd.DataFrame({'idx': U, 'ent': U_entropy})
    df = df.sort_values('ent', ascending=False).reset_index(drop=True)
    A_list      = sorted(A, reverse=True)
    query_idx   = []
    y_noisy     = []
    total_flip  = 0
    total_noise = 0.0
    last_query  = -1
    m_idx       = -1
    flag        = 1
    while m_idx < M - 1 and flag == 1:
        m_idx += 1
        acc    = A_list[m_idx]
        for i in range(last_query + 1, len(U)):
            if epsilon(acc, df.iloc[i, 1]) <= beta_param:
                last_query = min(i + Cap - 1, len(U) - 1)
                if last_query == len(U) - 1:
                    flag = 0
                for row in range(i, min(i + Cap, len(U))):
                    g_idx       = int(df.iloc[row, 0])
                    query_idx.append(g_idx)
                    noise       = epsilon(acc, df.iloc[row, 1])
                    total_noise += noise
                    true_label  = y[g_idx]
                    if alpha >= noise:
                        y_noisy.append(true_label)
                    else:
                        choices = [c for c in class_list if c != true_label]
                        y_noisy.append(np.random.choice(choices))
                        total_flip += 1
                break
    if len(query_idx) == 0:
        return [], [], 0.0, 0.0
    return (query_idx, y_noisy,
            total_flip  / len(query_idx),
            total_noise / len(query_idx))


def noisy_MV(y_true, model_entropy_query):
    n          = len(y_true)
    y_noisy    = []
    total_flip = 0
    for i in range(n):
        votes = []
        for j in range(M):
            acc   = A[j]
            noise = epsilon(acc, model_entropy_query[i])
            if alpha >= noise:
                votes.append(y_true[i])
            else:
                choices = [c for c in class_list if c != y_true[i]]
                votes.append(np.random.choice(choices))
        counts     = Counter(votes)
        max_count  = max(counts.values())
        candidates = [c for c, cnt in counts.items() if cnt == max_count]
        final      = np.random.choice(candidates)
        y_noisy.append(final)
        if final != y_true[i]:
            total_flip += 1
    return y_noisy, total_flip / n

# =============================================================================
# STEP 9 -- Data loading
# =============================================================================

def data_read(dataset_name):
    if dataset_name == 'Heart Statlog':
        df  = pd.read_csv(data_path('heart.dat'), header=None, delimiter=' ')
        arr = df.values
        return arr[:, :13].astype(float), arr[:, 13].astype(int) - 1
    elif dataset_name == 'Spambase':
        df  = pd.read_csv(data_path('spambase.data'), header=None)
        arr = df.values
        return arr[:, :57].astype(float), arr[:, 57].astype(int)
    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

# =============================================================================
# STEP 10 -- Single replication: all 8 methods
# =============================================================================

def run_one_rep(I, L_init, U_init):
    """
    Run all 8 methods for one replication.
    Returns dict of F1 scores keyed by method name.
    """
    F1s = {}

    # ---- RS+RLA ----
    learner = ActiveLearner(
        estimator=RandomForestClassifier(**RF_PARAMS),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init], y_training=y_train[L_init])
    L = list(L_init); U = list(U_init)
    for _ in range(n_cycles):
        U_arr = np.array(U)
        q     = min(int(budget), len(U))
        qi    = np.random.choice(len(U), size=q, replace=False)
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, _, _ = noisy_RLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]); U = list(set(U) - set(U_arr[qi].tolist()))
    F1s['RS+RLA'] = f1_score(y_test, learner.predict(X_test),
                             average=avg_method, zero_division=0)

    # ---- RS+OLA ----
    learner = ActiveLearner(
        estimator=RandomForestClassifier(**RF_PARAMS),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init], y_training=y_train[L_init])
    L = list(L_init); U = list(U_init)
    for _ in range(n_cycles):
        U_arr = np.array(U)
        q     = min(int(budget), len(U))
        qi    = np.random.choice(len(U), size=q, replace=False)
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, _, _ = noisy_OLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]); U = list(set(U) - set(U_arr[qi].tolist()))
    F1s['RS+OLA'] = f1_score(y_test, learner.predict(X_test),
                             average=avg_method, zero_division=0)

    # ---- ES+RLA ----
    learner = ActiveLearner(
        estimator=RandomForestClassifier(**RF_PARAMS),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init], y_training=y_train[L_init])
    L = list(L_init); U = list(U_init)
    for _ in range(n_cycles):
        U_arr = np.array(U)
        qi, _ = learner.query(X_train[U], n_instances=min(int(budget), len(U)))
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, _, _ = noisy_RLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]); U = list(set(U) - set(U_arr[qi].tolist()))
    F1s['ES+RLA'] = f1_score(y_test, learner.predict(X_test),
                             average=avg_method, zero_division=0)

    # ---- ES+OLA ----
    learner = ActiveLearner(
        estimator=RandomForestClassifier(**RF_PARAMS),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init], y_training=y_train[L_init])
    L = list(L_init); U = list(U_init)
    for _ in range(n_cycles):
        U_arr = np.array(U)
        qi, _ = learner.query(X_train[U], n_instances=min(int(budget), len(U)))
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, _, _ = noisy_OLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]); U = list(set(U) - set(U_arr[qi].tolist()))
    F1s['ES+OLA'] = f1_score(y_test, learner.predict(X_test),
                             average=avg_method, zero_division=0)

    # ---- OLAS ----
    learner = ActiveLearner(
        estimator=RandomForestClassifier(**RF_PARAMS),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init], y_training=y_train[L_init])
    L = list(L_init); U = list(U_init)
    for _ in range(n_cycles):
        U_ent = entropy(learner.predict_proba(X_train[U]).T)
        qi, yn, _, _ = OA(U, U_ent, y_train, beta)
        if len(qi) > 0:
            learner.teach(X=X_train[qi], y=np.array(yn))
            L += qi; U = [i for i in I if i not in set(L)]
    F1s['OLAS'] = f1_score(y_test, learner.predict(X_test),
                           average=avg_method, zero_division=0)

    # ---- MV ----
    learner = ActiveLearner(
        estimator=RandomForestClassifier(**RF_PARAMS),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init], y_training=y_train[L_init])
    L = list(L_init); U = list(U_init)
    for _ in range(n_cycles):
        U_arr = np.array(U)
        mv_q  = min(Cap, len(U))
        qi, _ = learner.query(X_train[U], n_instances=mv_q)
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, _ = noisy_MV(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]); U = list(set(U) - set(U_arr[qi].tolist()))
    F1s['MV'] = f1_score(y_test, learner.predict(X_test),
                         average=avg_method, zero_division=0)

    # ---- KC+RLA ----
    learner = ActiveLearner(
        estimator=RandomForestClassifier(**RF_PARAMS),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init], y_training=y_train[L_init])
    L = list(L_init); U = list(U_init)
    for _ in range(n_cycles):
        U_arr   = np.array(U)
        kc_q    = min(int(budget), len(U))
        loc_idx = kcenter_select(X_train_norm[U_arr], kc_q,
                                 X_train_norm[np.array(L)])
        qi      = np.array(loc_idx)
        ent     = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, _, _ = noisy_RLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]); U = list(set(U) - set(U_arr[qi].tolist()))
    F1s['KC+RLA'] = f1_score(y_test, learner.predict(X_test),
                             average=avg_method, zero_division=0)

    # ---- BADGE+RLA ----
    learner = ActiveLearner(
        estimator=RandomForestClassifier(**RF_PARAMS),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init], y_training=y_train[L_init])
    L = list(L_init); U = list(U_init)
    for _ in range(n_cycles):
        U_arr   = np.array(U)
        q       = min(int(budget), len(U))
        loc_idx = badge_select(X_train[U_arr], learner.estimator, q)
        qi      = np.array(loc_idx)
        ent     = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, _, _ = noisy_RLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]); U = list(set(U) - set(U_arr[qi].tolist()))
    F1s['BADGE+RLA'] = f1_score(y_test, learner.predict(X_test),
                                average=avg_method, zero_division=0)

    return F1s

# =============================================================================
# STEP 11 -- Main loop: datasets x alpha values
# =============================================================================

summary_rows = []

for dataset_name in DATASETS:

    print(f'\n{"="*60}')
    print(f'  Dataset: {dataset_name}')
    print(f'{"="*60}')

    X, y = data_read(dataset_name)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42)

    n_train    = X_train.shape[0]
    class_list = sorted(set(y_train.tolist()))
    c          = len(class_list)
    avg_method = 'binary' if c == 2 else 'macro'
    budget     = int(np.ceil(0.7 * (n_train - 6) / N_CYCLES))
    Cap        = int(np.floor(np.sqrt(budget)))
    M          = int(np.ceil(budget / Cap))

    np.random.seed()
    A         = np.sort(np.random.uniform(0.5, 0.95, size=M))
    A_boosted = np.repeat(A, Cap).tolist()

    _scaler      = StandardScaler()
    X_train_norm = _scaler.fit_transform(X_train)

    epsilon = make_epsilon_nm1()

    safe_name = dataset_name.replace(' ', '_')
    print(f'  n_train={n_train}, budget={budget}, Cap={Cap}, M={M}')

    for alpha_val in ALPHA_VALUES:

        alpha = alpha_val
        alpha_str = str(alpha_val).replace('.', '')   # e.g. 015, 020, 025
        out_file  = result_path(f'{safe_name}_NM1_alpha{alpha_str}.xlsx')

        print(f'\n  -- alpha={alpha_val} --')

        # Resume support
        if os.path.exists(out_file):
            existing  = pd.read_excel(out_file)
            start_rep = len(existing)
            rows      = existing.to_dict('records')
            print(f'  Resuming from rep {start_rep+1} ({start_rep} done)')
        else:
            start_rep = 0
            rows      = []
            print('  Starting fresh')

        for rep in range(start_rep, REPLICATIONS):

            # Build same L_init structure as main code
            I      = list(range(n_train))
            L_init = []
            for cls in class_list:
                cls_idx = [i for i in I if y_train[i] == cls]
                L_init += random.sample(cls_idx, min(INIT_NUMBER, len(cls_idx)))
            U_init = [i for i in I if i not in set(L_init)]

            F1s = run_one_rep(I, L_init, U_init)

            row = {'replication': rep + 1, 'alpha': alpha_val}
            for m in METHODS:
                row[f'F1 {m}'] = F1s[m]
            rows.append(row)

            # Save after every rep (crash safe)
            pd.DataFrame(rows).to_excel(out_file, index=False)

            if (rep + 1) % 10 == 0:
                print(f'  rep {rep+1}/{REPLICATIONS}  '
                      f'OLAS={F1s["OLAS"]:.3f}  '
                      f'BADGE={F1s["BADGE+RLA"]:.3f}')

        # Print summary for this dataset x alpha
        df = pd.DataFrame(rows)
        print(f'\n  Summary {dataset_name} alpha={alpha_val}:')
        for m in METHODS:
            col = f'F1 {m}'
            print(f'    {m}: '
                  f'${df[col].mean():.3f} \\pm {df[col].std():.3f}$')

        # Collect for summary table
        df_rep = pd.DataFrame(rows)
        for m in METHODS:
            summary_rows.append({
                'Dataset': dataset_name,
                'Alpha':   alpha_val,
                'Method':  m,
                'Mean F1': round(df_rep[f'F1 {m}'].mean(), 3),
                'Std F1':  round(df_rep[f'F1 {m}'].std(),  3)
            })

# =============================================================================
# STEP 12 -- Save summary table
# =============================================================================

df_summary = pd.DataFrame(summary_rows)
summary_file = result_path('Sensitivity_Summary.xlsx')
df_summary.to_excel(summary_file, index=False)

print(f'\n{"="*60}')
print('  SENSITIVITY ANALYSIS COMPLETE')
print(f'{"="*60}')

# Print the final summary table clearly
print('\n  Final Summary Table (Mean F1):')
print(f'  {"Dataset":<20} {"Alpha":<8} ', end='')
for m in METHODS:
    print(f'{m:<12}', end='')
print()
print('  ' + '-' * (28 + 12 * len(METHODS)))

for dataset_name in DATASETS:
    for alpha_val in ALPHA_VALUES:
        subset = df_summary[
            (df_summary['Dataset'] == dataset_name) &
            (df_summary['Alpha']   == alpha_val)
        ].set_index('Method')
        print(f'  {dataset_name:<20} {alpha_val:<8}', end='')
        for m in METHODS:
            mean = subset.loc[m, 'Mean F1'] if m in subset.index else float('nan')
            print(f'{mean:<12.3f}', end='')
        print()

print(f'\n  All files saved to: {DRIVE_RESULTS}')
print(f'  Summary table: Sensitivity_Summary.xlsx')
print(f'{"="*60}')