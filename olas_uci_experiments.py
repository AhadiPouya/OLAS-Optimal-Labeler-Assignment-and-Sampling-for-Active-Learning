
# =============================================================================
# CELL 2 — Imports and global parameters
# =============================================================================
import os
import copy
import time
import random
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, accuracy_score)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling

# --------------------------------------------------------------------------
# Patch pandas.DataFrame.append (removed in pandas >= 2.0)
# --------------------------------------------------------------------------
def _append_compat(self, other, ignore_index=False):
    if isinstance(other, dict):
        other = pd.DataFrame([other])
    return pd.concat([self, other], ignore_index=ignore_index)
pd.DataFrame.append = _append_compat

# --------------------------------------------------------------------------
# Drive paths
# --------------------------------------------------------------------------
DRIVE_BASE    = os.environ.get('OLAS_BASE_DIR', 'OLAS_Results')  # set this to your project folder
DRIVE_DATA    = os.path.join(DRIVE_BASE, 'Data')
DRIVE_RESULTS = os.path.join(DRIVE_BASE, 'Results')
os.makedirs(DRIVE_RESULTS, exist_ok=True)

def data_path(filename):
    """Return full path to a data file on Drive."""
    return os.path.join(DRIVE_DATA, filename)

def result_path(filename):
    """Return full path to a results file on Drive."""
    return os.path.join(DRIVE_RESULTS, filename)

# --------------------------------------------------------------------------
# Experiment parameters
# --------------------------------------------------------------------------
ALPHA            = 0.2
BETA             = 0.2
N_CYCLES         = 10
INIT_NUMBER      = 3        # samples per class for UCI initialisation
REPLICATION_UCI  = 100

METHODS = ['RS+RLA', 'RS+OLA', 'ES+RLA', 'ES+OLA', 'OLAS', 'MV', 'KC+RLA']

# Module-level variables set per dataset / noise model in the run loops
alpha      = ALPHA
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
# CELL 3 — Interactive dataset selector
# =============================================================================
print("=" * 55)
print("  OLAS EXPERIMENT RUNNER")
print("=" * 55)
print("  Select the dataset you want to run:\n")

OPTIONS = {
    '1': 'Heart Statlog',
    '2': 'Ionosphere',
    '3': 'Sonar',
    '4': 'Spambase',
    '5': 'Covertype',
}
for k, v in OPTIONS.items():
    print(f"    {k}.  {v}")
print("\n" + "=" * 55)

choice = input("  Enter number (1-6) and press Enter: ").strip()
while choice not in OPTIONS:
    print("  Invalid choice. Please enter a number between 1 and 6.")
    choice = input("  Enter number (1-6) and press Enter: ").strip()

SELECTED = OPTIONS[choice]
print(f"\n  >>> Selected: {SELECTED}")
print("  >>> Starting experiments...\n")

# =============================================================================
# CELL 4 — Noise model factory
# =============================================================================

def make_epsilon(nm):
    """Return the noise function for noise model nm ('NM1' or 'NM2')."""
    if nm == 'NM1':
        def _eps(a, e):
            e = float(np.clip(e, 0.0, 1.0))
            return e * (1.0 - a)
    else:  # NM2 -- code version: h = 0.4*e + 0.3
        def _eps(a, e):
            e = float(np.clip(e, 0.0, 1.0))
            h = 0.4 * e + 0.3
            if e <= 0.5:
                p = 2.0 * h
                return (1.0 - a ** p) ** (1.0 / p)
            else:
                p = 2.0 * (1.0 - h)
                return 1.0 - (1.0 - (1.0 - a) ** p) ** (1.0 / p)
    return _eps

# =============================================================================
# CELL 5 — Labeler assignment / noise functions
# =============================================================================

def noisy_RLA(y_true, model_entropy_query):
    """Random Labeler Assignment."""
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
    """Optimal Labeler Assignment — higher entropy -> higher accuracy labeler."""
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
    """
    OLAS: Optimal Labeler Assignment and Sampling (Algorithm 1 in paper).
    Jointly selects query samples and assigns labelers to minimise noise.
    """
    df = pd.DataFrame({'idx': U, 'ent': U_entropy})
    df = df.sort_values('ent', ascending=False).reset_index(drop=True)

    A_list      = sorted(A, reverse=True)
    query_idx   = []
    y_noisy     = []
    total_flip  = 0
    total_noise = 0.0
    last_query  = -1
    m           = -1
    flag        = 1

    while m < M - 1 and flag == 1:
        m   += 1
        acc  = A_list[m]
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

    return query_idx, y_noisy, total_flip / len(query_idx), total_noise / len(query_idx)


def noisy_MV(y_true, model_entropy_query):
    """
    Majority Voting baseline.
    Each of the Cap selected samples is labeled by ALL M labelers.
    Total annotation cost = Cap * M.
    Tie-breaking: random choice among tied classes.
    """
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


def kcenter_select(X_pool_norm, kc_budget, X_labeled_norm):
    """
    Greedy k-Center selection on L2-normalised feature space.
    Returns list of local indices into X_pool_norm.
    """
    n_pool    = len(X_pool_norm)
    kc_budget = min(kc_budget, n_pool)

    if len(X_labeled_norm) > 0:
        d_to_lab  = pairwise_distances(X_pool_norm, X_labeled_norm,
                                       metric='euclidean')
        min_dists = d_to_lab.min(axis=1).copy()
    else:
        min_dists = np.full(n_pool, np.inf)

    selected = []
    for _ in range(kc_budget):
        idx = int(np.argmax(min_dists))
        selected.append(idx)
        new_d      = pairwise_distances(X_pool_norm[idx:idx + 1],
                                        X_pool_norm,
                                        metric='euclidean').flatten()
        min_dists  = np.minimum(min_dists, new_d)
        min_dists[idx] = -np.inf
    return selected

# =============================================================================
# CELL 6 — Data loading
# =============================================================================

def data_read(dataset_name):
    """Load and return (X, y) for UCI / Covertype datasets from Drive."""
    if dataset_name == 'Heart Statlog':
        df  = pd.read_csv(data_path('heart.dat'), header=None, delimiter=' ')
        arr = df.values
        X   = arr[:, :13].astype(float)
        y   = arr[:, 13].astype(int) - 1

    elif dataset_name == 'Ionosphere':
        df  = pd.read_csv(data_path('ionosphere.data'), header=None)
        arr = df.values
        X   = arr[:, :34].astype(float)
        y   = np.where(arr[:, 34] == 'g', 0, 1).astype(int)

    elif dataset_name == 'Sonar':
        df  = pd.read_csv(data_path('sonar.all-data'), header=None)
        arr = df.values
        X   = arr[:, :60].astype(float)
        y   = np.where(arr[:, 60] == 'M', 0, 1).astype(int)

    elif dataset_name == 'Spambase':
        df  = pd.read_csv(data_path('spambase.data'), header=None)
        arr = df.values
        X   = arr[:, :57].astype(float)
        y   = arr[:, 57].astype(int)

    elif dataset_name == 'Covertype':
        from sklearn.datasets import fetch_covtype
        from sklearn.model_selection import StratifiedShuffleSplit
        print('  Downloading Covertype from sklearn (this may take a moment)...')
        raw   = fetch_covtype()
        X_all = raw.data.astype(float)
        y_all = (raw.target - 1).astype(int)
        sss   = StratifiedShuffleSplit(n_splits=1, train_size=50000,
                                       random_state=42)
        idx_keep, _ = next(sss.split(X_all, y_all))
        X = X_all[idx_keep]
        y = y_all[idx_keep]

    else:
        raise ValueError(f'Unknown dataset: {dataset_name}')

    return X, y

# =============================================================================
# =============================================================================







# =============================================================================
# CELL 8 — UCI / Covertype single replication
# =============================================================================

def repeat_uci():
    """
    Run one replication of all 7 methods on the current UCI / Covertype dataset.
    Returns (F1s dict, avg_anr dict, avg_time dict).
    Uses module-level globals: X_train, y_train, X_test, y_test, X_train_norm,
    n_train, class_list, avg_method, budget, Cap, M, A, A_boosted,
    alpha, beta, epsilon, n_cycles.
    """
    I = list(range(n_train))

    # Stratified initialisation: INIT_NUMBER samples per class
    L_init = []
    for cls in class_list:
        cls_idx = [i for i in I if y_train[i] == cls]
        n_take  = min(INIT_NUMBER, len(cls_idx))
        L_init += random.sample(cls_idx, n_take)
    U_init = [i for i in I if i not in set(L_init)]

    timing = {m: [] for m in METHODS}
    anr    = {m: [] for m in METHODS}

    # ------------------------------------------------------------------
    # RS + RLA
    # ------------------------------------------------------------------
    learner = ActiveLearner(
        estimator=RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=50),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)

    for _ in range(n_cycles):
        t0    = time.time()
        U_arr = np.array(U)
        q     = min(int(budget), len(U))
        qi    = np.random.choice(len(U), size=q, replace=False)
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        y_t   = y_train[U_arr[qi]]
        yn, fr, _ = noisy_RLA(list(y_t), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['RS+RLA'].append(fr);  timing['RS+RLA'].append(time.time() - t0)

    F1_RS_RLA = f1_score(y_test, learner.predict(X_test),
                         average=avg_method, zero_division=0)

    # ------------------------------------------------------------------
    # RS + OLA
    # ------------------------------------------------------------------
    learner = ActiveLearner(
        estimator=RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=50),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)

    for _ in range(n_cycles):
        t0    = time.time()
        U_arr = np.array(U)
        q     = min(int(budget), len(U))
        qi    = np.random.choice(len(U), size=q, replace=False)
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        y_t   = y_train[U_arr[qi]]
        yn, fr, _ = noisy_OLA(list(y_t), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['RS+OLA'].append(fr);  timing['RS+OLA'].append(time.time() - t0)

    F1_RS_OLA = f1_score(y_test, learner.predict(X_test),
                         average=avg_method, zero_division=0)

    # ------------------------------------------------------------------
    # ES + RLA
    # ------------------------------------------------------------------
    learner = ActiveLearner(
        estimator=RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=50),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)

    for _ in range(n_cycles):
        t0    = time.time()
        U_arr = np.array(U)
        qi, _ = learner.query(X_train[U], n_instances=min(int(budget), len(U)))
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        y_t   = y_train[U_arr[qi]]
        yn, fr, _ = noisy_RLA(list(y_t), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['ES+RLA'].append(fr);  timing['ES+RLA'].append(time.time() - t0)

    F1_ES_RLA = f1_score(y_test, learner.predict(X_test),
                         average=avg_method, zero_division=0)

    # ------------------------------------------------------------------
    # ES + OLA   *** BUG FIXED: entropy now uses X_train[U_arr[qi]] ***
    # ------------------------------------------------------------------
    learner = ActiveLearner(
        estimator=RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=50),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)

    for _ in range(n_cycles):
        t0    = time.time()
        U_arr = np.array(U)
        qi, _ = learner.query(X_train[U], n_instances=min(int(budget), len(U)))
        # FIXED: was X_train[qi] -- now correctly X_train[U_arr[qi]]
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        y_t   = y_train[U_arr[qi]]
        yn, fr, _ = noisy_OLA(list(y_t), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['ES+OLA'].append(fr);  timing['ES+OLA'].append(time.time() - t0)

    F1_ES_OLA = f1_score(y_test, learner.predict(X_test),
                         average=avg_method, zero_division=0)

    # ------------------------------------------------------------------
    # OLAS
    # ------------------------------------------------------------------
    learner = ActiveLearner(
        estimator=RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=50),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)

    for _ in range(n_cycles):
        t0    = time.time()
        U_ent = entropy(learner.predict_proba(X_train[U]).T)
        qi, yn, fr, _ = OA(U, U_ent, y_train, beta)
        if len(qi) > 0:
            learner.teach(X=X_train[qi], y=np.array(yn))
            L += qi;  U = [i for i in I if i not in set(L)]
            anr['OLAS'].append(fr)
        timing['OLAS'].append(time.time() - t0)

    F1_OA = f1_score(y_test, learner.predict(X_test),
                     average=avg_method, zero_division=0)

    # ------------------------------------------------------------------
    # MV -- Majority Voting
    # Cap unique samples x M labelers = Cap*M total annotation events
    # ------------------------------------------------------------------
    learner = ActiveLearner(
        estimator=RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=50),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)

    for _ in range(n_cycles):
        t0    = time.time()
        U_arr = np.array(U)
        mv_q  = min(Cap, len(U))
        qi, _ = learner.query(X_train[U], n_instances=mv_q)
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        y_t   = y_train[U_arr[qi]]
        yn, fr = noisy_MV(list(y_t), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['MV'].append(fr);  timing['MV'].append(time.time() - t0)

    F1_MV = f1_score(y_test, learner.predict(X_test),
                     average=avg_method, zero_division=0)

    # ------------------------------------------------------------------
    # KC + RLA -- k-Center selection + Random Labeler Assignment
    # ------------------------------------------------------------------
    learner = ActiveLearner(
        estimator=RandomForestClassifier(random_state=0, n_jobs=-1, n_estimators=50),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)

    for _ in range(n_cycles):
        t0       = time.time()
        U_arr    = np.array(U)
        kc_q     = min(int(budget), len(U))
        X_pool_n = X_train_norm[U_arr]
        X_lab_n  = X_train_norm[np.array(L)]
        loc_idx  = kcenter_select(X_pool_n, kc_q, X_lab_n)
        qi       = np.array(loc_idx)
        ent      = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        y_t      = y_train[U_arr[qi]]
        yn, fr, _ = noisy_RLA(list(y_t), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['KC+RLA'].append(fr);  timing['KC+RLA'].append(time.time() - t0)

    F1_KC = f1_score(y_test, learner.predict(X_test),
                     average=avg_method, zero_division=0)

    F1s = {
        'RS+RLA': F1_RS_RLA, 'RS+OLA': F1_RS_OLA,
        'ES+RLA': F1_ES_RLA, 'ES+OLA': F1_ES_OLA,
        'OLAS':   F1_OA,     'MV':     F1_MV,
        'KC+RLA': F1_KC
    }
    avg_anr  = {m: (np.mean(anr[m])    if anr[m]    else 0.0) for m in METHODS}
    avg_time = {m: (np.mean(timing[m]) if timing[m] else 0.0) for m in METHODS}
    return F1s, avg_anr, avg_time

# =============================================================================
# =============================================================================


# =============================================================================
# CELL 10 — RUN: UCI / Covertype (if selected)
# =============================================================================


# =============================================================================
# =============================================================================


# =============================================================================
# DONE
# =============================================================================
print(f'\n{"="*55}')
print(f'  DONE: {SELECTED}')
print(f'  All results saved to:')
print(f'  {DRIVE_RESULTS}')
print(f'{"="*55}')