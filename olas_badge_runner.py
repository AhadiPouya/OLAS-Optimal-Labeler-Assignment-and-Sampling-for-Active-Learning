
# =============================================================================
# STEP 1 -- Mount Drive
# =============================================================================
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# STEP 2 -- Imports
# =============================================================================
import os
import time
import random
import warnings
import numpy as np
import pandas as pd
from collections import Counter
from scipy.stats import entropy

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, accuracy_score)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.datasets import fetch_covtype

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
DRIVE_BASE    = '/content/drive/MyDrive/OLAS_Results'
DRIVE_DATA    = os.path.join(DRIVE_BASE, 'Data')
DRIVE_RESULTS = os.path.join(DRIVE_BASE, 'Results')
os.makedirs(DRIVE_RESULTS, exist_ok=True)

def data_path(f):   return os.path.join(DRIVE_DATA,    f)
def result_path(f): return os.path.join(DRIVE_RESULTS, f)

# =============================================================================
# STEP 4 -- Parameters  (must match your main code exactly)
# =============================================================================
ALPHA            = 0.2
BETA             = 0.2
N_CYCLES         = 10
INIT_NUMBER      = 3
REPLICATION_UCI  = 100
REPLICATION_COV  = 50
COV_SAMPLE_SIZE  = 50000
SAVE_EVERY       = 5

CAND_POOL_BADGE  = 8000   # candidate pool for Covertype (keep same as main code)

RF_PARAMS_UCI  = dict(random_state=0, n_jobs=-1, n_estimators=50)
RF_PARAMS_COV  = dict(random_state=0, n_jobs=-1, n_estimators=20, max_depth=10)

# Module-level globals
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
# STEP 5 -- Dataset selector
# =============================================================================
print("=" * 60)
print("  BADGE+RLA STANDALONE RUNNER")
print("=" * 60)
OPTIONS = {
    '1': 'Heart Statlog',
    '2': 'Ionosphere',
    '3': 'Sonar',
    '4': 'Spambase',
    '5': 'Covertype',
}
for k, v in OPTIONS.items():
    print(f"    {k}.  {v}")
print("=" * 60)
choice = input("  Enter number (1-6): ").strip()
while choice not in OPTIONS:
    choice = input("  Please enter 1-6: ").strip()
SELECTED = OPTIONS[choice]
print(f"\n  >>> Selected: {SELECTED}\n")

# =============================================================================
# STEP 6 -- Noise model factory
# =============================================================================

def make_epsilon(nm):
    if nm == 'NM1':
        def _eps(a, e):
            e = float(np.clip(e, 0.0, 1.0))
            return e * (1.0 - a)
    else:
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
# STEP 7 -- BADGE pseudo-gradient embedding and k-means++ selection
# =============================================================================

def badge_embedding(X_pool, model):
    """
    Pseudo-gradient embeddings for BADGE adaptation.
    g_x = (p - e_yhat) outer_product x
    following the structural form of Ash et al. (2020) eq. (1),
    using the raw feature vector in place of the neural network
    penultimate layer representation.
    """
    probs = model.predict_proba(X_pool)     # (n, K)
    yhat  = np.argmax(probs, axis=1)        # (n,)
    n, K  = probs.shape
    d     = X_pool.shape[1]

    embeddings = np.zeros((n, K * d), dtype=np.float32)
    for i in range(n):
        diff           = probs[i].copy()
        diff[yhat[i]] -= 1.0
        embeddings[i]  = np.outer(diff, X_pool[i]).flatten()
    return embeddings


def kmeans_pp_select(embeddings, budget):
    """k-means++ seeding -- returns list of selected local indices."""
    n      = len(embeddings)
    budget = min(budget, n)

    idx      = np.random.randint(0, n)
    selected = [idx]
    min_sq   = np.sum((embeddings - embeddings[idx]) ** 2, axis=1)

    for _ in range(1, budget):
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


def badge_select(X_pool, model, budget):
    """Compute embeddings and run k-means++ to select query indices."""
    embeddings = badge_embedding(X_pool, model)
    return kmeans_pp_select(embeddings, budget)


def sample_candidate_subset(U, max_size):
    if len(U) <= max_size:
        return list(U)
    return random.sample(list(U), max_size)

# =============================================================================
# STEP 8 -- Noise application for RLA  (needed to record ANR)
# =============================================================================

def noisy_RLA(y_true, model_entropy_query):
    """Random Labeler Assignment -- returns noisy labels and flip rate."""
    import copy
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

# =============================================================================
# STEP 9 -- Data loading
# =============================================================================

def data_read(dataset_name):
    if dataset_name == 'Heart Statlog':
        df  = pd.read_csv(data_path('heart.dat'), header=None, delimiter=' ')
        arr = df.values
        return arr[:, :13].astype(float), arr[:, 13].astype(int) - 1

    elif dataset_name == 'Ionosphere':
        df  = pd.read_csv(data_path('ionosphere.data'), header=None)
        arr = df.values
        return arr[:, :34].astype(float), np.where(arr[:, 34]=='g', 0, 1).astype(int)

    elif dataset_name == 'Sonar':
        df  = pd.read_csv(data_path('sonar.all-data'), header=None)
        arr = df.values
        return arr[:, :60].astype(float), np.where(arr[:, 60]=='M', 0, 1).astype(int)

    elif dataset_name == 'Spambase':
        df  = pd.read_csv(data_path('spambase.data'), header=None)
        arr = df.values
        return arr[:, :57].astype(float), arr[:, 57].astype(int)

    elif dataset_name == 'Covertype':
        print('  Downloading Covertype...')
        raw   = fetch_covtype()
        X_all = raw.data.astype(float)
        y_all = (raw.target - 1).astype(int)
        sss   = StratifiedShuffleSplit(n_splits=1,
                                       train_size=COV_SAMPLE_SIZE,
                                       random_state=42)
        idx_keep, _ = next(sss.split(X_all, y_all))
        return X_all[idx_keep], y_all[idx_keep]

    else:
        raise ValueError(f'Unknown: {dataset_name}')

# =============================================================================
# =============================================================================




# =============================================================================
# STEP 11 -- Single replication: UCI (1-4)
# =============================================================================

def run_badge_uci():
    """One replication of BADGE+RLA on UCI dataset."""
    I      = list(range(n_train))
    L_init = []
    for cls in class_list:
        cls_idx = [i for i in I if y_train[i] == cls]
        L_init += random.sample(cls_idx, min(INIT_NUMBER, len(cls_idx)))
    U_init = [i for i in I if i not in set(L_init)]

    learner = ActiveLearner(
        estimator=RandomForestClassifier(**RF_PARAMS_UCI),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])

    L = list(L_init)
    U = list(U_init)
    cycle_times = []
    cycle_anr   = []

    for _ in range(n_cycles):
        t0    = time.time()
        U_arr = np.array(U)
        q     = min(int(budget), len(U))

        loc_idx = badge_select(X_train[U_arr], learner.estimator, q)
        qi      = np.array(loc_idx)
        ent     = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, fr, _ = noisy_RLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi])
        U  = list(set(U) - set(U_arr[qi].tolist()))

        cycle_times.append(time.time() - t0)
        cycle_anr.append(fr)

    f1 = f1_score(y_test, learner.predict(X_test),
                  average=avg_method, zero_division=0)
    return f1, np.mean(cycle_anr), np.mean(cycle_times)

# =============================================================================
# STEP 12 -- Single replication: Covertype
# =============================================================================

def run_badge_covertype(rep_num=0):
    """One replication of BADGE+RLA on Covertype."""
    I      = list(range(n_train))
    L_init = []
    for cls in class_list:
        cls_idx = [i for i in I if y_train[i] == cls]
        L_init += random.sample(cls_idx, min(INIT_NUMBER, len(cls_idx)))
    U_init = [i for i in I if i not in set(L_init)]

    learner = ActiveLearner(
        estimator=ExtraTreesClassifier(**RF_PARAMS_COV),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])

    L = list(L_init)
    U = list(U_init)
    cycle_times = []
    cycle_anr   = []

    for cyc in range(n_cycles):
        t0     = time.time()
        U_cand = sample_candidate_subset(U, CAND_POOL_BADGE)
        U_arr  = np.array(U_cand)
        q      = min(int(budget), len(U_arr))

        loc_idx = badge_select(X_train[U_arr], learner.estimator, q)
        qi      = np.array(loc_idx)
        ent     = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, fr, _ = noisy_RLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi])
        U  = list(set(U) - set(U_arr[qi].tolist()))

        t_cyc = time.time() - t0
        cycle_times.append(t_cyc)
        cycle_anr.append(fr)
        if rep_num == 0:
            print(f'    cyc {cyc+1}/10: {t_cyc:.2f}s  queries={len(qi)}')

    f1 = f1_score(y_test, learner.predict(X_test),
                  average=avg_method, zero_division=0)
    return f1, np.mean(cycle_anr), np.mean(cycle_times)

# =============================================================================
# =============================================================================


# =============================================================================
# STEP 14 -- RUN: UCI (1-4)
# =============================================================================


# =============================================================================
# STEP 15 -- RUN: Covertype
# =============================================================================

if SELECTED == 'Covertype':

    X, y = data_read('Covertype')
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42)

    n_train    = X_train.shape[0]
    class_list = sorted(set(y_train.tolist()))
    c          = len(class_list)
    avg_method = 'macro'
    budget     = int(np.ceil(0.7 * (n_train - 6) / N_CYCLES))
    Cap        = int(np.floor(np.sqrt(budget)))
    M          = int(np.ceil(budget / Cap))

    np.random.seed()
    A         = np.sort(np.random.uniform(0.5, 0.95, size=M))
    A_boosted = np.repeat(A, Cap).tolist()

    print(f'  n_train={n_train}, c={c}, budget={budget}, Cap={Cap}, M={M}')
    print(f'  Candidate pool: {CAND_POOL_BADGE}')

    for nm in ['NM1', 'NM2']:
        print(f'\n  --- {nm} ---')
        epsilon  = make_epsilon(nm)
        out_file = result_path(f'Covertype_{nm}_BADGE.xlsx')

        if os.path.exists(out_file):
            existing  = pd.read_excel(out_file)
            start_rep = len(existing)
            rows      = existing.to_dict('records')
            print(f'  Resuming from rep {start_rep+1}')
        else:
            start_rep = 0
            rows      = []
            print('  Starting fresh')

        for rep in range(start_rep, REPLICATION_COV):
            f1, anr, t = run_badge_covertype(rep_num=rep)
            rows.append({
                'replication':     rep + 1,
                'F1 BADGE+RLA':   f1,
                'ANR BADGE+RLA':  anr,
                'time BADGE+RLA': t
            })
            if ((rep + 1) % SAVE_EVERY == 0) or ((rep + 1) == REPLICATION_COV):
                pd.DataFrame(rows).to_excel(out_file, index=False)
                print(f'  rep {rep+1}/{REPLICATION_COV} checkpoint  F1={f1:.3f}')

        df = pd.DataFrame(rows)
        print(f'\n  BADGE+RLA Covertype {nm}: '
              f'${df["F1 BADGE+RLA"].mean():.3f} '
              f'\\pm {df["F1 BADGE+RLA"].std():.3f}$')

    print('\n  Covertype BADGE+RLA done.')
    print('  Files: Covertype_NM1_BADGE.xlsx  and  Covertype_NM2_BADGE.xlsx')

# =============================================================================
# =============================================================================


# =============================================================================
# DONE
# =============================================================================
print(f'\n{"="*60}')
print(f'  BADGE+RLA DONE: {SELECTED}')
print(f'  Results saved to: {DRIVE_RESULTS}')
print(f'{"="*60}')