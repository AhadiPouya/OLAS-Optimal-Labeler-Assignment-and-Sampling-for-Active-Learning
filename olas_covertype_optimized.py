
# =============================================================================
# STEP 1 — Mount Drive
# =============================================================================
from google.colab import drive
drive.mount('/content/drive')

# =============================================================================
# STEP 2 — Imports, warnings, paths
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import (f1_score, precision_score,
                             recall_score, accuracy_score)
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.datasets import fetch_covtype

from modAL.models import ActiveLearner
from modAL.uncertainty import entropy_sampling

warnings.filterwarnings('ignore', category=FutureWarning)

# Patch pandas.DataFrame.append (removed in pandas >= 2.0)
def _append_compat(self, other, ignore_index=False):
    if isinstance(other, dict):
        other = pd.DataFrame([other])
    return pd.concat([self, other], ignore_index=ignore_index)
pd.DataFrame.append = _append_compat

# Drive paths
DRIVE_BASE    = '/content/drive/MyDrive/OLAS_Results'
DRIVE_DATA    = os.path.join(DRIVE_BASE, 'Data')
DRIVE_RESULTS = os.path.join(DRIVE_BASE, 'Results')
os.makedirs(DRIVE_RESULTS, exist_ok=True)

def data_path(f):   return os.path.join(DRIVE_DATA,    f)
def result_path(f): return os.path.join(DRIVE_RESULTS, f)

# =============================================================================
# STEP 3 — Experiment parameters
# =============================================================================
ALPHA            = 0.2
BETA             = 0.2
N_CYCLES         = 10
INIT_NUMBER      = 3
REPLICATION_COV  = 50

METHODS = ['RS+RLA', 'RS+OLA', 'ES+RLA', 'ES+OLA', 'OLAS', 'MV', 'KC+RLA']

# Covertype settings
COV_SAMPLE_SIZE  = 50000   # full 50K as requested by reviewer
CAND_POOL_ES     = 8000    # candidate pool for ES methods
CAND_POOL_OLAS   = 8000    # candidate pool for OLAS
CAND_POOL_MV     = 8000    # candidate pool for MV  <-- MV fix applied
CAND_POOL_KC     = 5000    # candidate pool for KC+RLA
SAVE_EVERY_COV   = 5       # save to Drive every 5 reps

RF_PARAMS_COV = dict(
    random_state=0,
    n_jobs=-1,
    n_estimators=20,
    max_depth=10
)

# Module-level variables set per dataset / noise model
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
# STEP 4 — Interactive selector
# =============================================================================
print("=" * 60)
print("=" * 60)
print("  1.  Covertype  (50K, 50 reps, NM1 + NM2, ~2-3 hrs)")
print("=" * 60)

choice = input("  Enter 1 or 2 and press Enter: ").strip()
while choice not in ('1', '2'):
    print("  Please enter 1 or 2.")
    choice = input("  Enter 1 or 2: ").strip()

SELECTED = 'Covertype'
print(f"\n  >>> Selected: {SELECTED}")
print("  >>> Starting...\n")

# =============================================================================
# STEP 5 — Noise model factory
# =============================================================================

def make_epsilon(nm):
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
# STEP 6 — Helper: candidate pool sampling
# =============================================================================

def sample_candidate_subset(U, max_size):
    """Sample a random candidate subset from unlabeled pool U."""
    if len(U) <= max_size:
        return list(U)
    return random.sample(list(U), max_size)

# =============================================================================
# STEP 7 — Labeler assignment / noise functions
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
    """Optimal Labeler Assignment."""
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
    """OLAS: Optimal Labeler Assignment and Sampling (Algorithm 1)."""
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
    return (query_idx, y_noisy,
            total_flip  / len(query_idx),
            total_noise / len(query_idx))


def noisy_MV(y_true, model_entropy_query):
    """
    Majority Voting baseline.
    Each of Cap selected samples labeled by all M labelers.
    Random tie-breaking.
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
    """Greedy k-Center with ball tree for fast initial distances."""
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
        new_d          = np.linalg.norm(
            X_pool_norm - X_pool_norm[idx], axis=1)
        min_dists      = np.minimum(min_dists, new_d)
        min_dists[idx] = -np.inf
    return selected

# =============================================================================
# STEP 8 — Covertype single replication
# =============================================================================

def repeat_covertype(rep_num=0):
    """
    Run one replication of all 7 methods on Covertype.
    For rep 0 only: prints detailed timing per method.
    Uses ExtraTreesClassifier with candidate pools for speed.
    """
    verbose = (rep_num == 0)
    I       = list(range(n_train))

    # Stratified initialisation
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
    if verbose: print('\n  [RS+RLA] starting...')
    t_method = time.time()
    learner  = ActiveLearner(
        estimator=ExtraTreesClassifier(**RF_PARAMS_COV),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)
    for cyc in range(n_cycles):
        t0    = time.time()
        U_arr = np.array(U)
        q     = min(int(budget), len(U))
        qi    = np.random.choice(len(U), size=q, replace=False)
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, fr, _ = noisy_RLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['RS+RLA'].append(fr);  timing['RS+RLA'].append(time.time()-t0)
        if verbose: print(f'    cyc {cyc+1}/10: {time.time()-t0:.2f}s')
    F1_RS_RLA = f1_score(y_test, learner.predict(X_test),
                         average=avg_method, zero_division=0)
    if verbose: print(f'  [RS+RLA] TOTAL: {time.time()-t_method:.2f}s  F1={F1_RS_RLA:.3f}')

    # ------------------------------------------------------------------
    # RS + OLA
    # ------------------------------------------------------------------
    if verbose: print('\n  [RS+OLA] starting...')
    t_method = time.time()
    learner  = ActiveLearner(
        estimator=ExtraTreesClassifier(**RF_PARAMS_COV),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)
    for cyc in range(n_cycles):
        t0    = time.time()
        U_arr = np.array(U)
        q     = min(int(budget), len(U))
        qi    = np.random.choice(len(U), size=q, replace=False)
        ent   = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, fr, _ = noisy_OLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['RS+OLA'].append(fr);  timing['RS+OLA'].append(time.time()-t0)
        if verbose: print(f'    cyc {cyc+1}/10: {time.time()-t0:.2f}s')
    F1_RS_OLA = f1_score(y_test, learner.predict(X_test),
                         average=avg_method, zero_division=0)
    if verbose: print(f'  [RS+OLA] TOTAL: {time.time()-t_method:.2f}s  F1={F1_RS_OLA:.3f}')

    # ------------------------------------------------------------------
    # ES + RLA
    # ------------------------------------------------------------------
    if verbose: print('\n  [ES+RLA] starting...')
    t_method = time.time()
    learner  = ActiveLearner(
        estimator=ExtraTreesClassifier(**RF_PARAMS_COV),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)
    for cyc in range(n_cycles):
        t0     = time.time()
        U_cand = sample_candidate_subset(U, CAND_POOL_ES)
        U_arr  = np.array(U_cand)
        qi, _  = learner.query(X_train[U_arr],
                               n_instances=min(int(budget), len(U_arr)))
        ent    = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, fr, _ = noisy_RLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['ES+RLA'].append(fr);  timing['ES+RLA'].append(time.time()-t0)
        if verbose: print(f'    cyc {cyc+1}/10: {time.time()-t0:.2f}s')
    F1_ES_RLA = f1_score(y_test, learner.predict(X_test),
                         average=avg_method, zero_division=0)
    if verbose: print(f'  [ES+RLA] TOTAL: {time.time()-t_method:.2f}s  F1={F1_ES_RLA:.3f}')

    # ------------------------------------------------------------------
    # ES + OLA   *** BUG FIXED: entropy uses X_train[U_arr[qi]] ***
    # ------------------------------------------------------------------
    if verbose: print('\n  [ES+OLA] starting...')
    t_method = time.time()
    learner  = ActiveLearner(
        estimator=ExtraTreesClassifier(**RF_PARAMS_COV),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)
    for cyc in range(n_cycles):
        t0     = time.time()
        U_cand = sample_candidate_subset(U, CAND_POOL_ES)
        U_arr  = np.array(U_cand)
        qi, _  = learner.query(X_train[U_arr],
                               n_instances=min(int(budget), len(U_arr)))
        # FIXED: was X_train[qi] -- now X_train[U_arr[qi]]
        ent    = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, fr, _ = noisy_OLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['ES+OLA'].append(fr);  timing['ES+OLA'].append(time.time()-t0)
        if verbose: print(f'    cyc {cyc+1}/10: {time.time()-t0:.2f}s')
    F1_ES_OLA = f1_score(y_test, learner.predict(X_test),
                         average=avg_method, zero_division=0)
    if verbose: print(f'  [ES+OLA] TOTAL: {time.time()-t_method:.2f}s  F1={F1_ES_OLA:.3f}')

    # ------------------------------------------------------------------
    # OLAS
    # ------------------------------------------------------------------
    if verbose: print('\n  [OLAS] starting...')
    t_method = time.time()
    learner  = ActiveLearner(
        estimator=ExtraTreesClassifier(**RF_PARAMS_COV),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)
    for cyc in range(n_cycles):
        t0     = time.time()
        U_cand = sample_candidate_subset(U, CAND_POOL_OLAS)
        U_ent  = entropy(learner.predict_proba(X_train[U_cand]).T)
        qi, yn, fr, _ = OA(U_cand, U_ent, y_train, beta)
        if len(qi) > 0:
            learner.teach(X=X_train[qi], y=np.array(yn))
            L += qi;  U = list(set(U) - set(qi))
            anr['OLAS'].append(fr)
        timing['OLAS'].append(time.time()-t0)
        if verbose: print(f'    cyc {cyc+1}/10: {time.time()-t0:.2f}s  queries={len(qi)}')
    F1_OA = f1_score(y_test, learner.predict(X_test),
                     average=avg_method, zero_division=0)
    if verbose: print(f'  [OLAS] TOTAL: {time.time()-t_method:.2f}s  F1={F1_OA:.3f}')

    # ------------------------------------------------------------------
    # MV -- Majority Voting with candidate pool (MV fix applied)
    # ------------------------------------------------------------------
    if verbose: print('\n  [MV] starting...')
    t_method = time.time()
    learner  = ActiveLearner(
        estimator=ExtraTreesClassifier(**RF_PARAMS_COV),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)
    for cyc in range(n_cycles):
        t0     = time.time()
        # MV FIX: use candidate pool instead of full unlabeled pool
        U_cand = sample_candidate_subset(U, CAND_POOL_MV)
        U_arr  = np.array(U_cand)
        mv_q   = min(Cap, len(U_arr))
        qi, _  = learner.query(X_train[U_arr], n_instances=mv_q)
        ent    = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, fr = noisy_MV(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['MV'].append(fr);  timing['MV'].append(time.time()-t0)
        if verbose: print(f'    cyc {cyc+1}/10: {time.time()-t0:.2f}s')
    F1_MV = f1_score(y_test, learner.predict(X_test),
                     average=avg_method, zero_division=0)
    if verbose: print(f'  [MV] TOTAL: {time.time()-t_method:.2f}s  F1={F1_MV:.3f}')

    # ------------------------------------------------------------------
    # KC + RLA
    # ------------------------------------------------------------------
    if verbose: print('\n  [KC+RLA] starting...')
    t_method = time.time()
    learner  = ActiveLearner(
        estimator=ExtraTreesClassifier(**RF_PARAMS_COV),
        query_strategy=entropy_sampling,
        X_training=X_train[L_init],
        y_training=y_train[L_init])
    L = list(L_init);  U = list(U_init)
    for cyc in range(n_cycles):
        t0       = time.time()
        U_cand   = sample_candidate_subset(U, CAND_POOL_KC)
        U_arr    = np.array(U_cand)
        kc_q     = min(int(budget), len(U_arr))
        loc_idx  = kcenter_select(X_train_norm[U_arr], kc_q,
                                  X_train_norm[np.array(L)])
        qi       = np.array(loc_idx)
        ent      = entropy(learner.predict_proba(X_train[U_arr[qi]]).T)
        yn, fr, _ = noisy_RLA(list(y_train[U_arr[qi]]), ent)
        learner.teach(X=X_train[U_arr[qi]], y=np.array(yn))
        L += list(U_arr[qi]);  U = list(set(U) - set(U_arr[qi].tolist()))
        anr['KC+RLA'].append(fr);  timing['KC+RLA'].append(time.time()-t0)
        if verbose: print(f'    cyc {cyc+1}/10: {time.time()-t0:.2f}s')
    F1_KC = f1_score(y_test, learner.predict(X_test),
                     average=avg_method, zero_division=0)
    if verbose: print(f'  [KC+RLA] TOTAL: {time.time()-t_method:.2f}s  F1={F1_KC:.3f}')

    if verbose: print('\n  === REP 1 COMPLETE ===')

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
# =============================================================================


# =============================================================================
# STEP 11 — RUN: Covertype
# =============================================================================

if SELECTED == 'Covertype':

    print(f'\n{"="*60}')
    print('  Loading Covertype...')
    print(f'{"="*60}')

    raw   = fetch_covtype()
    X_all = raw.data.astype(float)
    y_all = (raw.target - 1).astype(int)

    sss = StratifiedShuffleSplit(n_splits=1,
                                 train_size=COV_SAMPLE_SIZE,
                                 random_state=42)
    idx_keep, _ = next(sss.split(X_all, y_all))
    X = X_all[idx_keep]
    y = y_all[idx_keep]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=True, random_state=42)

    n_train    = X_train.shape[0]
    class_list = sorted(set(y_train.tolist()))
    c          = len(class_list)
    avg_method = 'macro'

    budget = int(np.ceil(0.7 * (n_train - 6) / N_CYCLES))
    Cap    = int(np.floor(np.sqrt(budget)))
    M      = int(np.ceil(budget / Cap))

    np.random.seed()
    A         = np.sort(np.random.uniform(0.5, 0.95, size=M))
    A_boosted = np.repeat(A, Cap).tolist()

    scaler_cov   = StandardScaler()
    X_train_norm = scaler_cov.fit_transform(X_train)

    print(f'  n_train={n_train}, classes={c}, budget={budget}, Cap={Cap}, M={M}')
    print(f'  Replications: {REPLICATION_COV} per noise model')
    print(f'  Sample size: {COV_SAMPLE_SIZE}')
    print(f'  Classifier: ExtraTrees n_estimators=20 max_depth=10')
    print(f'  Candidate pools: ES/MV={CAND_POOL_ES}, OLAS={CAND_POOL_OLAS}, KC={CAND_POOL_KC}')

    for nm in ['NM1', 'NM2']:
        print(f'\n  --- Noise model: {nm} ---')
        epsilon  = make_epsilon(nm)
        out_file = result_path(f'Covertype_{nm}.xlsx')

        # Resume check
        if os.path.exists(out_file):
            existing  = pd.read_excel(out_file)
            start_rep = len(existing)
            rows      = existing.to_dict('records')
            print(f'  Resuming from rep {start_rep + 1} ({start_rep} already done)')
        else:
            start_rep = 0
            rows      = []
            print('  Starting fresh')

        for rep in range(start_rep, REPLICATION_COV):
            F1s, avg_anr, avg_time = repeat_covertype(rep_num=rep)

            row = {'replication': rep + 1}
            for m in METHODS:
                row[f'F1 {m}']   = F1s[m]
                row[f'ANR {m}']  = avg_anr[m]
                row[f'time {m}'] = avg_time[m]
            rows.append(row)

            # Save every SAVE_EVERY_COV reps or at the very last rep
            if ((rep + 1) % SAVE_EVERY_COV == 0) or ((rep + 1) == REPLICATION_COV):
                pd.DataFrame(rows).to_excel(out_file, index=False)
                print(f'  rep {rep+1}/{REPLICATION_COV} done -- checkpoint saved to Drive')
            else:
                print(f'  rep {rep+1}/{REPLICATION_COV} done')

        # Settings file
        pd.DataFrame([{
            'alpha': ALPHA, 'beta': BETA, 'budget': budget,
            'M': M, 'Cap': Cap, 'n_train': n_train,
            'n_test': X_test.shape[0],
            'replications': REPLICATION_COV,
            'cov_sample_size': COV_SAMPLE_SIZE,
            'classifier': 'ExtraTreesClassifier',
            'n_estimators': RF_PARAMS_COV['n_estimators'],
            'max_depth': RF_PARAMS_COV['max_depth'],
            'cand_pool_es': CAND_POOL_ES,
            'cand_pool_olas': CAND_POOL_OLAS,
            'cand_pool_mv': CAND_POOL_MV,
            'cand_pool_kc': CAND_POOL_KC
        }]).to_excel(result_path('Covertype_Settings.xlsx'), index=False)

        df_res = pd.DataFrame(rows)
        print(f'\n  Summary -- Covertype {nm}:')
        for m in METHODS:
            col = f'F1 {m}'
            print(f'    {m}: ${df_res[col].mean():.3f} \\pm {df_res[col].std():.3f}$')

    # Timing summary
    timing_rows = []
    for nm in ['NM1', 'NM2']:
        try:
            df  = pd.read_excel(result_path(f'Covertype_{nm}.xlsx'))
            row = {'Dataset': 'Covertype', 'Noise Model': nm}
            for m in METHODS:
                row[f'Avg Time/Cycle {m} (s)'] = round(df[f'time {m}'].mean(), 4)
            timing_rows.append(row)
        except FileNotFoundError:
            pass

    if timing_rows:
        timing_out = result_path('Timing_Summary_UCI.xlsx')
        if os.path.exists(timing_out):
            existing_t = pd.read_excel(timing_out)
            existing_t = existing_t[existing_t['Dataset'] != 'Covertype']
            timing_rows = existing_t.to_dict('records') + timing_rows
        pd.DataFrame(timing_rows).to_excel(timing_out, index=False)
        print('\n  Timing summary updated.')

    print('\n  Covertype complete. All results saved to Drive.')

# =============================================================================
# =============================================================================


# =============================================================================
# DONE
# =============================================================================
print(f'\n{"="*60}')
print(f'  DONE: {SELECTED}')
print(f'  Results saved to: {DRIVE_RESULTS}')
print(f'{"="*60}')