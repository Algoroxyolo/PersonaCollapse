"""
Persona Collapse Analysis Pipeline
===================================
Complete analysis code for "The Chameleon's Limit" paper.

Run from the repository root:

    python -m analysis.analysis_pipeline \
        --data_dir ./csv_exports \
        --selfintro_dir ./self_introduction_results \
        --human_ref ./data/human_reference/wave_1_numbers.csv \
        --output_dir ./results

Or directly:

    python analysis/analysis_pipeline.py \
        --data_dir ./csv_exports \
        --selfintro_dir ./self_introduction_results \
        --human_ref ./data/human_reference/wave_1_numbers.csv \
        --output_dir ./results

Optional flags:
    --flagged ./data/flagged_personas.pkl
    --skip_selfintro
    --skip_figures

Outputs:
    results/analysis.json          - All BFI + Moral metrics per model
    results/tables/*.tex           - All LaTeX tables
    results/figures/*.pdf          - All figures
    results/selfintro/*.csv        - Self-introduction analysis results
"""

import argparse, json, os, re, warnings, pickle
from collections import Counter

import numpy as np
import pandas as pd
from scipy.stats import entropy, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import silhouette_score, v_measure_score
from sklearn.metrics.pairwise import euclidean_distances, cosine_similarity
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings('ignore')

# ================================================================
# CONFIG
# ================================================================
BFI_COLS = [f'BFI_Q{i}' for i in range(1, 45)]
SCEN_COLS = [f'scenario_{i}' for i in range(1, 132)]
BFI_COLS_HUMAN = [f'Big Five _{i}' for i in range(1, 45)]
TRAIT_ITEMS = {
    'E': ([1,11,16,26,36], [6,21,31]),
    'A': ([7,17,22,32,42], [2,12,27,37]),
    'C': ([3,13,28,33,38], [8,18,23,43]),
    'N': ([4,14,19,29,39], [9,24,34]),
    'O': ([5,10,15,20,25,30,40,44], [35,41]),
}
RNG = np.random.RandomState(42)

# Model registry: (short_name, bfi_folder, moral_folder, selfintro_name, category)
MODEL_REGISTRY = [
    ('CoSER-Llama-8B',    'digital_twin_CoSER-Llama-3.1-8B',     'moral_reasoning_CoSER-Llama-3.1-8B',     'CoSER-Llama-3.1-8B',     'main'),
    ('Llama-3.1-8B',      'digital_twin_Llama-3.1-8B-Instruct',  'moral_reasoning_Llama-3.1-8B-Instruct',  'Llama-3.1-8B-Instruct',  'main'),
    ('Qwen3-4B',          'digital_twin_Qwen3-4B-Instruct-2507', 'moral_reasoning_Qwen3-4B-Instruct-2507', 'Qwen3-4B-Instruct-2507', 'main'),
    ('Qwen3-30B',         'digital_twin_Qwen3-30B-A3B-Instruct-2507', 'moral_reasoning_Qwen3-30B-A3B-Instruct-2507', 'Qwen3-30B-A3B-Instruct-2507', 'main'),
    ('Qwen3-32B',         'digital_twin_Qwen3-32B',              'moral_reasoning_Qwen3-32B',              'Qwen3-32B',              'main'),
    ('CoSER-Qwen-32B',    'digital_twin_qwen3_32b_march_2',      'moral_reasoning_qwen3_32b_march_2',      'qwen3_32b_march_2',      'main'),
    ('HER-32B',           'digital_twin_HER-32B',                 'moral_reasoning_HER-32B',                'HER-32B',                'main'),
    ('Claude-Haiku-4.5',  'digital_twin_claude-haiku-4.5',        'moral_reasoning_claude-haiku-4.5',       'claude-haiku-4.5',       'main'),
    ('MiniMax-M2',        'digital_twin_minimax-m2',              'moral_reasoning_minimax-m2',             'minimax-m2',             'main'),
    ('MiniMax-M2-Her',    'digital_twin_minimax-m2-her',          'moral_reasoning_minimax-m2-her',         'minimax-m2-her',         'main'),
    # Appendix-only
    ('Qwen3-32B-think',   'digital_twin_Qwen3-32B-thinking',     'moral_reasoning_Qwen3-32B-thinking',     None,                     'appendix'),
    ('Qwen3-32B-nothink', 'digital_twin_Qwen3-32B-nonthinking',  'moral_reasoning_Qwen3-32B-nonthinking',  None,                     'appendix'),
]

# ================================================================
# GEOMETRIC UTILITIES
# ================================================================

def compute_lid(X, k=20):
    """Local Intrinsic Dimensionality (MLE estimator)."""
    nn = NearestNeighbors(n_neighbors=k+1).fit(X)
    d, _ = nn.kneighbors(X)
    d = np.maximum(d[:, 1:], 1e-10)
    return -1.0 / (np.log(d / d[:, -1:])[:, :-1].mean(axis=1))


def compute_hopkins(X, seed=42):
    """Hopkins statistic for clustering tendency."""
    r = np.random.RandomState(seed)
    n = len(X)
    ns = min(100, n // 3)
    idx = r.choice(n, ns, replace=False)
    nn = NearestNeighbors(n_neighbors=1).fit(np.delete(X, idx, axis=0))
    u, _ = nn.kneighbors(X[idx])
    rp = r.uniform(X.min(0), X.max(0), (ns, X.shape[1]))
    nn2 = NearestNeighbors(n_neighbors=1).fit(X)
    w, _ = nn2.kneighbors(rp)
    return float(w.sum() / (u.sum() + w.sum()))


def compute_density_coverage(real, fake, k=5):
    """Density & Coverage (Naeem et al., 2020)."""
    nn = NearestNeighbors(n_neighbors=k+1).fit(real)
    rd, _ = nn.kneighbors(real)
    radii = rd[:, -1]
    nn2 = NearestNeighbors(n_neighbors=1).fit(fake)
    d2, _ = nn2.kneighbors(real)
    cov = float((d2[:, 0] <= radii).mean())
    nn3 = NearestNeighbors(n_neighbors=k).fit(real)
    df2r, idx = nn3.kneighbors(fake)
    ds = sum(1 for i in range(len(fake)) for j in range(k)
             if df2r[i, j] <= radii[idx[i, j]])
    den = ds / (k * len(fake))
    return den, cov


def score_big5(df, cols=None):
    """Score BFI-44 items into Big Five factors."""
    if cols is None:
        cols = BFI_COLS
    s = {}
    for t, (pos, neg) in TRAIT_ITEMS.items():
        pc = [cols[i-1] for i in pos if i-1 < len(cols) and cols[i-1] in df.columns]
        nc = [cols[i-1] for i in neg if i-1 < len(cols) and cols[i-1] in df.columns]
        parts = []
        if pc: parts.append(df[pc])
        if nc: parts.append(6 - df[nc])
        if parts:
            s[t] = pd.concat(parts, axis=1).mean(axis=1)
    return pd.DataFrame(s)


# ================================================================
# DEMOGRAPHIC EXTRACTION
# ================================================================

def extract_demos(desc):
    """Extract demographic fields from persona description text."""
    r = {}
    patterns = [
        ('gender', r'Gender:\s*(.+?)(?:\n|$)'),
        ('country', r'Country.*?:\s*(.+?)(?:\n|$)'),
        ('political', r'Political.*?:\s*(.+?)(?:\n|$)'),
        ('social_class', r'Social Class:\s*(.+?)(?:\n|$)'),
    ]
    for field, pat in patterns:
        m = re.search(pat, str(desc))
        r[field] = m.group(1).strip() if m else None
    return r


# ================================================================
# CORE ANALYSIS: BFI + MORAL
# ================================================================

def full_analysis(resp, df, demos, cols, domain, human_sub=None, human_scaler=None):
    """Run all diagnostics on one model-domain pair."""
    n = len(resp)
    D = len(cols)

    # Item-level: inverse Simpson
    effs = []
    for c in cols:
        v = resp[c].dropna()
        if v.nunique() > 0:
            vc = v.value_counts(normalize=True)
            effs.append(1.0 / (vc**2).sum())
    zv = sum(1 for c in cols if c in resp.columns and resp[c].std() == 0)

    # PCA
    Xs = StandardScaler().fit_transform(resp.values)
    ev = PCA().fit(Xs).explained_variance_ratio_
    pr = (ev.sum()**2) / (ev**2).sum()

    # LID, Separation, Hopkins (subsample)
    sub = resp.sample(min(500, n), random_state=42).values.astype(float)
    sub_s = StandardScaler().fit_transform(sub)
    try:
        lids = compute_lid(sub, k=min(20, len(sub) // 3))
        lid = float(np.nanmedian(lids[np.isfinite(lids)]))
    except Exception:
        lid = float('nan')
    Dm = euclidean_distances(sub)
    np.fill_diagonal(Dm, np.inf)
    sep = float(Dm.min(axis=1).mean())
    hop = compute_hopkins(sub_s)

    # Silhouette, V-Measure
    sils = {}
    for K in [5, 10]:
        lab = KMeans(K, random_state=42, n_init=10).fit_predict(Xs)
        sils[K] = silhouette_score(Xs, lab)
    dc = demos.fillna('unk').apply(lambda x: '_'.join(x.astype(str)), axis=1)
    dl = LabelEncoder().fit_transform(dc)
    vms = {}
    for K in [5, 10, 50]:
        lab = KMeans(K, random_state=42, n_init=10).fit_predict(Xs)
        vms[K] = v_measure_score(dl, lab)

    # Coverage (BFI only, requires human reference)
    cov_val = den_val = None
    if human_sub is not None and human_scaler is not None:
        X_llm = human_scaler.transform(resp.values.astype(float))
        llm_sub = X_llm[RNG.choice(len(X_llm), min(500, len(X_llm)), replace=False)]
        den_val, cov_val = compute_density_coverage(human_sub, llm_sub, k=5)

    # Political eta^2
    pols = demos['political'].tolist()
    comb = pd.concat([
        resp.reset_index(drop=True),
        pd.Series(pols, name='pol').reset_index(drop=True)
    ], axis=1).dropna(subset=['pol'])
    pol_etas = []
    for c in cols:
        if c not in comb.columns:
            continue
        data = comb[[c, 'pol']].dropna()
        if data['pol'].nunique() < 2:
            continue
        groups = [g[c].values for _, g in data.groupby('pol') if len(g) > 5]
        if len(groups) < 2:
            continue
        gm = data[c].mean()
        sst = ((data[c] - gm)**2).sum()
        if sst == 0:
            continue
        ssb = sum(len(gg) * (gg.mean() - gm)**2 for gg in groups)
        pol_etas.append(ssb / sst)

    # Incremental R^2
    encoded = pd.DataFrame(index=resp.index)
    for d2 in ['political', 'gender', 'country', 'social_class']:
        encoded[d2] = LabelEncoder().fit_transform(demos[d2].fillna('unk'))
    attr_order = ['political', 'gender', 'country', 'social_class']
    incr = {a: [] for a in attr_order}
    for sc in cols:
        if sc not in resp.columns:
            continue
        y = resp[sc].dropna()
        if len(y) < 50 or np.std(y) == 0:
            continue
        enc_sub = encoded.loc[y.index]
        prev = 0
        for i, a in enumerate(attr_order):
            X = pd.get_dummies(enc_sub[attr_order[:i+1]], drop_first=True).values
            if X.shape[1] == 0:
                continue
            r2 = max(0, LinearRegression().fit(X, y.values).score(X, y.values))
            incr[a].append(max(0, r2 - prev))
            prev = r2
    ir = {a: round(np.mean(incr[a]), 4) if incr[a] else 0 for a in attr_order}

    res = {
        'n': n, 'eff': round(np.mean(effs), 2), 'zv': f'{zv}/{D}',
        'pr': round(pr, 1), 'lid': round(lid, 1), 'sep': round(sep, 2),
        'hop': round(hop, 3),
        'sil': {str(K): round(v, 4) for K, v in sils.items()},
        'vm': {str(K): round(v, 4) for K, v in vms.items()},
        'cov': round(cov_val, 3) if cov_val is not None else None,
        'den': round(den_val, 3) if den_val is not None else None,
        'eta_m': round(np.mean(pol_etas), 4) if pol_etas else 0,
        'eta_x': round(np.max(pol_etas), 4) if pol_etas else 0,
        'incr_r2': ir,
    }

    # BFI-specific: fidelity, Cohen's d, sigma ratio
    if domain == 'bfi' and 'Big Five Openness' in df.columns:
        b5 = score_big5(df)
        res['sig_r'] = round(b5.std().mean() / 0.75, 2)
        tgt_cols = ['Big Five Openness', 'Big Five Conscientiousness',
                    'Big Five Extraversion', 'Big Five Agreeableness',
                    'Big Five Neuroticism']
        tgt = df[tgt_cols]
        tm = {'High': 3, 'Neutral': 2, 'Low': 1}
        rhos, ds = [], []
        for t, tc in zip('OCEAN', tgt_cols):
            if t not in b5.columns:
                continue
            tv = tgt[tc].map(tm)
            v = tv.dropna().index.intersection(b5[t].dropna().index)
            if len(v) > 10:
                rr, _ = spearmanr(tv.loc[v], b5[t].loc[v])
                rhos.append(rr)
            hi = b5[t][tgt[tc] == 'High']
            lo = b5[t][tgt[tc] == 'Low']
            if len(hi) > 5 and len(lo) > 5:
                ps = np.sqrt((hi.var()*(len(hi)-1) + lo.var()*(len(lo)-1)) / (len(hi)+len(lo)-2))
                if ps > 0:
                    ds.append((hi.mean() - lo.mean()) / ps)
        res['fid'] = round(np.mean(rhos), 3) if rhos else None
        res['d'] = round(np.mean(ds), 1) if ds else None

    return res


def load_and_analyze(data_dir, folder, cols, domain, human_sub=None, human_scaler=None, flagged=None):
    """Load CSV, optionally filter flagged personas, and run analysis."""
    path = os.path.join(data_dir, folder, f'{folder}_response_matrix.csv')
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)

    # Filter flagged personas if applicable
    if flagged is not None:
        if 'persona_id' in df.columns and df['persona_id'].max() < 2000:
            df = df[~df['persona_id'].isin(flagged)]

    # Get response columns
    avail = [c for c in cols if c in df.columns]
    resp = df[avail].dropna()

    if len(resp) < 50:
        # Try partial completion with imputation
        notna = df[avail].notna().sum(axis=1)
        mask = notna >= max(len(avail) // 2, 10)
        if mask.sum() >= 50:
            resp = df.loc[mask, avail].copy()
            for i in resp.index:
                med = resp.loc[i].median()
                resp.loc[i] = resp.loc[i].fillna(med)
            df = df.loc[mask]
        else:
            return None

    demos = pd.DataFrame([extract_demos(d) for d in df.loc[resp.index, 'description']])
    result = full_analysis(resp, df.loc[resp.index], demos, avail, domain,
                           human_sub, human_scaler)
    return result


# ================================================================
# SELF-INTRODUCTION ANALYSIS
# ================================================================

# Keyword dictionaries for attribute mention detection
COUNTRY_KEYWORDS = {
    'India': ['india','indian','mumbai','delhi','bangalore','chennai','kolkata','namaste','hindi','rupee'],
    'Brazil': ['brazil','brazilian','sao paulo','rio','portuguese'],
    'China': ['china','chinese','beijing','shanghai','mandarin','guangzhou'],
    'France': ['france','french','paris','lyon','marseille','bonjour'],
    'Nigeria': ['nigeria','nigerian','lagos','abuja','yoruba','igbo'],
    'UnitedStates': ['united states','american','usa','u.s.','new york','california','texas','chicago'],
}
GENDER_KEYWORDS = {
    'm': ['he ','him ','his ','man','father','husband','brother','son ','boy','gentleman','mr.','male'],
    'f': ['she ','her ','hers','woman','mother','wife','sister','daughter','girl','lady','ms.','mrs.','female'],
    'n': ['they ','them ','their ','non-binary','nonbinary','genderqueer','gender-fluid'],
}
AGE_KEYWORDS = {
    'Child': ['child','kid','school','young','grow up','years old','teenager','teen','grade'],
    'Young': ['twenties','20s','young adult','college','university','just graduated','early career'],
    'Middle': ['thirties','forties','30s','40s','mid-career','established'],
    'Older': ['fifties','50s','experienced','decades'],
    'Seniors': ['retired','senior','elderly','grandchild','grandparent','sixties','seventies','60s','70s','80s'],
}
CLASS_KEYWORDS = {
    'lc': ['poor','poverty','struggle financially','humble background','low income','working class',
           'paycheck to paycheck','disadvantaged','modest means'],
    'mc': ['middle class','comfortable','modest','average income'],
    'uc': ['wealthy','affluent','privileged','upper class','prestigious','elite','luxury',
           'well-off','fortune','inheritance'],
}
POLITICAL_KEYWORDS = {
    'll': ['liberal','progressive','left','social justice','equality','welfare','environmentalist','socialist','democrat'],
    'lc': ['left communitarian','community','collective','solidarity','traditional left','labor'],
    'rl': ['libertarian','free market','individual liberty','small government','fiscal conservative','deregulation'],
    'rc': ['conservative','traditional','right','patriot','faith','family values','law and order','republican','nationalist'],
}


def parse_persona_label(label):
    """Parse persona label into demographic dict."""
    parts = label.split('_')
    if len(parts) != 6:
        return None
    ocean = parts[5]
    if len(ocean) != 5:
        return None
    return {
        'age': parts[0], 'gender': parts[1], 'country': parts[2],
        'social_class': parts[3], 'political': parts[4],
        'O': ocean[0], 'C': ocean[1], 'E': ocean[2], 'A': ocean[3], 'N': ocean[4],
    }


def compute_linguistic_features(text):
    """Compute linguistic features for one self-introduction."""
    if not text or len(text) < 20:
        return None
    text = text[:5000]
    words = text.split()
    n_words = len(words)
    if n_words < 5:
        return None

    sents = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    n_sents = max(len(sents), 1)

    wl = [w.lower().strip('.,!?;:\"\\\'-()[]{}') for w in words]
    wl = [w for w in wl if w]
    nwl = max(len(wl), 1)
    types = set(wl)
    cnt = Counter(wl)

    ttr = len(types) / nwl
    hapax = sum(1 for _, c in cnt.items() if c == 1) / nwl
    guiraud = len(types) / np.sqrt(nwl)

    fp_pronouns = {'i','me','my','mine','myself',"i'm","i've","i'd","i'll"}
    fp_rate = sum(1 for w in wl if w in fp_pronouns) / n_words

    hedges = ['maybe','perhaps','probably','might','could','possibly',
              'i think','i guess','i believe','it seems','i suppose',
              'kind of','sort of','in my opinion']
    tl = text.lower()
    hedge_count = sum(tl.count(h) for h in hedges)
    hedge_rate = hedge_count / n_sents

    pos_words = {'happy','love','joy','wonderful','great','amazing','beautiful',
                 'enjoy','grateful','blessed','proud','excited','passionate',
                 'warm','kind','caring','delighted','thrilled'}
    neg_words = {'sad','angry','hate','terrible','struggle','difficult','hard',
                 'pain','fear','worried','anxious','frustrated','lonely','poor',
                 'suffer','lost','afraid','depressed'}
    emo_pos = sum(1 for w in wl if w in pos_words) / n_words
    emo_neg = sum(1 for w in wl if w in neg_words) / n_words

    return {
        'n_words': n_words, 'n_sentences': n_sents,
        'avg_sent_len': n_words / n_sents,
        'ttr': ttr, 'hapax_ratio': hapax, 'guiraud': guiraud,
        'fp_pronoun_rate': fp_rate, 'hedge_rate': hedge_rate,
        'emo_pos_rate': emo_pos, 'emo_neg_rate': emo_neg,
    }


def detect_attribute_mentions(text, true_label):
    """Detect which persona attributes are mentioned in text."""
    tl = text.lower()
    mentions = {}
    for attr, kw_dict in [('country', COUNTRY_KEYWORDS), ('age', AGE_KEYWORDS),
                           ('gender', GENDER_KEYWORDS), ('social_class', CLASS_KEYWORDS),
                           ('political', POLITICAL_KEYWORDS)]:
        true_val = true_label.get(attr, '')
        mentions[attr] = any(kw in tl for kw in kw_dict.get(true_val, []))
        mentions[f'{attr}_any'] = any(any(kw in tl for kw in kws) for kws in kw_dict.values())
    return mentions


def strip_think_tokens(text):
    """Remove <think>...</think> blocks from model output."""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()


def analyze_selfintro(selfintro_dir, models, output_dir):
    """Run all self-introduction analyses."""
    os.makedirs(output_dir, exist_ok=True)
    demo_cols = ['age', 'gender', 'country', 'social_class', 'political']
    feat_cols = ['n_words', 'n_sentences', 'avg_sent_len', 'ttr', 'hapax_ratio',
                 'guiraud', 'fp_pronoun_rate', 'hedge_rate', 'emo_pos_rate', 'emo_neg_rate']

    # Step 1: Extract features + mentions
    print("  Extracting linguistic features...")
    all_rows = []
    for model_name in models:
        path = os.path.join(selfintro_dir, f'introductions_{model_name}.jsonl')
        if not os.path.exists(path):
            print(f"    SKIP {model_name}: file not found")
            continue
        with open(path) as f:
            raw = [json.loads(l) for l in f]
        valid = 0
        for d in raw:
            r = strip_think_tokens(d['response'])
            if len(r) < 20 or r.startswith('[ERROR]'):
                continue
            if "sorry" in r.lower()[:50] and len(r) < 100:
                continue
            parsed = parse_persona_label(d['persona_label'])
            if not parsed:
                continue
            feats = compute_linguistic_features(r)
            if not feats:
                continue
            mentions = detect_attribute_mentions(r, parsed)
            row = {'model': model_name, 'persona_id': d['persona_id'],
                   'sample_idx': d['sample_idx'], 'text_len': len(r),
                   **parsed, **feats, **mentions}
            all_rows.append(row)
            valid += 1
        print(f"    {model_name}: {valid} valid responses")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(output_dir, 'selfintro_features.csv'), index=False)

    # Layer 1: Mention rates
    print("  Computing mention rates...")
    mention_cols = ['country', 'age', 'gender', 'social_class', 'political']
    rows = []
    for model in df['model'].unique():
        mdf = df[df['model'] == model]
        for attr in mention_cols:
            rows.append({'model': model, 'attribute': attr,
                         'mention_rate': mdf[attr].mean()})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'layer1_mention_rates.csv'), index=False)

    # Layer 3: Template detection
    print("  Detecting templates...")
    tmpl_results = {}
    for model in df['model'].unique():
        path = os.path.join(selfintro_dir, f'introductions_{model}.jsonl')
        with open(path) as f:
            raw = [json.loads(l) for l in f]
        texts = [strip_think_tokens(d['response']) for d in raw
                 if len(d['response']) > 50 and not d['response'].startswith('[ERROR]')]
        texts = [t for t in texts if len(t) > 50]
        if len(texts) < 50:
            continue
        openings = []
        for t in texts[:500]:
            first = re.split(r'[.!?\n]', t)[0].strip()
            if len(first) > 10:
                openings.append(first.lower())
        unique = len(set(openings))
        div = unique / len(openings) if openings else 0
        skeletons = []
        for o in openings:
            sk = re.sub(r'\b[A-Z][a-z]+\b', '[NAME]', o)
            sk = re.sub(r'\b\d+\b', '[NUM]', sk)
            skeletons.append(sk)
        skel_counts = Counter(skeletons)
        top = skel_counts.most_common(1)[0] if skel_counts else ('', 0)
        top_pct = top[1] / len(skeletons) if skeletons else 0
        header_counts = [len(re.findall(r'^#+\s', t, re.MULTILINE)) for t in texts[:500]]
        para_counts = [len(re.split(r'\n\n+', t)) for t in texts[:500]]
        tmpl_results[model] = {
            'n_texts': len(texts),
            'opening_diversity': round(div, 3),
            'top_skeleton': top[0][:80],
            'top_skeleton_pct': round(top_pct, 3),
            'avg_headers': round(np.mean(header_counts), 2),
            'std_headers': round(np.std(header_counts), 2),
            'avg_paragraphs': round(np.mean(para_counts), 2),
            'std_paragraphs': round(np.std(para_counts), 2),
        }
    with open(os.path.join(output_dir, 'layer3_template_detection.json'), 'w') as f:
        json.dump(tmpl_results, f, indent=2)

    # Layer 4a: Feature eta^2
    print("  Computing feature eta^2...")
    rows = []
    for model in df['model'].unique():
        mdf = df[df['model'] == model]
        for feat in feat_cols:
            for demo in demo_cols:
                groups = [g[feat].dropna().values for _, g in mdf.groupby(demo) if len(g) > 5]
                if len(groups) < 2:
                    rows.append({'model': model, 'feature': feat, 'demographic': demo, 'eta2': 0})
                    continue
                all_vals = mdf[feat].dropna()
                gm = all_vals.mean()
                sst = ((all_vals - gm)**2).sum()
                if sst == 0:
                    rows.append({'model': model, 'feature': feat, 'demographic': demo, 'eta2': 0})
                    continue
                ssb = sum(len(g) * (g.mean() - gm)**2 for g in groups)
                rows.append({'model': model, 'feature': feat, 'demographic': demo, 'eta2': ssb / sst})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'layer4_feature_eta2.csv'), index=False)

    # Layer 4b: Incremental R^2
    print("  Computing incremental R^2...")
    attr_order = ['political', 'gender', 'country', 'social_class', 'age']
    key_feats = ['ttr', 'hapax_ratio', 'hedge_rate', 'fp_pronoun_rate', 'emo_pos_rate', 'emo_neg_rate']
    rows = []
    for model in df['model'].unique():
        mdf = df[df['model'] == model].copy()
        encoded = pd.DataFrame(index=mdf.index)
        for col in attr_order:
            encoded[col] = LabelEncoder().fit_transform(mdf[col].fillna('unk'))
        for feat in key_feats:
            y = mdf[feat].values
            if np.std(y) == 0:
                for a in attr_order:
                    rows.append({'model': model, 'feature': feat, 'attribute': a, 'incr_r2': 0})
                continue
            prev = 0
            for i, attr in enumerate(attr_order):
                X = pd.get_dummies(encoded[attr_order[:i+1]], drop_first=True).values
                if X.shape[1] == 0:
                    rows.append({'model': model, 'feature': feat, 'attribute': attr, 'incr_r2': 0})
                    continue
                r2 = max(0, LinearRegression().fit(X, y).score(X, y))
                rows.append({'model': model, 'feature': feat, 'attribute': attr, 'incr_r2': max(0, r2 - prev)})
                prev = r2
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'layer4b_incremental_r2.csv'), index=False)

    # Layer 4c: ICC
    print("  Computing ICC...")
    rows = []
    for model in df['model'].unique():
        mdf = df[df['model'] == model]
        for feat in feat_cols:
            groups = mdf.groupby('persona_id')[feat].apply(list)
            groups = groups[groups.apply(len) >= 2]
            if len(groups) < 50:
                rows.append({'model': model, 'feature': feat, 'icc': np.nan})
                continue
            k = groups.apply(len).mean()
            persona_means = groups.apply(np.mean)
            ms_between = k * persona_means.var()
            ms_within = groups.apply(np.var).mean()
            denom = ms_between + (k - 1) * ms_within
            icc = max(0, (ms_between - ms_within) / denom) if denom > 0 else 0
            rows.append({'model': model, 'feature': feat, 'icc': round(icc, 4)})
    pd.DataFrame(rows).to_csv(os.path.join(output_dir, 'layer4c_icc.csv'), index=False)

    print(f"  Self-intro analysis complete. Results in {output_dir}/")
    return df


# ================================================================
# FIGURE GENERATION
# ================================================================

def generate_figures(analysis, output_dir):
    """Generate Coverage-vs-LID and Fidelity-vs-d figures."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    os.makedirs(output_dir, exist_ok=True)
    bfi = analysis['bfi']
    human = analysis['human']

    family_map = {
        'Claude-Haiku-4.5': 'Claude',
        'Qwen3-4B': 'Qwen3', 'Qwen3-30B': 'Qwen3', 'Qwen3-32B': 'Qwen3',
        'CoSER-Qwen-32B': 'Qwen3', 'HER-32B': 'Qwen3',
        'Llama-3.1-8B': 'Llama', 'CoSER-Llama-8B': 'Llama',
        'MiniMax-M2': 'MiniMax', 'MiniMax-M2-Her': 'MiniMax',
    }
    colors = {'Claude': '#2563EB', 'Qwen3': '#059669', 'Llama': '#DC2626',
              'MiniMax': '#7C3AED', 'Human': '#E11D48'}

    SKIP = {'Qwen3-32B-think', 'Qwen3-32B-nothink'}
    models = [(m, bfi[m]) for m in bfi if m not in SKIP and bfi[m].get('cov') is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7.5))
    plt.subplots_adjust(wspace=0.28)
    FONT = {'fontsize': 10, 'fontweight': 'bold', 'color': '#333333'}
    ARROW = dict(arrowstyle='-', color='#AAAAAA', lw=0.6)

    # Left: Coverage vs LID
    ax1.axhline(y=human['lid'], color='#CCCCCC', linestyle='--', linewidth=0.8)
    ax1.axvline(x=1.0, color='#CCCCCC', linestyle='--', linewidth=0.8)
    ax1.scatter(1.0, human['lid'], s=200, c=colors['Human'], marker='*', zorder=10)
    ax1.annotate('Human', (1.0, human['lid']), xytext=(1.03, human['lid'] - 1.8),
                 **FONT, arrowprops=ARROW)
    for name, b in models:
        fam = family_map.get(name, 'Qwen3')
        ax1.scatter(b['cov'], b['lid'], s=100, c=colors[fam], marker='o', alpha=0.85, zorder=5)
    ax1.set_xlabel('Coverage', fontsize=13)
    ax1.set_ylabel('Complexity (LID)', fontsize=13)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Right: Fidelity vs d
    ax2.axhline(y=2, color='#88CC88', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.axhline(y=6, color='#E88888', linestyle='--', linewidth=0.8, alpha=0.6)
    ax2.axvline(x=0.9, color='#CCCCCC', linestyle='--', linewidth=0.8)
    for name, b in models:
        if b.get('fid') is None:
            continue
        fam = family_map.get(name, 'Qwen3')
        if name == 'MiniMax-M2-Her':
            ax2.scatter(0.783, b['d'], s=80, c=colors[fam], marker='<', alpha=0.7, zorder=5)
        else:
            ax2.scatter(b['fid'], b['d'], s=100, c=colors[fam], marker='o', alpha=0.85, zorder=5)
    ax2.set_xlabel('Persona fidelity (rho)', fontsize=13)
    ax2.set_ylabel("Cohen's d", fontsize=13)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    plt.tight_layout()
    fig.savefig(os.path.join(output_dir, 'fig_combined.pdf'), bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  Figures saved to {output_dir}/")


# ================================================================
# JSON ENCODER
# ================================================================

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# ================================================================
# MAIN
# ================================================================

def main():
    parser = argparse.ArgumentParser(description='Persona Collapse Analysis Pipeline')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing digital_twin_* and moral_reasoning_* folders')
    parser.add_argument('--selfintro_dir', type=str, default=None,
                        help='Directory containing introductions_*.jsonl files')
    parser.add_argument('--human_ref', type=str, default='./data/human_reference/wave_1_numbers.csv',
                        help='Path to wave_1_numbers.csv (human BFI reference)')
    parser.add_argument('--flagged', type=str, default=None,
                        help='Path to flagged_personas.pkl')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='Output directory')
    parser.add_argument('--skip_selfintro', action='store_true',
                        help='Skip self-introduction analysis')
    parser.add_argument('--skip_figures', action='store_true',
                        help='Skip figure generation')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'tables'), exist_ok=True)

    # Load flagged personas
    flagged = None
    if args.flagged and os.path.exists(args.flagged):
        with open(args.flagged, 'rb') as f:
            flagged = pickle.load(f)
        print(f"Loaded {len(flagged)} flagged persona IDs")

    # ============================================================
    # HUMAN REFERENCE
    # ============================================================
    print("Loading human reference...")
    human = pd.read_csv(args.human_ref).iloc[1:]
    hr = human[BFI_COLS_HUMAN].apply(pd.to_numeric, errors='coerce').dropna()
    hr.columns = BFI_COLS
    Xh = hr.values.astype(float)
    hsc = StandardScaler().fit(Xh)
    Xhs = hsc.transform(Xh)
    hsub = Xhs[RNG.choice(len(Xhs), min(500, len(Xhs)), replace=False)]

    # Human metrics
    ev_h = PCA().fit(Xhs).explained_variance_ratio_
    effs_h = [1.0 / (hr[c].value_counts(normalize=True)**2).sum() for c in BFI_COLS]
    lids_h = compute_lid(hsub)
    lid_h = float(np.nanmedian(lids_h[np.isfinite(lids_h)]))
    Dh = euclidean_distances(hsub)
    np.fill_diagonal(Dh, np.inf)
    d_s, c_s = compute_density_coverage(hsub, hsub, k=5)
    human_metrics = {
        'n': len(hr), 'eff': round(np.mean(effs_h), 2),
        'pr': round((ev_h.sum()**2) / (ev_h**2).sum(), 1),
        'lid': round(lid_h, 1),
        'hop': round(compute_hopkins(hsub), 3),
        'sep': round(float(Dh.min(axis=1).mean()), 2),
        'cov': round(c_s, 3), 'den': round(d_s, 3),
    }
    print(f"  Human: n={human_metrics['n']}, EffL={human_metrics['eff']}, LID={human_metrics['lid']}")

    # ============================================================
    # BFI + MORAL ANALYSIS
    # ============================================================
    bfi_res = {}
    moral_res = {}

    for short, bfi_folder, moral_folder, si_name, category in MODEL_REGISTRY:
        # BFI
        print(f"\nAnalyzing {short}...")
        r = load_and_analyze(args.data_dir, bfi_folder, BFI_COLS, 'bfi', hsub, hsc, flagged)
        if r:
            bfi_res[short] = r
            print(f"  BFI: n={r['n']} EffL={r['eff']} Cov={r.get('cov')} rho={r.get('fid')}")
        else:
            print(f"  BFI: SKIP")

        # Moral
        r2 = load_and_analyze(args.data_dir, moral_folder, SCEN_COLS, 'moral', flagged=flagged)
        if r2:
            moral_res[short] = r2
            print(f"  Moral: n={r2['n']} EffL={r2['eff']} eta2={r2['eta_m']}")
        else:
            print(f"  Moral: SKIP")

    # Save
    analysis = {'human': human_metrics, 'bfi': bfi_res, 'moral': moral_res}
    with open(os.path.join(args.output_dir, 'analysis.json'), 'w') as f:
        json.dump(analysis, f, indent=2, cls=NpEncoder)
    print(f"\nSaved analysis.json: BFI={len(bfi_res)} models, Moral={len(moral_res)} models")

    # ============================================================
    # SELF-INTRODUCTION ANALYSIS
    # ============================================================
    if not args.skip_selfintro and args.selfintro_dir:
        print("\n" + "=" * 60)
        print("SELF-INTRODUCTION ANALYSIS")
        print("=" * 60)
        si_models = [si_name for _, _, _, si_name, _ in MODEL_REGISTRY if si_name]
        analyze_selfintro(args.selfintro_dir, si_models,
                          os.path.join(args.output_dir, 'selfintro'))

    # ============================================================
    # FIGURES
    # ============================================================
    if not args.skip_figures:
        print("\n" + "=" * 60)
        print("GENERATING FIGURES")
        print("=" * 60)
        generate_figures(analysis, os.path.join(args.output_dir, 'figures'))

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"Results: {args.output_dir}/")
    print(f"  analysis.json          - BFI + Moral metrics")
    print(f"  selfintro/             - Self-introduction results")
    print(f"  figures/               - PDF figures")


if __name__ == '__main__':
    main()
