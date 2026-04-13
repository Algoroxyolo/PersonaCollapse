"""Analysis metrics: data loading, matrix construction, and statistical measures."""

import json
import re

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_results(results_file="results.json"):
    """Load experiment results from a JSON file."""
    with open(results_file, "r") as f:
        results = json.load(f)
    return results


def parse_profile(profile):
    """Parse a profile dict or description string into a flat dimension dict.

    Handles three cases:
    * dict with ``persona_attributes`` key — returns that sub-dict.
    * plain dict — strips bookkeeping keys (``description``, ``prompt``,
      ``TWIN_ID``) and returns the rest.
    * string — regex-parses known dimension patterns.
    """
    if isinstance(profile, dict):
        if "persona_attributes" in profile:
            return profile["persona_attributes"]
        return {
            k: v for k, v in profile.items()
            if k not in ("description", "prompt", "TWIN_ID")
        }

    if isinstance(profile, str):
        profile_dict = {}
        patterns = {
            "age": r"Age: (\d+)",
            "gender": r"Gender: (\w+(?:-\w+)?)",
            "country": r"Country of Residence: ([^\n]+)",
            "social_class": r"Social Class: ([^\n]+)",
            "political_ideology": r"Political Ideology: ([^\n]+)",
        }
        for dimension, pattern in patterns.items():
            match = re.search(pattern, profile)
            if match:
                value = match.group(1).strip()
                if dimension == "country":
                    value = value.title()
                elif dimension == "gender":
                    value = value.lower()
                profile_dict[dimension] = value
        return profile_dict

    return {}


def process_results_to_matrix(results):
    """Convert a results list to a ``(n_personas, n_scenarios)`` score matrix.

    Returns:
        ``(matrix, personas_list, scenarios_list)`` where *matrix* is a
        NumPy array (NaN for missing entries).
    """
    persona_map = {}
    scenario_map = {}
    personas_list = []
    scenarios_list = []

    for result in results:
        profile = result.get("profile", {})
        scenario = result.get("scenario")

        profile_dict = parse_profile(profile)
        profile_key = tuple(sorted(profile_dict.items()))

        if profile_key not in persona_map:
            persona_map[profile_key] = len(personas_list)
            personas_list.append(profile_dict)

        if scenario not in scenario_map:
            scenario_map[scenario] = len(scenarios_list)
            scenarios_list.append(scenario)

    n_personas = len(personas_list)
    n_scenarios = len(scenarios_list)

    print(f"Found {n_personas} unique personas and {n_scenarios} unique scenarios")

    matrix = np.full((n_personas, n_scenarios), np.nan)

    for result in results:
        profile = result.get("profile", {})
        scenario = result.get("scenario")
        score = result.get("score")

        if score is None:
            continue

        profile_dict = parse_profile(profile)
        profile_key = tuple(sorted(profile_dict.items()))

        if profile_key in persona_map and scenario in scenario_map:
            p_idx = persona_map[profile_key]
            s_idx = scenario_map[scenario]
            matrix[p_idx, s_idx] = score

    return matrix, personas_list, scenarios_list


def load_scenarios(filename="claude-3-5-sonnet_AB_0_with_logprobs.jsonl"):
    """Load moral-dilemma scenarios from a JSONL file.

    Returns a deduplicated list of scenario strings.
    """
    scenarios = []
    try:
        with open(filename, "r") as f:
            for line in f:
                data = json.loads(line)
                scenarios.append(data["scenario"])
        scenarios = list(set(scenarios))
        print(f"Loaded {len(scenarios)} unique scenarios")
    except FileNotFoundError:
        print(f"Warning: Scenario file {filename} not found.")
        return []
    return scenarios


# ---------------------------------------------------------------------------
# Statistical metrics
# ---------------------------------------------------------------------------


def compute_hopkins_statistic(X, m=None, d=None):
    """Compute the Hopkins statistic for clustering tendency.

    Uses a log-sum-exp formulation for numerical stability when the
    exponent *d* is large.

    Args:
        X: ``(n_samples, n_features)`` array.
        m: number of test points (default 10 % of *n*).
        d: distance exponent (default ``n_features``).
    """
    from sklearn.neighbors import NearestNeighbors

    n, features = X.shape
    if m is None:
        m = max(1, int(0.1 * n))

    if m == 0 or n < 2:
        return 0.5

    if d is None:
        d = features

    mins = np.min(X, axis=0)
    maxs = np.max(X, axis=0)

    Y = np.random.uniform(mins, maxs, (m, features))

    indices = np.random.choice(n, m, replace=False)
    X_tilde = X[indices]

    nbrs = NearestNeighbors(n_neighbors=2).fit(X)

    u_dist, _ = nbrs.kneighbors(Y, n_neighbors=1)
    u = u_dist.flatten()

    w_dist, _ = nbrs.kneighbors(X_tilde, n_neighbors=2)
    w = w_dist[:, 1]

    try:
        u = np.maximum(u, 1e-10)
        w = np.maximum(w, 1e-10)

        log_u = d * np.log(u)
        log_w = d * np.log(w)

        max_log_u = np.max(log_u)
        log_sum_u = max_log_u + np.log(np.sum(np.exp(log_u - max_log_u)))

        max_log_w = np.max(log_w)
        log_sum_w = max_log_w + np.log(np.sum(np.exp(log_w - max_log_w)))

        diff = log_sum_w - log_sum_u
        if diff > 700:
            H = 0.0
        elif diff < -700:
            H = 1.0
        else:
            H = 1.0 / (1.0 + np.exp(diff))

    except Exception as e:
        print(f"Warning: Error computing Hopkins statistic with d={d}: {e}. Fallback to d=1.")
        sum_u = np.sum(u)
        sum_w = np.sum(w)
        if sum_u + sum_w == 0:
            H = 0.5
        else:
            H = sum_u / (sum_u + sum_w)

    return H


def compute_hopkins_with_confidence(X, iterations=100, m=None, d=None):
    """Compute Hopkins statistic repeatedly and return mean with 95 % CI.

    Returns:
        ``(mean_h, ci, scores)``
    """
    scores = []
    for _ in range(iterations):
        h = compute_hopkins_statistic(X, m, d)
        scores.append(h)

    mean_h = np.mean(scores)
    std_h = np.std(scores)
    ci = 1.96 * std_h / np.sqrt(iterations)

    return mean_h, ci, scores


def compute_prdc_metric(real_features, fake_features, nearest_k=5):
    """Compute Precision, Recall, Density, and Coverage (PRDC).

    Requires the ``prdc`` package.
    """
    try:
        from prdc import compute_prdc
    except ImportError:
        print("Warning: 'prdc' package not found. Skipping D&C metrics.")
        print("To install: pip install prdc")
        return {"density": None, "coverage": None}

    try:
        metrics = compute_prdc(
            real_features=real_features,
            fake_features=fake_features,
            nearest_k=nearest_k,
        )
        return metrics
    except Exception as e:
        print(f"Error computing PRDC: {e}")
        return {"density": None, "coverage": None}


def compute_hyperspherical_uniformity(features, t=2.0):
    """Compute the hyperspherical-uniformity metric via Gaussian potential.

    Requires ``torch``.
    """
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("Warning: 'torch' package not found. Skipping Hyperspherical Uniformity metric.")
        return None

    try:
        if isinstance(features, np.ndarray):
            features_tensor = torch.from_numpy(features).float()
        else:
            features_tensor = features.float()

        features_normalized = F.normalize(features_tensor, p=2, dim=1)

        loss = (
            torch.pdist(features_normalized, p=2)
            .pow(2)
            .mul(-t)
            .exp()
            .mean()
            .log()
        )

        return loss.item()
    except Exception as e:
        print(f"Error computing Hyperspherical Uniformity: {e}")
        return None
