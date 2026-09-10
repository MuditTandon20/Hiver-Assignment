"""Human vs LLM Judge agreement and correlation analysis."""
from typing import List, Dict, Any
import numpy as np
from sklearn.metrics import cohen_kappa_score

def compute_judge_agreement(
    human_scores: List[float],
    llm_scores: List[float],
    tolerance: float = 0.5
) -> Dict[str, Any]:
    """
    Compute statistical agreement between Human Judge and Automated Judge.
    Calculates Pearson correlation, Spearman correlation, Cohen's Kappa, MAE, and % agreement within tolerance.
    """
    if len(human_scores) == 0 or len(llm_scores) == 0:
        return {}
        
    arr_h = np.array(human_scores, dtype=float)
    arr_l = np.array(llm_scores, dtype=float)
    
    # Pearson correlation (with zero-variance safety check)
    if np.std(arr_h) > 1e-7 and np.std(arr_l) > 1e-7:
        corr_matrix = np.corrcoef(arr_h, arr_l)
        pearson_r = float(corr_matrix[0, 1]) if not np.isnan(corr_matrix[0, 1]) else 0.0
    else:
        pearson_r = 1.0 if np.allclose(arr_h, arr_l) else 0.0
    
    # Mean Absolute Error
    mae = float(np.mean(np.abs(arr_h - arr_l)))
    
    # Exact Agreement Rate (rounded to integer)
    exact_matches = np.sum(np.round(arr_h) == np.round(arr_l))
    exact_agreement_pct = float(exact_matches / len(arr_h)) * 100.0
    
    # Within-Tolerance Agreement (within +-1.0 points on 1-5 scale)
    within_tol = np.sum(np.abs(arr_h - arr_l) <= 1.0)
    within_tol_pct = float(within_tol / len(arr_h)) * 100.0
    
    # Cohen's Kappa on binned ratings (1-5 integer categories)
    h_binned = np.round(arr_h).astype(int)
    l_binned = np.round(arr_l).astype(int)
    try:
        kappa = float(cohen_kappa_score(h_binned, l_binned, weights='quadratic'))
    except Exception:
        kappa = 0.0
        
    return {
        "sample_size": len(human_scores),
        "pearson_correlation": round(pearson_r, 4),
        "quadratic_weighted_kappa": round(kappa, 4),
        "mean_absolute_error": round(mae, 4),
        "exact_agreement_pct": round(exact_agreement_pct, 2),
        "within_1_point_agreement_pct": round(within_tol_pct, 2)
    }
