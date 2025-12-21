"""
Strict SPICE wrapper using pycocoevalcap implementation.

This module delegates to pycocoevalcap.spice.Spice for standard SPICE scores.
Note: SPICE requires the parsing models and may need additional dependencies.
"""
from typing import List, Dict, Any


def _to_coco_format(candidates, references):
    if isinstance(candidates, dict) and isinstance(references, dict):
        return references, candidates

    gts = {}
    res = {}
    for i, cand_tokens in enumerate(candidates):
        pred_str = cand_tokens if isinstance(cand_tokens, str) else " ".join(cand_tokens)
        res[i] = [pred_str]

        ref_lists = references[i]
        refs_strs = []
        for ref in ref_lists:
            refs_strs.append(ref if isinstance(ref, str) else " ".join(ref))
        gts[i] = refs_strs

    return gts, res


def compute_spice(candidates, references) -> float:
    """
    Compute SPICE using pycocoevalcap.Spice.
    Returns the overall SPICE score (float).
    Raises ImportError with install instructions if dependency missing.
    """
    try:
        from pycocoevalcap.spice.spice import Spice
    except Exception as e:
        raise ImportError(
            "pycocoevalcap is required for strict SPICE. "
            "Install via `pip install git+https://github.com/salaniz/pycocoevalcap.git` "
            "and ensure required NLP resources are available."
        ) from e

    gts, res = _to_coco_format(candidates, references)

    spice = Spice()
    score, _scores = spice.compute_score(gts, res)
    return float(score)


