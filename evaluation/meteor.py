"""
Strict METEOR wrapper using pycocoevalcap implementation.

This module delegates to pycocoevalcap.meteor.Meteor to ensure strict,
standard METEOR scores consistent with COCO evaluation.
"""
from typing import List, Dict, Any


def _to_coco_format(candidates, references):
    """
    Convert inputs to COCO-like dict format:
      gts: {idx: [ref_str, ...]}
      res: {idx: [pred_str]}
    Accepts either:
      - candidates: List[List[str]] (tokens) and references: List[List[List[str]]]
      - or dicts: candidates={i: [pred_str]}, references={i: [ref_str,...]}
    """
    if isinstance(candidates, dict) and isinstance(references, dict):
        # assume already in COCO-like string lists
        return references, candidates

    gts = {}
    res = {}
    for i, cand_tokens in enumerate(candidates):
        # candidate may already be a string or token list
        pred_str = cand_tokens if isinstance(cand_tokens, str) else " ".join(cand_tokens)
        res[i] = [pred_str]

        ref_lists = references[i]
        refs_strs = []
        for ref in ref_lists:
            refs_strs.append(ref if isinstance(ref, str) else " ".join(ref))
        gts[i] = refs_strs

    return gts, res


def compute_meteor(candidates, references) -> float:
    """
    Compute METEOR using pycocoevalcap.Meteor.
    Returns the average METEOR score (float).
    Raises ImportError with instructions if pycocoevalcap is not available.
    """
    try:
        from pycocoevalcap.meteor.meteor import Meteor
    except Exception as e:
        raise ImportError(
            "pycocoevalcap is required for strict METEOR. "
            "Install via `pip install git+https://github.com/salaniz/pycocoevalcap.git` "
            "and ensure NLTK punkt resource is available."
        ) from e

    gts, res = _to_coco_format(candidates, references)

    meteor = Meteor()
    score, _scores = meteor.compute_score(gts, res)
    # compute_score returns (overall_score, score_list)
    return float(score)

