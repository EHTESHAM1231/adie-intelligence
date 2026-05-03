"""
ADIE — LEGACY Data Cleaning Module (DEPRECATED)
=================================================

⚠️  THIS MODULE IS DEPRECATED AND MUST NOT BE USED.
    All cleaning must go through utils.adaptive_cleaning.

    This file is retained ONLY as an archive reference.
    Any import will raise a RuntimeError.
"""

raise RuntimeError(
    "Legacy cleaning module (data_cleaning_legacy.py) is DEPRECATED. "
    "Use 'from utils.adaptive_cleaning import clean_dataset' instead. "
    "The legacy module dropped columns, which violates ADIE v3 guarantees."
)
