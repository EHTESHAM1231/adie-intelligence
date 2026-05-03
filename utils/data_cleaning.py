"""
ADIE — Data Cleaning Compatibility Shim
=========================================

This module previously contained the legacy column-dropping cleaning pipeline.
It has been replaced by the Adaptive Data Preparation Engine.

ALL calls are now redirected to utils.adaptive_cleaning which guarantees:
  1. NO COLUMN IS EVER DROPPED
  2. All transformations are traceable
  3. Domain context drives decisions
  4. Feature importance protects critical columns

The original destructive code has been archived in data_cleaning_legacy.py.
"""

# Re-export everything from the adaptive module so any existing
# `from utils.data_cleaning import clean_dataset` keeps working.
from utils.adaptive_cleaning import (
    clean_dataset,
    clean_dataset_adaptive,
    get_adaptive_diagnostics,
)

__all__ = ['clean_dataset', 'clean_dataset_adaptive', 'get_adaptive_diagnostics']
