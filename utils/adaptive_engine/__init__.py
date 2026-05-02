"""
ADIE — Context-Aware Adaptive Data Preparation Engine
======================================================

A fully intelligent, domain-aware, non-destructive data preparation system
that replaces static rule-based cleaning with adaptive, context-sensitive
transformations.

CORE PRINCIPLES:
1. NO COLUMN IS EVER DROPPED
2. All transformations are reversible or traceable
3. Domain context drives all decisions
4. Feature importance protects critical columns
5. Dataset type determines strategy

Modules:
- profiler: Comprehensive dataset profiling
- detector: Dataset type and domain detection
- classifier: Column role classification
- importance: Feature importance pre-check
- protector: Column protection system
- transformer: Adaptive transformation engine
- engine: Main orchestration engine
"""

from .engine import AdaptiveDataPreparationEngine
from .profiler import DatasetProfiler
from .detector import DatasetTypeDetector, DomainDetector
from .classifier import ColumnRoleClassifier
from .importance import FeatureImportanceAnalyzer
from .protector import ColumnProtectionSystem
from .transformer import AdaptiveTransformer

__all__ = [
    'AdaptiveDataPreparationEngine',
    'DatasetProfiler',
    'DatasetTypeDetector',
    'DomainDetector',
    'ColumnRoleClassifier',
    'FeatureImportanceAnalyzer',
    'ColumnProtectionSystem',
    'AdaptiveTransformer'
]

__version__ = '2.0.0'
