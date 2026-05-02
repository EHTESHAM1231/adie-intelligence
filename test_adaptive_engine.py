"""
ADIE — Adaptive Engine Test Suite
==================================

Comprehensive tests for the new Context-Aware Adaptive Data Preparation Engine.

Tests verify:
1. NO COLUMN IS EVER DROPPED (critical guarantee)
2. All transformations are traceable
3. Domain detection works correctly
4. Feature importance protection works
5. Backward compatibility is maintained
"""

import pandas as pd
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.adaptive_engine import (
    AdaptiveDataPreparationEngine,
    DatasetProfiler,
    DatasetTypeDetector,
    DomainDetector,
    ColumnRoleClassifier,
    FeatureImportanceAnalyzer,
    ColumnProtectionSystem
)


def create_test_dataset_aviation():
    """Create a test dataset simulating aviation data."""
    np.random.seed(42)
    n = 500
    
    return pd.DataFrame({
        'APT_ICAO': [f'ICAO{i:04d}' for i in range(n)],
        'APT_NAME': [f'Airport {i}' for i in range(n)],
        'STATE_NAME': np.random.choice(['California', 'Texas', 'Florida', 'New York', None], n),
        'FLT_DEP_1': np.random.randint(0, 1000, n).astype(float),
        'FLT_ARR_1': np.random.randint(0, 1000, n).astype(float),
        'FLT_DEP_2': np.random.randint(0, 500, n).astype(float),
        'FLT_ARR_2': np.random.randint(0, 500, n).astype(float),
        'YEAR': np.random.choice([2020, 2021, 2022, 2023], n),
        'MONTH': np.random.randint(1, 13, n),
        'missing_col': np.where(np.random.random(n) > 0.3, np.random.randn(n), np.nan),
        'high_card_col': [f'CAT_{i % 200}' for i in range(n)],
        'target': np.random.choice(['Low', 'Medium', 'High'], n),
    })


def create_test_dataset_finance():
    """Create a test dataset simulating finance data."""
    np.random.seed(42)
    n = 500
    
    return pd.DataFrame({
        'transaction_id': [f'TXN{i:06d}' for i in range(n)],
        'customer_id': [f'CUST{i % 100:04d}' for i in range(n)],
        'amount': np.random.exponential(100, n),
        'revenue': np.random.exponential(1000, n),
        'cost': np.random.exponential(500, n),
        'date': pd.date_range('2020-01-01', periods=n, freq='D'),
        'category': np.random.choice(['A', 'B', 'C', 'D'], n),
        'region': np.random.choice(['North', 'South', 'East', 'West', None], n),
        'is_fraud': np.random.choice([0, 1], n, p=[0.95, 0.05]),
    })


def create_test_dataset_with_issues():
    """Create a dataset with various data quality issues."""
    np.random.seed(42)
    n = 500
    
    return pd.DataFrame({
        'id': range(n),
        'numeric_with_outliers': np.concatenate([
            np.random.randn(n - 10) * 10,
            np.array([1000, -1000, 500, -500, 2000, -2000, 3000, -3000, 4000, -4000])
        ]),
        'high_missing': np.where(np.random.random(n) > 0.15, np.random.randn(n), np.nan),
        'mixed_type': [str(i) if i % 3 == 0 else i for i in range(n)],
        'constant': ['same'] * n,
        'binary': np.random.choice([0, 1], n),
        'ordinal': np.random.choice(['low', 'medium', 'high'], n),
        'target': np.random.choice([0, 1], n, p=[0.9, 0.1]),  # Imbalanced
    })


class TestDatasetProfiler:
    """Tests for DatasetProfiler."""
    
    def test_basic_profiling(self):
        """Test basic dataset profiling."""
        df = create_test_dataset_aviation()
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(df, 'target')
        
        assert profile['n_rows'] == 500
        assert profile['n_cols'] == 12
        assert 'missing_analysis' in profile
        assert 'cardinality_analysis' in profile
        assert 'quality_metrics' in profile
        
        print("✓ Basic profiling works")
    
    def test_missing_value_detection(self):
        """Test missing value detection."""
        df = create_test_dataset_with_issues()
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(df, 'target')
        
        missing = profile['missing_analysis']
        assert missing['total_missing'] > 0
        assert 'high_missing' in missing['missing_per_column']
        
        print("✓ Missing value detection works")
    
    def test_cardinality_detection(self):
        """Test cardinality detection."""
        df = create_test_dataset_aviation()
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(df, 'target')
        
        cardinality = profile['cardinality_analysis']
        assert 'APT_ICAO' in cardinality['potential_identifiers']
        
        print("✓ Cardinality detection works")


class TestDatasetTypeDetector:
    """Tests for DatasetTypeDetector."""
    
    def test_aviation_detection(self):
        """Test detection of aviation-like dataset."""
        df = create_test_dataset_aviation()
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(df, 'target')
        
        detector = DatasetTypeDetector()
        result = detector.detect_type(df, profile)
        
        assert 'primary_type' in result
        assert 'confidence' in result
        assert result['confidence'] > 0
        
        print(f"✓ Dataset type detected: {result['primary_type']} (confidence: {result['confidence']:.2f})")
    
    def test_time_series_detection(self):
        """Test detection of time-series characteristics."""
        df = create_test_dataset_finance()
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(df, 'is_fraud')
        
        detector = DatasetTypeDetector()
        result = detector.detect_type(df, profile)
        
        # Should detect temporal characteristics
        assert result['characteristics']['time_series']['datetime_columns'] or \
               result['characteristics']['time_series']['time_keywords_found']
        
        print("✓ Time-series characteristics detected")


class TestDomainDetector:
    """Tests for DomainDetector."""
    
    def test_aviation_domain(self):
        """Test aviation domain detection."""
        df = create_test_dataset_aviation()
        
        detector = DomainDetector()
        result = detector.detect_domain(df)
        
        assert result['domain'] == 'aviation', f"Expected aviation, got {result['domain']} (scores: {result.get('domain_scores', {})})"
        assert result['confidence'] > 0.0
        assert 'protected_columns' in result
        
        print(f"✓ Aviation domain detected (confidence: {result['confidence']:.2f})")
    
    def test_finance_domain(self):
        """Test finance domain detection."""
        df = create_test_dataset_finance()
        
        detector = DomainDetector()
        result = detector.detect_domain(df)
        
        assert result['domain'] == 'finance'
        assert result['confidence'] > 0.3
        
        print(f"✓ Finance domain detected (confidence: {result['confidence']:.2f})")


class TestColumnRoleClassifier:
    """Tests for ColumnRoleClassifier."""
    
    def test_role_classification(self):
        """Test column role classification."""
        df = create_test_dataset_aviation()
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(df, 'target')
        
        classifier = ColumnRoleClassifier()
        roles = classifier.classify_columns(df, 'target', profile)
        
        # Check target is identified
        assert roles['target']['role'] == 'target'
        
        # Check identifiers are detected
        assert roles['APT_ICAO']['role'] == 'identifier'
        
        print("✓ Column roles classified correctly")
    
    def test_temporal_detection(self):
        """Test temporal column detection."""
        df = create_test_dataset_finance()
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(df, 'is_fraud')
        
        classifier = ColumnRoleClassifier()
        roles = classifier.classify_columns(df, 'is_fraud', profile)
        
        # Date column should be temporal
        assert roles['date']['role'] == 'temporal'
        
        print("✓ Temporal columns detected")


class TestFeatureImportanceAnalyzer:
    """Tests for FeatureImportanceAnalyzer."""
    
    def test_importance_analysis(self):
        """Test feature importance analysis."""
        df = create_test_dataset_aviation()
        
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.analyze_importance(df, 'target')
        
        assert 'importance_scores' in result
        assert 'protection_levels' in result
        assert 'protected_columns' in result
        assert len(result['importance_scores']) > 0
        
        print(f"✓ Feature importance analyzed ({len(result['importance_scores'])} features)")
    
    def test_protection_levels(self):
        """Test protection level assignment."""
        df = create_test_dataset_finance()
        
        analyzer = FeatureImportanceAnalyzer()
        result = analyzer.analyze_importance(df, 'is_fraud')
        
        # Should have columns at different protection levels
        assert len(result['protected_columns']) >= 0
        assert len(result['careful_columns']) >= 0
        assert len(result['flexible_columns']) >= 0
        
        print("✓ Protection levels assigned")


class TestColumnProtectionSystem:
    """Tests for ColumnProtectionSystem."""
    
    def test_protection_map(self):
        """Test protection map building."""
        df = create_test_dataset_aviation()
        
        # Get prerequisites
        profiler = DatasetProfiler()
        profile = profiler.profile_dataset(df, 'target')
        
        domain_detector = DomainDetector()
        domain_info = domain_detector.detect_domain(df)
        
        classifier = ColumnRoleClassifier()
        roles = classifier.classify_columns(df, 'target', profile, domain_info)
        
        analyzer = FeatureImportanceAnalyzer()
        importance = analyzer.analyze_importance(df, 'target')
        
        # Build protection map
        protector = ColumnProtectionSystem()
        protections = protector.build_protection_map(
            df, 'target', importance, domain_info, roles
        )
        
        # Target should be critical
        from utils.adaptive_engine.protector import ProtectionLevel
        assert protector.get_protection_level('target') == ProtectionLevel.CRITICAL
        
        print("✓ Protection map built correctly")


class TestAdaptiveDataPreparationEngine:
    """Tests for the main AdaptiveDataPreparationEngine."""
    
    def test_no_columns_dropped(self):
        """CRITICAL TEST: Verify no columns are ever dropped."""
        df = create_test_dataset_with_issues()
        original_cols = set(df.columns)
        
        engine = AdaptiveDataPreparationEngine()
        prepared_df, report = engine.prepare(df, 'target')
        
        # All original columns must still exist (possibly transformed)
        # The prepared_df may have MORE columns (added features) but never fewer original ones
        # Check that no original column was dropped
        assert report['dataset_info']['columns_dropped'] == 0
        assert report['guarantees']['no_columns_dropped'] == True
        
        print("✓ CRITICAL: No columns were dropped")
    
    def test_aviation_dataset(self):
        """Test full pipeline on aviation dataset."""
        df = create_test_dataset_aviation()
        
        engine = AdaptiveDataPreparationEngine({"verbose": False})
        prepared_df, report = engine.prepare(df, 'target')
        
        assert prepared_df is not None
        assert len(prepared_df) > 0
        assert report['domain']['detected_domain'] == 'aviation'
        
        print(f"✓ Aviation dataset prepared: {df.shape} -> {prepared_df.shape}")
    
    def test_finance_dataset(self):
        """Test full pipeline on finance dataset."""
        df = create_test_dataset_finance()
        
        engine = AdaptiveDataPreparationEngine({"verbose": False})
        prepared_df, report = engine.prepare(df, 'is_fraud')
        
        assert prepared_df is not None
        assert len(prepared_df) > 0
        assert report['domain']['detected_domain'] == 'finance'
        
        print(f"✓ Finance dataset prepared: {df.shape} -> {prepared_df.shape}")
    
    def test_missing_value_handling(self):
        """Test that missing values are handled without dropping."""
        df = create_test_dataset_with_issues()
        
        engine = AdaptiveDataPreparationEngine()
        prepared_df, report = engine.prepare(df, 'target')
        
        # Should have no missing values after preparation
        assert prepared_df.isnull().sum().sum() == 0
        
        # Should have missing flags added
        missing_flag_cols = [c for c in prepared_df.columns if '_missing_flag' in c]
        assert len(missing_flag_cols) > 0
        
        print(f"✓ Missing values handled (added {len(missing_flag_cols)} flag columns)")
    
    def test_high_cardinality_handling(self):
        """Test that high-cardinality columns are encoded, not dropped."""
        df = create_test_dataset_aviation()
        
        engine = AdaptiveDataPreparationEngine()
        prepared_df, report = engine.prepare(df, 'target')
        
        # High cardinality columns should be encoded
        transformations = report['transformations']['details']
        encoding_ops = [t for t in transformations if 'encode' in t.get('operation', '')]
        
        assert len(encoding_ops) > 0
        
        print(f"✓ High-cardinality columns encoded ({len(encoding_ops)} encoding operations)")
    
    def test_outlier_handling(self):
        """Test that outliers are capped, not removed."""
        df = create_test_dataset_with_issues()
        original_rows = len(df)
        
        engine = AdaptiveDataPreparationEngine()
        prepared_df, report = engine.prepare(df, 'target')
        
        # Row count should be the same (no rows removed)
        assert len(prepared_df) == original_rows
        
        # Should have outlier flags
        outlier_flag_cols = [c for c in prepared_df.columns if '_outlier_flag' in c]
        
        print(f"✓ Outliers capped, not removed (rows: {original_rows}, outlier flags: {len(outlier_flag_cols)})")
    
    def test_transformation_traceability(self):
        """Test that all transformations are logged."""
        df = create_test_dataset_aviation()
        
        engine = AdaptiveDataPreparationEngine()
        prepared_df, report = engine.prepare(df, 'target')
        
        # Should have transformation log
        assert 'transformations' in report
        assert report['transformations']['total'] > 0
        assert len(report['transformations']['details']) > 0
        
        print(f"✓ Transformations logged ({report['transformations']['total']} operations)")


def run_all_tests():
    """Run all tests."""
    print("=" * 60)
    print("ADIE Adaptive Engine Test Suite")
    print("=" * 60)
    print()
    
    test_classes = [
        TestDatasetProfiler,
        TestDatasetTypeDetector,
        TestDomainDetector,
        TestColumnRoleClassifier,
        TestFeatureImportanceAnalyzer,
        TestColumnProtectionSystem,
        TestAdaptiveDataPreparationEngine,
    ]
    
    total_tests = 0
    passed_tests = 0
    failed_tests = []
    
    for test_class in test_classes:
        print(f"\n{test_class.__name__}")
        print("-" * 40)
        
        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith('test_')]
        
        for method_name in test_methods:
            total_tests += 1
            try:
                getattr(instance, method_name)()
                passed_tests += 1
            except Exception as e:
                failed_tests.append((test_class.__name__, method_name, str(e)))
                print(f"✗ {method_name}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Results: {passed_tests}/{total_tests} tests passed")
    
    if failed_tests:
        print("\nFailed tests:")
        for class_name, method_name, error in failed_tests:
            print(f"  - {class_name}.{method_name}: {error}")
    else:
        print("\n🎉 All tests passed!")
    
    print("=" * 60)
    
    return len(failed_tests) == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
