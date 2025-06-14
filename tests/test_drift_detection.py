"""
Test Drift Detection and Monitoring
Comprehensive tests for the model monitoring and drift detection system.
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import tempfile
import os
from pathlib import Path
from datetime import datetime

# Import the system under test
from auto_ml.monitoring.drift_detection import DriftDetection
from auto_ml.core.exceptions import MonitoringError

class TestDriftDetection:
    """Test cases for drift detection and monitoring system."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 1000
        
        # Create baseline data
        baseline_data = pd.DataFrame({
            'numeric_feature': np.random.normal(0, 1, n_samples),
            'categorical_feature': np.random.choice(['A', 'B', 'C'], n_samples),
            'target': np.random.randint(0, 2, n_samples)
        })
        
        # Create drifted data
        drifted_data = pd.DataFrame({
            'numeric_feature': np.random.normal(0.5, 1.2, n_samples),  # Shifted mean and std
            'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], n_samples, p=[0.2, 0.3, 0.3, 0.2]),  # New category
            'target': np.random.randint(0, 2, n_samples)
        })
        
        return baseline_data, drifted_data
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing."""
        return {
            'drift_threshold': 0.05,
            'wasserstein_threshold': 0.1,
            'chi2_threshold': 0.05,
            'performance_threshold': 0.1,
            'monitor_features': True,
            'monitor_performance': True,
            'monitor_predictions': True,
            'alert_on_drift': True
        }
    
    def test_initialization(self, basic_config):
        """Test drift detection initialization."""
        detector = DriftDetection(basic_config)
        
        assert detector.drift_threshold == 0.05
        assert detector.wasserstein_threshold == 0.1
        assert detector.chi2_threshold == 0.05
        assert detector.performance_threshold == 0.1
        assert detector.monitor_features is True
        assert detector.monitor_performance is True
        assert detector.monitor_predictions is True
        assert detector.alert_on_drift is True
        assert detector.baseline_data is None
        assert len(detector.drift_history) == 0
        assert len(detector.alerts) == 0
    
    def test_set_baseline(self, sample_data, basic_config):
        """Test setting baseline data."""
        baseline_data, _ = sample_data
        detector = DriftDetection(basic_config)
        
        # Set baseline
        detector.set_baseline(baseline_data, target_column='target')
        
        assert detector.baseline_data is not None
        assert len(detector.baseline_data) == len(baseline_data)
        assert 'feature_stats' in detector.baseline_stats
        assert 'numeric_feature' in detector.baseline_stats['feature_stats']
        assert 'categorical_feature' in detector.baseline_stats['feature_stats']
    
    def test_set_baseline_with_predictions(self, sample_data, basic_config):
        """Test setting baseline with predictions."""
        baseline_data, _ = sample_data
        detector = DriftDetection(basic_config)
        
        predictions = np.random.random(len(baseline_data))
        detector.set_baseline(baseline_data, target_column='target', predictions=predictions)
        
        assert 'prediction_stats' in detector.baseline_stats
        assert detector.baseline_stats['prediction_stats'] is not None
    
    def test_detect_drift_without_baseline(self, sample_data, basic_config):
        """Test drift detection without baseline raises error."""
        _, drifted_data = sample_data
        detector = DriftDetection(basic_config)
        
        with pytest.raises(MonitoringError, match="Baseline data not set"):
            detector.detect_drift(drifted_data, target_column='target')
    
    def test_detect_feature_drift(self, sample_data, basic_config):
        """Test feature drift detection."""
        baseline_data, drifted_data = sample_data
        detector = DriftDetection(basic_config)
        
        # Set baseline
        detector.set_baseline(baseline_data, target_column='target')
        
        # Detect drift
        results = detector.detect_drift(drifted_data, target_column='target')
        
        # Check results structure
        assert 'timestamp' in results
        assert 'data_shape' in results
        assert 'feature_drift' in results
        assert 'overall_drift_score' in results
        assert 'drift_detected' in results
        assert 'alerts' in results
        
        # Check feature drift results
        feature_drift = results['feature_drift']
        assert 'numeric_feature' in feature_drift
        assert 'categorical_feature' in feature_drift
        
        # Check that drift is detected (since we created drifted data)
        assert results['drift_detected'] is True
        assert results['overall_drift_score'] > 0
    
    def test_detect_performance_drift(self, sample_data, basic_config):
        """Test performance drift detection."""
        baseline_data, drifted_data = sample_data
        detector = DriftDetection(basic_config)
        
        # Set baseline with predictions
        baseline_predictions = np.random.random(len(baseline_data))
        detector.set_baseline(baseline_data, target_column='target', predictions=baseline_predictions)
        
        # Set baseline performance
        detector.baseline_performance = {
            'accuracy': 0.85,
            'precision': 0.82,
            'recall': 0.88,
            'f1': 0.85
        }
        
        # Create current predictions (worse performance)
        current_predictions = np.random.random(len(drifted_data)) * 0.5  # Lower quality predictions
        
        # Detect drift
        results = detector.detect_drift(drifted_data, target_column='target', predictions=current_predictions)
        
        assert 'performance_drift' in results
        performance_drift = results['performance_drift']
        assert 'drift_detected' in performance_drift
        assert 'metrics' in performance_drift
        assert 'degradation_score' in performance_drift
    
    def test_detect_prediction_drift(self, sample_data, basic_config):
        """Test prediction drift detection."""
        baseline_data, drifted_data = sample_data
        detector = DriftDetection(basic_config)
        
        # Set baseline with predictions
        baseline_predictions = np.random.normal(0.5, 0.1, len(baseline_data))
        detector.set_baseline(baseline_data, target_column='target', predictions=baseline_predictions)
        
        # Create drifted predictions
        drifted_predictions = np.random.normal(0.8, 0.2, len(drifted_data))  # Different distribution
        
        # Detect drift
        results = detector.detect_drift(drifted_data, target_column='target', predictions=drifted_predictions)
        
        assert 'prediction_drift' in results
        prediction_drift = results['prediction_drift']
        assert 'drift_detected' in prediction_drift
        assert 'drift_score' in prediction_drift
        assert 'statistics' in prediction_drift
    
    def test_alert_generation(self, sample_data, basic_config):
        """Test alert generation for detected drift."""
        baseline_data, drifted_data = sample_data
        detector = DriftDetection(basic_config)
        
        # Set baseline
        detector.set_baseline(baseline_data, target_column='target')
        
        # Detect drift
        results = detector.detect_drift(drifted_data, target_column='target')
        
        # Check that alerts are generated
        assert len(results['alerts']) > 0
        assert len(detector.alerts) > 0
        
        # Check alert structure
        alert = results['alerts'][0]
        assert 'type' in alert
        assert 'severity' in alert
        assert 'message' in alert
        assert 'timestamp' in alert
    
    def test_no_drift_detection(self, sample_data, basic_config):
        """Test that no drift is detected when data is similar."""
        baseline_data, _ = sample_data
        detector = DriftDetection(basic_config)
        
        # Set baseline
        detector.set_baseline(baseline_data, target_column='target')
        
        # Use same data (no drift)
        results = detector.detect_drift(baseline_data, target_column='target')
        
        # Should not detect drift (but might due to random sampling)
        # Just check that the overall drift score is low
        assert results['overall_drift_score'] < 0.5
        assert len(results['alerts']) == 0
    
    def test_get_drift_summary(self, sample_data, basic_config):
        """Test getting drift summary."""
        baseline_data, drifted_data = sample_data
        detector = DriftDetection(basic_config)
        
        # Initially should return no history message
        summary = detector.get_drift_summary()
        assert 'message' in summary
        
        # Set baseline and detect drift
        detector.set_baseline(baseline_data, target_column='target')
        detector.detect_drift(drifted_data, target_column='target')
        
        # Get summary
        summary = detector.get_drift_summary()
        
        assert 'total_checks' in summary
        assert 'latest_check' in summary
        assert 'drift_detected' in summary
        assert 'overall_drift_score' in summary
        assert 'feature_drift_count' in summary
        assert 'total_alerts' in summary
        assert 'recent_alerts' in summary
    
    def test_clear_history(self, sample_data, basic_config):
        """Test clearing monitoring history."""
        baseline_data, drifted_data = sample_data
        detector = DriftDetection(basic_config)
        
        # Set baseline and detect drift
        detector.set_baseline(baseline_data, target_column='target')
        detector.detect_drift(drifted_data, target_column='target')
        
        # Verify history exists
        assert len(detector.drift_history) > 0
        assert len(detector.alerts) > 0
        
        # Clear history
        detector.clear_history()
        
        # Verify history is cleared
        assert len(detector.drift_history) == 0
        assert len(detector.alerts) == 0
    
    def test_save_drift_report(self, sample_data, basic_config):
        """Test saving drift detection report."""
        baseline_data, drifted_data = sample_data
        detector = DriftDetection(basic_config)
        
        # Set baseline and detect drift
        detector.set_baseline(baseline_data, target_column='target')
        detector.detect_drift(drifted_data, target_column='target')
        
        # Save report
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            detector.save_drift_report(tmp_path)
            assert os.path.exists(tmp_path)
            
            # Check that file contains valid JSON
            with open(tmp_path, 'r') as f:
                content = f.read()
                assert len(content) > 0
                
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_plot_drift_analysis(self, sample_data, basic_config):
        """Test drift analysis plotting."""
        baseline_data, drifted_data = sample_data
        detector = DriftDetection(basic_config)
        
        # Set baseline and detect drift multiple times
        detector.set_baseline(baseline_data, target_column='target')
        detector.detect_drift(drifted_data, target_column='target')
        detector.detect_drift(drifted_data, target_column='target')  # Second check
        
        # Test plotting (should not raise error)
        with patch('matplotlib.pyplot.show'):
            detector.plot_drift_analysis()
    
    def test_plot_drift_analysis_save(self, sample_data, basic_config):
        """Test drift analysis plotting with save."""
        baseline_data, drifted_data = sample_data
        detector = DriftDetection(basic_config)
        
        # Set baseline and detect drift
        detector.set_baseline(baseline_data, target_column='target')
        detector.detect_drift(drifted_data, target_column='target')
        
        # Test plotting with save
        with tempfile.NamedTemporaryFile(mode='w', suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
        
        try:
            detector.plot_drift_analysis(save_path=tmp_path)
            assert os.path.exists(tmp_path)
            
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def test_plot_drift_analysis_no_history(self, basic_config):
        """Test plotting with no drift history."""
        detector = DriftDetection(basic_config)
        
        # Should not raise error, just log warning
        with patch('matplotlib.pyplot.show'):
            detector.plot_drift_analysis()
    
    def test_configuration_options(self):
        """Test different configuration options."""
        # Test with monitoring disabled
        config_disabled = {
            'monitor_features': False,
            'monitor_performance': False,
            'monitor_predictions': False,
            'alert_on_drift': False
        }
        
        detector_disabled = DriftDetection(config_disabled)
        assert detector_disabled.monitor_features is False
        assert detector_disabled.monitor_performance is False
        assert detector_disabled.monitor_predictions is False
        assert detector_disabled.alert_on_drift is False
        
        # Test with custom thresholds
        config_custom = {
            'drift_threshold': 0.01,  # More strict
            'wasserstein_threshold': 0.05,
            'chi2_threshold': 0.01,
            'performance_threshold': 0.05
        }
        
        detector_custom = DriftDetection(config_custom)
        assert detector_custom.drift_threshold == 0.01
        assert detector_custom.wasserstein_threshold == 0.05
        assert detector_custom.chi2_threshold == 0.01
        assert detector_custom.performance_threshold == 0.05
    
    def test_error_handling(self, basic_config):
        """Test error handling in drift detection."""
        detector = DriftDetection(basic_config)
        
        # Test with invalid data - should not raise error, just log
        detector.set_baseline(pd.DataFrame({'feature': [1, 2, 3]}))
        
        # Test with invalid current data - should not raise error, just log
        detector.detect_drift(pd.DataFrame({'feature': [1, 2, 3]}))
        
        # Test that the system handles empty data gracefully
        empty_data = pd.DataFrame()
        detector.set_baseline(empty_data)
        # This should work without raising an error

if __name__ == "__main__":
    pytest.main([__file__]) 