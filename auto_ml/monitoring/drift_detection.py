"""
Model Monitoring and Drift Detection
Production monitoring capabilities for ML models including data drift detection,
model performance monitoring, and alerting systems.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
import logging
from pathlib import Path
import json
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical libraries
from scipy import stats
from scipy.stats import ks_2samp, chi2_contingency, wasserstein_distance
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

from auto_ml.core.base_classes import BaseMonitoring
from auto_ml.core.exceptions import MonitoringError

logger = logging.getLogger(__name__)

class DriftDetection(BaseMonitoring):
    """
    Comprehensive model monitoring and drift detection system.
    
    Features:
    - Statistical drift detection (KS test, Chi-square, Wasserstein distance)
    - Feature distribution monitoring
    - Model performance degradation detection
    - Automated alerting system
    - Drift visualization and reporting
    - Historical drift tracking
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize drift detection system.
        
        Args:
            config (Optional[Dict[str, Any]]): Configuration dictionary
        """
        super().__init__(config)
        
        # Drift detection settings
        self.drift_threshold = config.get('drift_threshold', 0.05)  # p-value threshold
        self.wasserstein_threshold = config.get('wasserstein_threshold', 0.1)
        self.chi2_threshold = config.get('chi2_threshold', 0.05)
        self.performance_threshold = config.get('performance_threshold', 0.1)  # 10% degradation
        
        # Monitoring settings
        self.monitor_features = config.get('monitor_features', True)
        self.monitor_performance = config.get('monitor_performance', True)
        self.monitor_predictions = config.get('monitor_predictions', True)
        self.alert_on_drift = config.get('alert_on_drift', True)
        
        # Storage
        self.baseline_data = None
        self.baseline_stats = {}
        self.drift_history = []
        self.performance_history = []
        self.alerts = []
        
    def set_baseline(self, data: pd.DataFrame, target_column: Optional[str] = None,
                    predictions: Optional[np.ndarray] = None) -> None:
        """
        Set baseline data for drift detection.
        
        Args:
            data (pd.DataFrame): Baseline data
            target_column (Optional[str]): Target column name
            predictions (Optional[np.ndarray]): Model predictions on baseline data
        """
        logger.info("Setting baseline data for drift detection...")
        
        try:
            self.baseline_data = data.copy()
            
            # Calculate baseline statistics
            self.baseline_stats = self._calculate_baseline_statistics(
                data, target_column, predictions
            )
            
            logger.info(f"Baseline set with {len(data)} samples and {len(data.columns)} features")
            
        except Exception as e:
            raise MonitoringError(f"Failed to set baseline: {e}")
    
    def detect_drift(self, current_data: pd.DataFrame, 
                    target_column: Optional[str] = None,
                    predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Detect drift in current data compared to baseline.
        
        Args:
            current_data (pd.DataFrame): Current data to check for drift
            target_column (Optional[str]): Target column name
            predictions (Optional[np.ndarray]): Model predictions on current data
            
        Returns:
            Dict[str, Any]: Drift detection results
            
        Raises:
            MonitoringError: If drift detection fails
        """
        logger.info("Detecting drift in current data...")
        
        if self.baseline_data is None:
            raise MonitoringError("Baseline data not set. Call set_baseline() first.")
        
        try:
            drift_results = {
                'timestamp': datetime.now().isoformat(),
                'data_shape': current_data.shape,
                'feature_drift': {},
                'overall_drift_score': 0.0,
                'drift_detected': False,
                'alerts': []
            }
            
            # Feature drift detection
            if self.monitor_features:
                feature_drift = self._detect_feature_drift(current_data)
                drift_results['feature_drift'] = feature_drift
                
                # Calculate overall drift score
                drift_scores = [result['drift_score'] for result in feature_drift.values()]
                drift_results['overall_drift_score'] = np.mean(drift_scores) if drift_scores else 0.0
            
            # Performance drift detection
            if self.monitor_performance and target_column is not None:
                performance_drift = self._detect_performance_drift(
                    current_data, target_column, predictions
                )
                drift_results['performance_drift'] = performance_drift
            
            # Prediction drift detection
            if self.monitor_predictions and predictions is not None:
                prediction_drift = self._detect_prediction_drift(predictions)
                drift_results['prediction_drift'] = prediction_drift
            
            # Determine if drift is detected
            drift_detected = self._evaluate_drift_severity(drift_results)
            drift_results['drift_detected'] = drift_detected
            
            # Generate alerts
            if self.alert_on_drift and drift_detected:
                alerts = self._generate_alerts(drift_results)
                drift_results['alerts'] = alerts
                self.alerts.extend(alerts)
            
            # Store in history
            self.drift_history.append(drift_results)
            
            logger.info(f"Drift detection completed. Overall score: {drift_results['overall_drift_score']:.3f}")
            logger.info(f"Drift detected: {drift_detected}")
            
            return drift_results
            
        except Exception as e:
            raise MonitoringError(f"Drift detection failed: {e}")
    
    def _calculate_baseline_statistics(self, data: pd.DataFrame, 
                                     target_column: Optional[str] = None,
                                     predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Calculate baseline statistics for drift detection."""
        
        stats = {
            'feature_stats': {},
            'correlation_matrix': None,
            'prediction_stats': None
        }
        
        # Feature statistics
        for column in data.columns:
            if column == target_column:
                continue
                
            col_data = data[column]
            if pd.api.types.is_numeric_dtype(col_data):
                stats['feature_stats'][column] = {
                    'mean': col_data.mean(),
                    'std': col_data.std(),
                    'min': col_data.min(),
                    'max': col_data.max(),
                    'percentiles': col_data.quantile([0.25, 0.5, 0.75]).to_dict(),
                    'type': 'numeric'
                }
            else:
                value_counts = col_data.value_counts(normalize=True)
                stats['feature_stats'][column] = {
                    'value_counts': value_counts.to_dict(),
                    'unique_count': col_data.nunique(),
                    'type': 'categorical'
                }
        
        # Correlation matrix for numeric features
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 1:
            stats['correlation_matrix'] = data[numeric_cols].corr().to_dict()
        
        # Prediction statistics
        if predictions is not None:
            stats['prediction_stats'] = {
                'mean': np.mean(predictions),
                'std': np.std(predictions),
                'min': np.min(predictions),
                'max': np.max(predictions),
                'percentiles': np.percentile(predictions, [25, 50, 75]).tolist()
            }
        
        return stats
    
    def _detect_feature_drift(self, current_data: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """Detect drift in individual features."""
        
        feature_drift = {}
        
        for column in current_data.columns:
            if column not in self.baseline_stats['feature_stats']:
                continue
            
            baseline_stat = self.baseline_stats['feature_stats'][column]
            current_col = current_data[column]
            baseline_col = self.baseline_data[column]
            
            drift_info = {
                'drift_detected': False,
                'drift_score': 0.0,
                'test_type': None,
                'p_value': None,
                'effect_size': None
            }
            
            if baseline_stat['type'] == 'numeric':
                # KS test for numeric features
                try:
                    statistic, p_value = ks_2samp(baseline_col, current_col)
                    drift_info['test_type'] = 'ks_test'
                    drift_info['p_value'] = p_value
                    drift_info['effect_size'] = statistic
                    drift_info['drift_score'] = 1 - p_value  # Higher score = more drift
                    drift_info['drift_detected'] = p_value < self.drift_threshold
                except Exception as e:
                    logger.warning(f"KS test failed for {column}: {e}")
                    continue
                
                # Wasserstein distance for additional measure
                try:
                    wasserstein_dist = wasserstein_distance(baseline_col, current_col)
                    drift_info['wasserstein_distance'] = wasserstein_dist
                    if wasserstein_dist > self.wasserstein_threshold:
                        drift_info['drift_detected'] = True
                except Exception as e:
                    logger.warning(f"Wasserstein distance failed for {column}: {e}")
                
            else:  # categorical
                # Chi-square test for categorical features
                try:
                    # Create contingency table
                    baseline_counts = pd.Series(baseline_stat['value_counts'])
                    current_counts = current_col.value_counts(normalize=True)
                    
                    # Align indices
                    all_categories = set(baseline_counts.index) | set(current_counts.index)
                    baseline_counts = baseline_counts.reindex(all_categories, fill_value=0)
                    current_counts = current_counts.reindex(all_categories, fill_value=0)
                    
                    # Chi-square test
                    chi2_stat, p_value, dof, expected = chi2_contingency(
                        [baseline_counts.values, current_counts.values]
                    )
                    
                    drift_info['test_type'] = 'chi2_test'
                    drift_info['p_value'] = p_value
                    drift_info['effect_size'] = chi2_stat
                    drift_info['drift_score'] = 1 - p_value
                    drift_info['drift_detected'] = p_value < self.chi2_threshold
                    
                except Exception as e:
                    logger.warning(f"Chi-square test failed for {column}: {e}")
                    continue
            
            feature_drift[column] = drift_info
        
        return feature_drift
    
    def _detect_performance_drift(self, current_data: pd.DataFrame, 
                                 target_column: str,
                                 predictions: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """Detect performance drift."""
        
        performance_drift = {
            'drift_detected': False,
            'metrics': {},
            'degradation_score': 0.0
        }
        
        if predictions is not None:
            # Calculate current performance metrics
            current_metrics = {
                'accuracy': accuracy_score(current_data[target_column], predictions > 0.5),
                'precision': precision_score(current_data[target_column], predictions > 0.5, average='weighted'),
                'recall': recall_score(current_data[target_column], predictions > 0.5, average='weighted'),
                'f1': f1_score(current_data[target_column], predictions > 0.5, average='weighted')
            }
            
            # Compare with baseline if available
            if hasattr(self, 'baseline_performance'):
                degradation_scores = []
                for metric, current_value in current_metrics.items():
                    if metric in self.baseline_performance:
                        baseline_value = self.baseline_performance[metric]
                        degradation = (baseline_value - current_value) / baseline_value
                        degradation_scores.append(max(0, degradation))
                
                performance_drift['degradation_score'] = np.mean(degradation_scores) if degradation_scores else 0.0
                performance_drift['drift_detected'] = performance_drift['degradation_score'] > self.performance_threshold
            
            performance_drift['metrics'] = current_metrics
        
        return performance_drift
    
    def _detect_prediction_drift(self, predictions: np.ndarray) -> Dict[str, Any]:
        """Detect drift in model predictions."""
        
        prediction_drift = {
            'drift_detected': False,
            'drift_score': 0.0,
            'statistics': {}
        }
        
        if self.baseline_stats.get('prediction_stats') is not None:
            baseline_stats = self.baseline_stats['prediction_stats']
            
            # Compare prediction distributions
            current_mean = np.mean(predictions)
            current_std = np.std(predictions)
            
            # Calculate drift score based on distribution differences
            mean_diff = abs(current_mean - baseline_stats['mean']) / baseline_stats['std']
            std_diff = abs(current_std - baseline_stats['std']) / baseline_stats['std']
            
            drift_score = (mean_diff + std_diff) / 2
            prediction_drift['drift_score'] = drift_score
            prediction_drift['drift_detected'] = drift_score > self.wasserstein_threshold
            
            prediction_drift['statistics'] = {
                'current_mean': current_mean,
                'current_std': current_std,
                'baseline_mean': baseline_stats['mean'],
                'baseline_std': baseline_stats['std'],
                'mean_difference': mean_diff,
                'std_difference': std_diff
            }
        
        return prediction_drift
    
    def _evaluate_drift_severity(self, drift_results: Dict[str, Any]) -> bool:
        """Evaluate overall drift severity."""
        
        # Check feature drift
        feature_drift_detected = any(
            result['drift_detected'] for result in drift_results.get('feature_drift', {}).values()
        )
        
        # Check performance drift
        performance_drift_detected = drift_results.get('performance_drift', {}).get('drift_detected', False)
        
        # Check prediction drift
        prediction_drift_detected = drift_results.get('prediction_drift', {}).get('drift_detected', False)
        
        # Overall drift score threshold
        overall_score_threshold = drift_results.get('overall_drift_score', 0) > 0.3
        
        return (feature_drift_detected or performance_drift_detected or 
                prediction_drift_detected or overall_score_threshold)
    
    def _generate_alerts(self, drift_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alerts for detected drift."""
        
        alerts = []
        
        # Feature drift alerts
        for feature, result in drift_results.get('feature_drift', {}).items():
            if result['drift_detected']:
                alerts.append({
                    'type': 'feature_drift',
                    'severity': 'high' if result['drift_score'] > 0.7 else 'medium',
                    'message': f"Drift detected in feature '{feature}' (score: {result['drift_score']:.3f})",
                    'feature': feature,
                    'drift_score': result['drift_score'],
                    'timestamp': drift_results['timestamp']
                })
        
        # Performance drift alerts
        performance_drift = drift_results.get('performance_drift', {})
        if performance_drift.get('drift_detected', False):
            alerts.append({
                'type': 'performance_drift',
                'severity': 'high' if performance_drift['degradation_score'] > 0.2 else 'medium',
                'message': f"Performance degradation detected (score: {performance_drift['degradation_score']:.3f})",
                'degradation_score': performance_drift['degradation_score'],
                'timestamp': drift_results['timestamp']
            })
        
        # Prediction drift alerts
        prediction_drift = drift_results.get('prediction_drift', {})
        if prediction_drift.get('drift_detected', False):
            alerts.append({
                'type': 'prediction_drift',
                'severity': 'high' if prediction_drift['drift_score'] > 0.5 else 'medium',
                'message': f"Prediction drift detected (score: {prediction_drift['drift_score']:.3f})",
                'drift_score': prediction_drift['drift_score'],
                'timestamp': drift_results['timestamp']
            })
        
        return alerts
    
    def get_drift_summary(self) -> Dict[str, Any]:
        """Get summary of drift detection results."""
        
        if not self.drift_history:
            return {'message': 'No drift detection history available'}
        
        latest_drift = self.drift_history[-1]
        
        summary = {
            'total_checks': len(self.drift_history),
            'latest_check': latest_drift['timestamp'],
            'drift_detected': latest_drift['drift_detected'],
            'overall_drift_score': latest_drift['overall_drift_score'],
            'feature_drift_count': sum(
                1 for result in latest_drift.get('feature_drift', {}).values()
                if result['drift_detected']
            ),
            'total_alerts': len(self.alerts),
            'recent_alerts': self.alerts[-5:] if self.alerts else []
        }
        
        return summary
    
    def get_monitoring_summary(self) -> Dict[str, Any]:
        """
        Get summary of monitoring results.
        
        Returns:
            Dict[str, Any]: Monitoring summary
        """
        return self.get_drift_summary()
    
    def plot_drift_analysis(self, save_path: Optional[str] = None) -> None:
        """Create visualization of drift analysis."""
        
        if not self.drift_history:
            logger.warning("No drift history available for plotting")
            return
        
        latest_drift = self.drift_history[-1]
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Model Drift Analysis', fontsize=16)
        
        # 1. Feature drift scores
        if latest_drift.get('feature_drift'):
            feature_scores = {
                feature: result['drift_score']
                for feature, result in latest_drift['feature_drift'].items()
            }
            
            if feature_scores:
                features = list(feature_scores.keys())
                scores = list(feature_scores.values())
                
                axes[0, 0].barh(features, scores)
                axes[0, 0].axvline(x=self.drift_threshold, color='red', linestyle='--', label='Threshold')
                axes[0, 0].set_title('Feature Drift Scores')
                axes[0, 0].set_xlabel('Drift Score')
                axes[0, 0].legend()
        
        # 2. Drift history over time
        if len(self.drift_history) > 1:
            timestamps = [drift['timestamp'] for drift in self.drift_history]
            scores = [drift['overall_drift_score'] for drift in self.drift_history]
            
            axes[0, 1].plot(timestamps, scores, marker='o')
            axes[0, 1].axhline(y=0.3, color='red', linestyle='--', label='Drift Threshold')
            axes[0, 1].set_title('Drift Score Over Time')
            axes[0, 1].set_ylabel('Overall Drift Score')
            axes[0, 1].tick_params(axis='x', rotation=45)
            axes[0, 1].legend()
        
        # 3. Alert distribution
        if self.alerts:
            alert_types = [alert['type'] for alert in self.alerts]
            alert_counts = pd.Series(alert_types).value_counts()
            
            axes[1, 0].pie(alert_counts.values, labels=alert_counts.index, autopct='%1.1f%%')
            axes[1, 0].set_title('Alert Distribution by Type')
        
        # 4. Performance metrics (if available)
        if latest_drift.get('performance_drift', {}).get('metrics'):
            metrics = latest_drift['performance_drift']['metrics']
            metric_names = list(metrics.keys())
            metric_values = list(metrics.values())
            
            axes[1, 1].bar(metric_names, metric_values)
            axes[1, 1].set_title('Current Performance Metrics')
            axes[1, 1].set_ylabel('Score')
            axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Drift analysis plot saved to {save_path}")
        else:
            plt.show()
    
    def save_drift_report(self, filepath: str) -> None:
        """Save comprehensive drift detection report."""
        
        try:
            report = {
                'baseline_info': {
                    'data_shape': self.baseline_data.shape if self.baseline_data is not None else None,
                    'features': list(self.baseline_stats.get('feature_stats', {}).keys()) if self.baseline_stats else None
                },
                'drift_history': self.drift_history,
                'alerts': self.alerts,
                'summary': self.get_drift_summary(),
                'config': {
                    'drift_threshold': self.drift_threshold,
                    'wasserstein_threshold': self.wasserstein_threshold,
                    'chi2_threshold': self.chi2_threshold,
                    'performance_threshold': self.performance_threshold
                }
            }
            
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            
            logger.info(f"Drift detection report saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to save drift report: {e}")
    
    def clear_history(self) -> None:
        """Clear drift detection history."""
        self.drift_history = []
        self.alerts = []
        logger.info("Drift detection history cleared") 