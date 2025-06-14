#!/usr/bin/env python3
"""
Test Advanced Feature Engineering
Comprehensive testing of advanced feature engineering techniques.
"""

import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

# Add the project root to the Python path (go up one level from tests/)
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_ml.features.engineering import AdvancedFeatureEngineering
from auto_ml.data.ingestion import DatasetRegistry

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_basic_advanced_feature_engineering():
    """Test basic advanced feature engineering functionality."""
    logger.info("=" * 60)
    logger.info("TESTING BASIC ADVANCED FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    # Load a small dataset for testing
    registry = DatasetRegistry()
    ingestion = registry.create_ingestion('iris')
    train_data, test_data = ingestion.load_data()
    
    # Basic configuration
    config = {
        'feature_selection_method': 'mutual_info',
        'feature_selection_k': 3,
        'use_target_encoding': False,
        'use_feature_hashing': False,
        'use_pca': False,
        'use_clustering_features': False,
        'use_interaction_features': False,
        'use_statistical_features': False
    }
    
    # Create advanced feature engineering instance
    feature_engineering = AdvancedFeatureEngineering(config)
    
    # Apply feature engineering
    train_engineered, test_engineered = feature_engineering.engineer_features(
        train_data, test_data, ingestion.get_target_column()
    )
    
    # Verify results
    logger.info(f"Original training shape: {train_data.shape}")
    logger.info(f"Engineered training shape: {train_engineered.shape}")
    logger.info(f"Original test shape: {test_data.shape}")
    logger.info(f"Engineered test shape: {test_engineered.shape}")
    
    # Check feature information
    feature_info = feature_engineering.get_engineered_features_info()
    logger.info(f"Feature engineering info: {feature_info}")
    
    # Verify target column is preserved
    assert ingestion.get_target_column() in train_engineered.columns
    assert ingestion.get_target_column() in test_engineered.columns
    
    logger.info("✓ Basic advanced feature engineering test passed")
    return feature_engineering

def test_feature_selection_methods():
    """Test different feature selection methods."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING FEATURE SELECTION METHODS")
    logger.info("=" * 60)
    
    # Load dataset
    registry = DatasetRegistry()
    ingestion = registry.create_ingestion('iris')
    train_data, test_data = ingestion.load_data()
    
    selection_methods = ['mutual_info', 'chi2', 'f_classif', 'percentile', 'rfe']
    
    for method in selection_methods:
        logger.info(f"\nTesting feature selection method: {method}")
        
        config = {
            'feature_selection_method': method,
            'feature_selection_k': 2,
            'feature_selection_percentile': 50,
            'use_target_encoding': False,
            'use_feature_hashing': False,
            'use_pca': False,
            'use_clustering_features': False,
            'use_interaction_features': False,
            'use_statistical_features': False
        }
        
        feature_engineering = AdvancedFeatureEngineering(config)
        train_engineered, test_engineered = feature_engineering.engineer_features(
            train_data, test_data, ingestion.get_target_column()
        )
        
        feature_info = feature_engineering.get_engineered_features_info()
        logger.info(f"  Selected {feature_info['selected_features']} features")
        logger.info(f"  Final shape: {train_engineered.shape}")
        
        # Get feature importance if available
        importance = feature_engineering.get_feature_importance()
        if importance:
            logger.info(f"  Feature importance available: {len(importance)} features")
        
        logger.info(f"  ✓ {method} feature selection test passed")

def test_advanced_encoding():
    """Test advanced encoding techniques."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING ADVANCED ENCODING TECHNIQUES")
    logger.info("=" * 60)
    
    # Load dataset with categorical features
    registry = DatasetRegistry()
    ingestion = registry.create_ingestion('adult_income')
    train_data, test_data = ingestion.load_data()
    
    # Test target encoding
    logger.info("\nTesting target encoding...")
    config = {
        'feature_selection_method': 'none',
        'use_target_encoding': True,
        'use_feature_hashing': False,
        'use_pca': False,
        'use_clustering_features': False,
        'use_interaction_features': False,
        'use_statistical_features': False
    }
    
    feature_engineering = AdvancedFeatureEngineering(config)
    train_engineered, test_engineered = feature_engineering.engineer_features(
        train_data, test_data, ingestion.get_target_column()
    )
    
    # Check for target encoded columns
    target_encoded_cols = [col for col in train_engineered.columns if 'target_encoded' in col]
    logger.info(f"  Created {len(target_encoded_cols)} target encoded features")
    logger.info(f"  Target encoded columns: {target_encoded_cols[:3]}...")  # Show first 3
    
    # Test feature hashing
    logger.info("\nTesting feature hashing...")
    config['use_target_encoding'] = False
    config['use_feature_hashing'] = True
    
    feature_engineering = AdvancedFeatureEngineering(config)
    train_engineered, test_engineered = feature_engineering.engineer_features(
        train_data, test_data, ingestion.get_target_column()
    )
    
    # Check for hashed columns
    hashed_cols = [col for col in train_engineered.columns if 'hash_' in col]
    logger.info(f"  Created {len(hashed_cols)} hashed features")
    logger.info(f"  Hashed columns: {hashed_cols[:3]}...")  # Show first 3
    
    logger.info("✓ Advanced encoding tests passed")

def test_feature_creation():
    """Test feature creation techniques."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING FEATURE CREATION TECHNIQUES")
    logger.info("=" * 60)
    
    # Load dataset
    registry = DatasetRegistry()
    ingestion = registry.create_ingestion('wine')
    train_data, test_data = ingestion.load_data()
    
    # Test interaction features
    logger.info("\nTesting interaction features...")
    config = {
        'feature_selection_method': 'none',
        'use_target_encoding': False,
        'use_feature_hashing': False,
        'use_pca': False,
        'use_clustering_features': False,
        'use_interaction_features': True,
        'use_statistical_features': False
    }
    
    feature_engineering = AdvancedFeatureEngineering(config)
    train_engineered, test_engineered = feature_engineering.engineer_features(
        train_data, test_data, ingestion.get_target_column()
    )
    
    # Check for interaction columns
    interaction_cols = [col for col in train_engineered.columns if '_x_' in col]
    logger.info(f"  Created {len(interaction_cols)} interaction features")
    logger.info(f"  Interaction columns: {interaction_cols[:3]}...")  # Show first 3
    
    # Test statistical features
    logger.info("\nTesting statistical features...")
    config['use_interaction_features'] = False
    config['use_statistical_features'] = True
    
    feature_engineering = AdvancedFeatureEngineering(config)
    train_engineered, test_engineered = feature_engineering.engineer_features(
        train_data, test_data, ingestion.get_target_column()
    )
    
    # Check for statistical columns
    statistical_cols = [col for col in train_engineered.columns if col.startswith('row_')]
    logger.info(f"  Created {len(statistical_cols)} statistical features")
    logger.info(f"  Statistical columns: {statistical_cols}")
    
    # Test clustering features
    logger.info("\nTesting clustering features...")
    config['use_statistical_features'] = False
    config['use_clustering_features'] = True
    config['n_clusters'] = 3
    
    feature_engineering = AdvancedFeatureEngineering(config)
    train_engineered, test_engineered = feature_engineering.engineer_features(
        train_data, test_data, ingestion.get_target_column()
    )
    
    # Check for clustering columns
    clustering_cols = [col for col in train_engineered.columns if 'cluster' in col]
    logger.info(f"  Created {len(clustering_cols)} clustering features")
    logger.info(f"  Clustering columns: {clustering_cols}")
    
    logger.info("✓ Feature creation tests passed")

def test_dimensionality_reduction():
    """Test dimensionality reduction techniques."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING DIMENSIONALITY REDUCTION")
    logger.info("=" * 60)
    
    # Load dataset
    registry = DatasetRegistry()
    ingestion = registry.create_ingestion('wine')
    train_data, test_data = ingestion.load_data()
    
    # Test PCA
    logger.info("\nTesting PCA...")
    config = {
        'feature_selection_method': 'none',
        'use_target_encoding': False,
        'use_feature_hashing': False,
        'use_pca': True,
        'pca_components': 0.8,  # Keep 80% of variance
        'use_clustering_features': False,
        'use_interaction_features': False,
        'use_statistical_features': False
    }
    
    feature_engineering = AdvancedFeatureEngineering(config)
    train_engineered, test_engineered = feature_engineering.engineer_features(
        train_data, test_data, ingestion.get_target_column()
    )
    
    # Check for PCA columns
    pca_cols = [col for col in train_engineered.columns if col.startswith('pca_')]
    logger.info(f"  Created {len(pca_cols)} PCA components")
    logger.info(f"  PCA columns: {pca_cols}")
    
    # Check PCA transformer
    if 'pca' in feature_engineering.transformers:
        pca = feature_engineering.transformers['pca']
        logger.info(f"  PCA explained variance ratio: {pca.explained_variance_ratio_}")
        logger.info(f"  Total explained variance: {sum(pca.explained_variance_ratio_):.3f}")
    
    logger.info("✓ Dimensionality reduction tests passed")

def test_comprehensive_feature_engineering():
    """Test comprehensive feature engineering with multiple techniques."""
    logger.info("\n" + "=" * 60)
    logger.info("TESTING COMPREHENSIVE FEATURE ENGINEERING")
    logger.info("=" * 60)
    
    # Load dataset
    registry = DatasetRegistry()
    ingestion = registry.create_ingestion('adult_income')
    train_data, test_data = ingestion.load_data()
    
    # Comprehensive configuration
    config = {
        'feature_selection_method': 'mutual_info',
        'feature_selection_k': 15,
        'use_target_encoding': True,
        'use_feature_hashing': True,
        'use_pca': True,
        'pca_components': 0.9,
        'use_clustering_features': True,
        'n_clusters': 4,
        'use_interaction_features': True,
        'use_statistical_features': True
    }
    
    logger.info("Applying comprehensive feature engineering...")
    feature_engineering = AdvancedFeatureEngineering(config)
    train_engineered, test_engineered = feature_engineering.engineer_features(
        train_data, test_data, ingestion.get_target_column()
    )
    
    # Analyze results
    feature_info = feature_engineering.get_engineered_features_info()
    logger.info(f"Original features: {feature_info['original_features']}")
    logger.info(f"Engineered features: {feature_info['engineered_features']}")
    logger.info(f"Selected features: {feature_info['selected_features']}")
    logger.info(f"Transformers used: {feature_info['transformers_used']}")
    
    # Check for different types of features
    target_encoded_cols = [col for col in train_engineered.columns if 'target_encoded' in col]
    hashed_cols = [col for col in train_engineered.columns if 'hash_' in col]
    interaction_cols = [col for col in train_engineered.columns if '_x_' in col]
    statistical_cols = [col for col in train_engineered.columns if col.startswith('row_')]
    clustering_cols = [col for col in train_engineered.columns if 'cluster' in col]
    pca_cols = [col for col in train_engineered.columns if col.startswith('pca_')]
    
    logger.info(f"Target encoded features: {len(target_encoded_cols)}")
    logger.info(f"Hashed features: {len(hashed_cols)}")
    logger.info(f"Interaction features: {len(interaction_cols)}")
    logger.info(f"Statistical features: {len(statistical_cols)}")
    logger.info(f"Clustering features: {len(clustering_cols)}")
    logger.info(f"PCA components: {len(pca_cols)}")
    
    # Verify data integrity
    assert not train_engineered.isnull().any().any(), "Training data contains null values"
    assert not test_engineered.isnull().any().any(), "Test data contains null values"
    assert train_engineered.shape[1] == test_engineered.shape[1], "Train and test have different feature counts"
    
    logger.info("✓ Comprehensive feature engineering test passed")

def main():
    """Run all advanced feature engineering tests."""
    logger.info("Starting Advanced Feature Engineering Tests")
    logger.info("=" * 60)
    
    try:
        # Test 1: Basic functionality
        test_basic_advanced_feature_engineering()
        
        # Test 2: Feature selection methods
        test_feature_selection_methods()
        
        # Test 3: Advanced encoding
        test_advanced_encoding()
        
        # Test 4: Feature creation
        test_feature_creation()
        
        # Test 5: Dimensionality reduction
        test_dimensionality_reduction()
        
        # Test 6: Comprehensive testing
        test_comprehensive_feature_engineering()
        
        logger.info("\n" + "=" * 60)
        logger.info("ALL ADVANCED FEATURE ENGINEERING TESTS COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info("Advanced feature engineering is working correctly.")
        logger.info("Implemented techniques:")
        logger.info("  - Feature selection (mutual_info, chi2, f_classif, percentile, RFE)")
        logger.info("  - Advanced encoding (target encoding, feature hashing)")
        logger.info("  - Feature interactions")
        logger.info("  - Statistical features")
        logger.info("  - Clustering features")
        logger.info("  - Dimensionality reduction (PCA)")
        
    except Exception as e:
        logger.error(f"Advanced feature engineering test suite failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 