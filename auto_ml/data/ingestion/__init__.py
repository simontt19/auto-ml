"""
Data Ingestion Module
Handles data loading, validation, and preparation for various datasets.
"""

from .adult_income_ingestion import AdultIncomeDataIngestion
from .iris_ingestion import IrisDataIngestion
from .wine_ingestion import WineDataIngestion
from .breast_cancer_ingestion import BreastCancerDataIngestion
from .dataset_registry import DatasetRegistry

__all__ = [
    'AdultIncomeDataIngestion',
    'IrisDataIngestion', 
    'WineDataIngestion',
    'BreastCancerDataIngestion',
    'DatasetRegistry'
] 