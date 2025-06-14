"""
Adult Income Dataset Data Ingestion
Concrete implementation of BaseDataIngestion for the Adult Income dataset.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path

from auto_ml.core.base_classes import BaseDataIngestion
from auto_ml.core.exceptions import DataIngestionError

logger = logging.getLogger(__name__)

class AdultIncomeDataIngestion(BaseDataIngestion):
    # ... existing code ... 