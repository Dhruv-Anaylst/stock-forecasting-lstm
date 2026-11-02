"""
Helper utilities for plotting and seed initialization.
"""

import random
import os
import numpy as np


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility across libraries."""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
