from typing import Tuple, Optional, Dict, List, Callable
from abc import ABC, abstractmethod
import os
import math
from ulid import ULID
from uuid import UUID

import pandas as pd
import numpy as np
from numpy.random import seed, choice
from pathos.multiprocessing import ProcessingPool
import dill
from scipy.stats import gaussian_kde
from collections import OrderedDict
from tensorflow.keras.preprocessing.text import Tokenizer
from slugify import slugify
from loguru import logger
from attrs import define, field

from syngen.ml.vae import *  # noqa: F403
from syngen.ml.data_loaders import DataLoader, DataFrameFetcher
from syngen.ml.reporters import Report
from syngen.ml.vae.models.dataset import Dataset
from syngen.ml.utils import (
    fetch_config,
    check_if_features_assigned,
    get_initial_table_name,
    ProgressBarHandler
)
from syngen.ml.context import get_context


# Rest of the file content remains the same, with our optimization applied:
flat_json = {key: flatten(value) for key, value in data_dict.items()}