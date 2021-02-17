"""
General utilities for simulation.
"""
from typing import Optional
import pandas as pd
from shapely.geometry import Point
from fractopo.analysis.network import Network
import numpy as np
from fractopo_scripts.simulation.schema import describe_df_schema


GEOM_COL = "geometry"
