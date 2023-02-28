import sys
import pickle
import json
import ast
import pandas as pd
import numpy as np
import geopandas as gpd

from utility import *
import pareto

# Read the sentence from stdin
sentence = sys.stdin.read().strip()

# Manipulate the sentence
manipulated_sentence = sentence.upper()

# Print the manipulated sentence to stdout
print(manipulated_sentence)
