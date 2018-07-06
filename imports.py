import pandas as pd
from collections import Counter
import matplotlib.dates as mdates
import datetime
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
from keras.layers import Input, Dense, Flatten, Embedding, Dropout, BatchNormalization
from keras.models import Model
from keras import initializers
import keras
from keras.callbacks import ModelCheckpoint
from collections import Counter
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

import pandas as pd
from IPython.display import display
import numpy as np
import re
from pandas.api.types import is_string_dtype, is_numeric_dtype
import os

from sklearn_pandas import DataFrameMapper
from sklearn.preprocessing import LabelEncoder, Imputer, StandardScaler
from pandas.api.types import is_string_dtype, is_numeric_dtype
from sklearn.ensemble import forest
from sklearn.tree import export_graphviz

def split_vals(a,n): return a[:n].copy(), a[n:].copy()
def split_cols(arr): return np.hsplit(arr,arr.shape[1])
from pandas_summary import DataFrameSummary
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from IPython.display import display

import altair as alt

def emb_init(shape, name=None):
    return initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None)
#initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=None, name = name)
#initializers.uniform(shape, scale=2/(shape[1]+1), name=name)

def get_emb(feat_name):
    c = cat_var_levels[feat_name]
    #name, c = cat_map_info(feat)
    #c2 = cat_var_dict[name]
    c2 = (c+1)//2
    if c2>50: c2=50
    inp = Input((1,), dtype='int64', name=feat_name+'_in')
    # , W_regularizer=l2(1e-6)
   # u = Flatten(name=feat_name+'_flt')(Embedding(c, c2, input_length=1, init=emb_init)(inp))
    u = Flatten(name=feat_name+'_flt')(Embedding(c, c2, input_length=1)(inp))
    return inp,u

def get_contin(feat):
    name = feat[0][0]
    inp = Input((1,), name=name+'_in')
    return inp, Dense(1, name=name+'_d', init=my_init(1.))(inp)
