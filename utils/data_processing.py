import pandas as pd
import numpy as np
from math import isclose
from typing import Literal
from dataclasses import dataclass, field
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

from .general_functions import def_var_value_if_none, label_encoder


def apply_pca(dataframe, remaining_dim=3):
    return pd.DataFrame(
        data=PCA(n_components=remaining_dim).fit_transform(dataframe),
        index=dataframe.index,
        columns=[f"PC{i+1}" for i in range(remaining_dim)]
    )


def apply_lda(dataframe, labels, remaining_dim=3):
    if isinstance(labels, pd.Series):
        labels = label_encoder(labels=labels, unique_vals=labels.unique())
    return pd.DataFrame(
        data=LDA(n_components=remaining_dim).fit_transform(dataframe.values, labels),
        index=dataframe.index,
        columns=[f"LD{i+1}" for i in range(remaining_dim)]
    )


def scale_to_percent(series: pd.Series, scale_factor=None, norm_ord=2):
    if scale_factor is None:
        scale = np.sum(series.values ** norm_ord)
        if isclose(scale, 1, abs_tol=0.0001):
            return series
        else:
            return series / (scale ** (1 / norm_ord))
    else:
        return series * scale_factor


def standardize_sr(series):
    return (series - series.mean()) / series.std()


def min_max_scale_sr(data, max_is_best=True):
    data_min = data.min()
    data_max = data.max()
    delta = data_max - data_min
    if delta == 0:
        return data / (data_max * len(data.index))

    scale = (data - data_min) / delta
    if max_is_best:
        return scale
    else:
        return 1 - scale


