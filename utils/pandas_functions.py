import pandas as pd
import numpy as np

from .general_functions import def_var_value_if_none, tuple_to_text


def check_col_dtype(df, datatype, rtn_list=True):
    if rtn_list:
        bl_list = []
        for col in df.columns:
            if isinstance(df[col].values[0], datatype):
                bl_list = [col]
        return bl_list
    else:
        bl_list = [isinstance(df[col].values[0], datatype) for col in df.columns]
        return pd.Series(bl_list, index=df.columns)


def add_idx_to_df(dataframe: pd.DataFrame, idx_to_add: (list, pd.Series, str), index_name=None):
    if isinstance(idx_to_add, pd.Series | pd.Index | pd.MultiIndex) and index_name is None:
        idx_name = idx_to_add.name
    elif index_name is None:
        idx_name = "Index"
    else:
        idx_name = index_name

    if isinstance(idx_to_add, str):
        idx_to_add = [idx_to_add] * len(dataframe.index)

    mult_idx_df = pd.DataFrame(dataframe, copy=True)
    mult_idx_df[idx_name] = idx_to_add
    return mult_idx_df.set_index([idx_name], append=True).squeeze()


def add_1st_lvl_index_to_df(dataframe, index_list: (list, pd.Series, str), index_name=None):
    mult_idx_df = add_idx_to_df(dataframe, index_list, index_name)
    # dataframe = dataframe.reorder_levels([dataframe.index.names[1], dataframe.index.names[0]])
    lvl_order = [mult_idx_df.index.nlevels-1] + [i for i in range(mult_idx_df.index.nlevels-1)]
    mult_idx_df = mult_idx_df.reorder_levels(lvl_order)
    return mult_idx_df


def m_idx_to_tuple_idx_pd(data_pd: pd.DataFrame, axis=0):
    if axis == 0 or isinstance(data_pd, pd.Series):
        m_idx = data_pd.index
    else:
        m_idx = data_pd.columns
    """
    print("mf_line_1911")
    print(m_idx)
    print(m_idx.to_flat_index())
    print(data_pd.set_axis(m_idx.to_flat_index(), axis=axis))
    print(pd.DataFrame(data_pd, index=m_idx.to_flat_index()))
    """
    return data_pd.set_axis(m_idx.to_flat_index(), axis=axis)


def merge_data_pd_multi_index(data, sep_str=None, index_name=None):
    index_name = def_var_value_if_none(index_name, "merged_idx")
    sep_str = def_var_value_if_none(sep_str, " |-| ")
    data_df = pd.DataFrame(data.copy())
    if not isinstance(data.index, pd.MultiIndex):
        return data_df.squeeze()

    data_df[index_name] = [f"{tuple_to_text(idx_tuple, sep=sep_str)}" for idx_tuple in data.index]
    return data_df.reset_index(drop=True).set_index(keys=index_name, append=False)


def merge_dfs(df_old, df_new, axis=1, overwrite_common=True):
    if isinstance(df_old, pd.Series):
        df_old = pd.DataFrame(df_old)
    if isinstance(df_new, pd.Series):
        df_new = pd.DataFrame(df_new)

    if axis == 1:
        df_new_unique = df_new.loc[:, ~df_new.columns.isin(df_old.columns)]
        if overwrite_common:
            df_new_common = df_new.loc[:, df_new.columns.isin(df_old.columns)]
            df_old[df_new_common.columns] = df_new_common
        return pd.concat([df_old, df_new_unique], axis=axis)
    if axis == 0:
        df_new_unique = df_new.loc[~df_new.index.isin(df_old.index)]
        if overwrite_common:
            df_new_common = df_new.loc[df_new.index.isin(df_old.index)]
            df_old.loc[df_new_common.index] = df_new_common
        return pd.concat([df_old, df_new_unique], axis=axis)


