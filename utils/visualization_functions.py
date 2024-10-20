import pandas as pd
import numpy as np
from typing import Literal
import plotly.express as px

from .general_functions import def_var_value_if_none, tuple_to_text
from .data_processing import apply_pca, apply_lda


def set_up_3d_graph_data(data: pd.DataFrame, labels: pd.Series, select_redux=Literal["PCA", "LDA"]):
    # data_m = merge_data_pd_multi_index(data)
    # labels_m = merge_data_pd_multi_index(labels)
    redux_df = None
    if select_redux == "PCA":
        """
        redux_df = create_dim_reduction_with(
            lambda x: apply_pca(x, remaining_dim=3),
            data=data_m,
            labels=labels_m
        )
        """
        redux_df = apply_pca(data, remaining_dim=3)
        redux_df["label"] = labels.squeeze().apply(lambda x: f"cp: {x}")
    elif select_redux == "LDA":
        """
        le = LabelEncoder()
        le.classes_ = labels_m.unique()
        labels_c = le.transform(labels_m)

        redux_df = create_dim_reduction_with(
            lambda x: apply_lda(x, labels_c, remaining_dim=3),
            data=data_m,
            labels=labels_m
        )
        """
        redux_df = apply_lda(data, labels, remaining_dim=3)
        redux_df["label"] = labels.squeeze().apply(lambda x: f"cp: {x}")

    axis_str = select_redux[:-1]
    return px.scatter_3d(redux_df.copy(), x=f"{axis_str}1", y=f"{axis_str}2", z=f"{axis_str}3", color="label")


def set_up_df_h_bar_chart(data_df, histogram=True, axis=0):
    if isinstance(data_df.index, pd.MultiIndex) or isinstance(data_df.index[0], tuple):
        data_df = data_df.set_axis(pd.Index([tuple_to_text(i) for i in data_df.index]), axis=0)

    if isinstance(data_df.columns, pd.MultiIndex) or isinstance(data_df.columns[0], tuple):
        data_df = data_df.set_axis(pd.Index([tuple_to_text(i) for i in data_df.columns]), axis=1)

    idx_name_1 = def_var_value_if_none(value_passed=data_df.index.name, default="index")
    idx_name_2 = def_var_value_if_none(value_passed=data_df.columns.name, default="columns")
    col_names = "value"
    data = data_df.reset_index(names=[idx_name_1]).melt(
        var_name=idx_name_2, id_vars=[idx_name_1]
    )  # .set_index(drop=True, keys=["index", "columns"]).squeeze()

    print("mf_line_1283")
    print(data)

    if axis == 0:
        return px.bar(
            data, y=idx_name_1, color=idx_name_2, x=col_names, barmode='group' if histogram else "stack"
        )   # .update_yaxes(categoryorder='array', categoryarray=desired_order)
    else:
        return px.bar(
            data, x=idx_name_1, color=idx_name_2, y=col_names, barmode='group' if histogram else "stack"
        )


def set_up_m_idx_sr_h_bar_chart(data_df, histogram=True, axis=0):
    if data_df.index.nlevels > 2:
        raise Exception(f"More than 2 levels in multi-index: {data_df.index.nlevels}")

    idx_lvl_1 = data_df.index.get_level_values(0)
    idx_lvl_2 = data_df.index.get_level_values(1)

    idx_name_1 = def_var_value_if_none(value_passed=idx_lvl_1.name, default="lv_1")
    idx_name_2 = def_var_value_if_none(value_passed=idx_lvl_2.name, default="lv_2")

    if isinstance(idx_lvl_1[0], tuple):
        idx_lvl_1 = [tuple_to_text(tpl, sep=" : ") for tpl in idx_lvl_1]
    if isinstance(idx_lvl_2[0], tuple):
        idx_lvl_2 = [tuple_to_text(tpl, sep=" : ") for tpl in idx_lvl_2]

    col_names = data_df.name

    data = pd.DataFrame({
        col_names: data_df.values,
        idx_name_1: idx_lvl_1,
        idx_name_2: idx_lvl_2
    })
    # data = pd.DataFrame(data_df).reset_index(names=[idx_name_1, idx_name_2])
    if axis == 0:
        return px.bar(
            data, y=idx_name_1, color=idx_name_2, x=col_names, barmode='group' if histogram else "stack"
        )  # .update_yaxes(categoryorder='array', categoryarray=desired_order)
    else:
        return px.bar(
            data, x=idx_name_1, color=idx_name_2, y=col_names, barmode='group' if histogram else "stack"
        )
    # return set_up_df_h_bar_chart(data_df.unstack(level=0 if not flip_color else -1))


def set_up_h_bar_chart(data_df: pd.DataFrame | pd.Series, histogram=True, axis=0, flip_color=False):
    data_df = data_df.astype("float32")
    print("\nmf_line_1247")
    print(data_df)
    m_idx = isinstance(data_df.index, pd.MultiIndex)
    if isinstance(data_df, pd.DataFrame):
        """
        if isinstance(data_df.columns, pd.MultiIndex) or isinstance(data_df.columns[0], tuple):
            data_df = data_df.set_axis(pd.Index([tuple_to_text(i) for i in data_df.columns]), axis=1)
        if not m_idx:
            idx_name_1 = def_var_value_if_none(value_passed=data_df.index.name, default="index")
            idx_name_2 = def_var_value_if_none(value_passed=data_df.columns.name, default="columns")
            col_names = "value"
            data = data_df.reset_index(names=[idx_name_1]).melt(
                var_name=idx_name_2, id_vars=[idx_name_1]
            )   # .set_index(drop=True, keys=["index", "columns"]).squeeze()
        else:
            raise Exception("Multiindex df is not supported")
        """
        return set_up_df_h_bar_chart(
            data_df=data_df if not flip_color else data_df.T,
            histogram=histogram, axis=axis
        )

    data_df.name = def_var_value_if_none(data_df.name, default="value")
    if m_idx:
        """
        if data_df.index.nlevels > 2:
            raise Exception(f"More than 2 levels in multi-index: {data_df.index.nlevels}")

        idx_lvl_1 = data_df.index.get_level_values(0)
        idx_lvl_2 = data_df.index.get_level_values(1)

        idx_name_1 = def_var_value_if_none(value_passed=idx_lvl_1.name, default="lv_1")
        idx_name_2 = def_var_value_if_none(value_passed=idx_lvl_2.name, default="lv_2")

        if isinstance(idx_lvl_1[0], tuple):
            idx_lvl_1 = [tuple_to_text(tpl, sep=" : ") for tpl in idx_lvl_1]
        if isinstance(idx_lvl_2[0], tuple):
            idx_lvl_2 = [tuple_to_text(tpl, sep=" : ") for tpl in idx_lvl_2]

        col_names = data_df.name

        data = pd.DataFrame({
            col_names: data_df.values,
            idx_name_1: idx_lvl_1,
            idx_name_2: idx_lvl_2
        })
        # data = pd.DataFrame(data_df).reset_index(names=[idx_name_1, idx_name_2])
        if axis == 0:
            return px.bar(
                data, y=idx_name_1, color=idx_name_2, x=col_names, barmode='group' if histogram else "stack"
            )  # .update_yaxes(categoryorder='array', categoryarray=desired_order)
        else:
            return px.bar(
                data, x=idx_name_1, color=idx_name_2, y=col_names, barmode='group' if histogram else "stack"
            )
        # return set_up_df_h_bar_chart(data_df.unstack(level=0 if not flip_color else -1))
        """
        return set_up_m_idx_sr_h_bar_chart(data_df, histogram=histogram, axis=axis)

    idx = data_df.index
    if isinstance(idx[0], tuple):
        idx = [tuple_to_text(tpl, sep=" : ") for tpl in idx]

    data = data_df.values
    """
    if axis == 0:
        # return px.bar(pd.DataFrame(data_df), x=data_df.index, y=data_df.name)
        return px.bar(pd.Series(data, index=idx, name=data_df.name))
    else:
        # return px.bar(pd.DataFrame(data_df), y=data_df.index, x=data_df.name)
    """
    return px.bar(pd.Series(data, index=idx, name=data_df.name), orientation="h" if axis == 1 else None)


