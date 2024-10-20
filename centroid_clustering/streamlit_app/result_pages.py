import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from typing import Callable, Literal

from centroid_clustering.utils.data_processing import standardize_sr, min_max_scale_sr
from centroid_clustering.utils.visualization_functions import set_up_h_bar_chart, set_up_3d_graph_data
from centroid_clustering.utils.general_functions import type_check
from centroid_clustering.utils.streamlit_functions import multiselect_submit, multiline_chart_from_df_columns, dec_key

from .app_classes import ClSettings, ClRes
from .. import clustering_selection as pam


def create_n_clm_plt_line_chart_pam_kms(fig, row, idx):
    if idx[0] == "cl(1)":
        fig.add_trace(go.Scatter(
            x=row.index, y=row.values, name=f"{idx[0]} | {idx[1]}", mode="lines+markers",
            line=dict(width=1, dash="dash"),
            marker=dict(symbol="circle-open-dot", size=7, angleref="previous")
        ))
    elif idx[0] == "cl(2)":
        fig.add_trace(go.Scatter(
            x=row.index, y=row.values, name=f"{idx[0]} | {idx[1]}", mode="lines+markers",
            line=dict(width=1, dash="dot"),
            marker=dict(symbol="star-diamond-open-dot", size=8, angleref="previous")
        ))


def create_n_clm_dif_max_min_plt_line_chart(df_max_dif, df_min_dif):
    def create_n_cml_plt(df_dif):
        fig_dif = go.Figure()
        for idx in df_dif.index:
            create_n_clm_plt_line_chart_pam_kms(fig_dif, df_dif.loc[idx], idx)
        st.plotly_chart(fig_dif, use_container_width=True)

    col1, col2 = st.columns((1, 1))
    with col1:
        create_n_cml_plt(df_max_dif)
    with col2:
        create_n_cml_plt(df_min_dif)


def fitting_curve_to_n_clm_plt_line_chart(df):
    fig = go.Figure()
    for idx in df.index:
        row = df.loc[idx]
        create_n_clm_plt_line_chart_pam_kms(fig, row, idx)

        if idx[0] == "cl(1)-fit":
            fig.add_trace(go.Scatter(
                x=row.index, y=row.values, name=f"{idx[0]} | {idx[1]}", mode="lines",
                line=dict(width=2),
                # marker=dict(symbol="asterisk", size=8, angleref="previous")
            ))
        elif idx[0] == "cl(2)-fit":
            fig.add_trace(go.Scatter(
                x=row.index, y=row.values, name=f"{idx[0]} | {idx[1]}", mode="lines",
                line=dict(width=2),
                # marker=dict(symbol="asterisk", size=8, angleref="previous")
            ))
    st.plotly_chart(fig, use_container_width=True)


def add_n_cl_metrics_to_line_chart(
        fig_dif: go.Figure,
        n_cl_obj: pam.ClSelect,
        metric_names: list[str],
        method_name: str = "cl(1)",
        var_df_name: str = "n_cl_m_delta",
        apply_fn: Callable = None
):
    for col in metric_names:
        metric_sr = getattr(n_cl_obj, var_df_name)[col]
        metric_sr = metric_sr if apply_fn is None else apply_fn(metric_sr)
        create_n_clm_plt_line_chart_pam_kms(fig_dif, metric_sr, (method_name, col))


def show_n_cl_metrics_line_chart(
        metric_var_name: str,
        pam_n_cl_obj: pam.ClSelect,
        kms_n_cl_obj: pam.ClSelect | None = None,
        var_df_name: str = "n_cl_m_delta",
        apply_fn=None
):
    fig_dif = go.Figure()

    def add_n_cl_m_partial(n_cl_obj, method_name):
        add_n_cl_metrics_to_line_chart(
            fig_dif=fig_dif,
            n_cl_obj=n_cl_obj,
            metric_names=getattr(n_cl_obj, metric_var_name),
            method_name=method_name,
            var_df_name=var_df_name,
            apply_fn=apply_fn
        )

    add_n_cl_m_partial(pam_n_cl_obj, method_name="cl(1)")
    add_n_cl_m_partial(kms_n_cl_obj, method_name="cl(2)") if kms_n_cl_obj is not None else None

    st.plotly_chart(fig_dif, use_container_width=True)


def create_n_cl_metrics_plotly_multy_line_charts(
        pam_n_cl_obj: pam.ClSelect | ClSettings,
        kms_n_cl_obj: pam.ClSelect | ClSettings | None = None,
        pam_name=None,
        kms_name=None
):
    kms_true = kms_n_cl_obj is not None
    pam_name = "K-medoids" if pam_name is None else pam_name
    kms_name = "K-means" if kms_name is None else kms_name

    metrics = multiselect_submit(
        label="Select metrics for evaluating the number of clusters",
        options=pam_n_cl_obj.metrics_for_fitting,
        default=["Simplified Silhouette"],
        label_above=True
    )

    if not kms_true:
        df_best_fit = pd.concat([
            pam_n_cl_obj.n_cl_metrics[metrics].T,
            pam_n_cl_obj.n_cl_m_fit[metrics].T,
        ], keys=["cl(1)", "cl(1)-fit"])
    else:
        df_best_fit = pd.concat([
            pam_n_cl_obj.n_cl_metrics[metrics].T,
            pam_n_cl_obj.n_cl_m_fit[metrics].T,
            kms_n_cl_obj.n_cl_metrics[metrics].T,
            kms_n_cl_obj.n_cl_m_fit[metrics].T
        ], keys=["cl(1)", "cl(1)-fit", "cl(2)", "cl(2)-fit"])

    c1, c2 = st.columns((5, 1), vertical_alignment="bottom")
    # show_data = c1.toggle(label="Show metrics Data Table", value=False)

    r_delta = c2.toggle(label="Relative delta chart", value=False)

    fitting_curve_to_n_clm_plt_line_chart(df_best_fit)

    # ---------------------------
    def show_n_cl_m_plot(**kwargs):
        show_n_cl_metrics_line_chart(pam_n_cl_obj=pam_n_cl_obj, kms_n_cl_obj=kms_n_cl_obj, **kwargs)

    col1, col2 = st.columns((1, 1))
    with col1:
        if r_delta:
            show_n_cl_m_plot(
                metric_var_name="decreasing_val_metric_names", var_df_name="n_cl_m_r_delta"
            )
        else:
            show_n_cl_m_plot(
                metric_var_name="decreasing_val_metric_names", var_df_name="n_cl_m_delta", apply_fn=standardize_sr
            )

    with col2:
        if r_delta:
            show_n_cl_m_plot(
                metric_var_name="increasing_val_metric_names", var_df_name="n_cl_m_r_delta"
            )
        else:
            show_n_cl_m_plot(
                metric_var_name="increasing_val_metric_names", var_df_name="n_cl_m_delta", apply_fn=standardize_sr
            )

    """
    if kms_true:
        df_max_dif = pd.concat([
            pam_n_cl_obj.n_cl_m_delta[pam_n_cl_obj.decreasing_val_metric_names].T,
            kms_n_cl_obj.n_cl_m_delta[pam_n_cl_obj.decreasing_val_metric_names].T,
        ], keys=["cl(1)", "cl(2)"]).apply(standardize_sr, axis=1)

        df_min_dif = pd.concat([
            pam_n_cl_obj.n_cl_m_delta[pam_n_cl_obj.increasing_val_metric_names].T,
            kms_n_cl_obj.n_cl_m_delta[pam_n_cl_obj.increasing_val_metric_names].T,
        ], keys=["cl(1)", "cl(2)"]).apply(standardize_sr, axis=1)
    else:
        df_max_dif = pd.concat([
            pam_n_cl_obj.n_cl_m_delta[pam_n_cl_obj.decreasing_val_metric_names].T,
        ], keys=["cl(1)"]).apply(standardize_sr, axis=1)

        df_min_dif = pd.concat([
            pam_n_cl_obj.n_cl_m_delta[pam_n_cl_obj.increasing_val_metric_names].T,
        ], keys=["cl(1)"]).apply(standardize_sr, axis=1)

    create_n_clm_dif_max_min_plt_line_chart(df_max_dif, df_min_dif)
    """

    t1, t2 = c1.tabs(["All data", "Selected data"])
    with t2:
        if not df_best_fit.empty:
            cols = ["cl(1)", "cl(2)"] if kms_true else ["cl(1)"]
            with st.popover(label=f"Open {pam_name} data"):   # , icon=":material/table:"):
                st.write("")
                st.dataframe(df_best_fit.loc[cols].T, use_container_width=True)
    with t1:
        if kms_true:
            c3, c4 = st.columns(2)
            with c3.popover(label=f"Open {pam_name} data"):   # , icon=":material/table:"):
                st.write("")
                st.dataframe(pam_n_cl_obj.n_cl_metrics)
            with c4.popover(label=f"Open {kms_name} data"):
                st.dataframe(kms_n_cl_obj.n_cl_metrics)
        else:
            with st.popover(label=f"Open {pam_name} data"):   # , icon=":material/table:"):
                st.write("")
                st.dataframe(pam_n_cl_obj.n_cl_metrics)


def specific_k_cluster_target_labels_comp(data_obj: ClRes):
    input_obj = data_obj.input_obj
    cl_stg_obj_dict = data_obj.cl_stg_obj_dict

    n_cl_method_key = st.selectbox(
        label="Select centroid_clustering method", options=list(cl_stg_obj_dict.keys()), index=len(cl_stg_obj_dict.keys()) - 1
    )
    n_cl_s_obj = cl_stg_obj_dict[n_cl_method_key]

    if input_obj.target_labels is None:
        st.warning("Target labels have not been specified")
        return

    clm_t_obj = pam.ClMetrics.from_target_labels(
        data=input_obj.data,
        target_labels=input_obj.target_labels,
        center_points=input_obj.center_points,
    )

    specific_k_clusters_results(
        pam_n_cl_obj=n_cl_s_obj.n_cl_obj,
        kms_n_cl_obj=clm_t_obj,
        pam_name=n_cl_s_obj.cl_name,
        kms_name="Target labels metrics"
    )


def specific_k_clusters_results(
        pam_n_cl_obj: pam.ClSelect,
        kms_n_cl_obj: pam.ClSelect | pam.ClMetrics | None = None,
        pam_name=None,
        kms_name=None
):
    kms_true = kms_n_cl_obj is not None
    pam_name = "K-medoids" if pam_name is None else pam_name
    kms_name = "K-means" if kms_name is None else kms_name

    c1, c2 = st.columns(2)

    n_cl_slt = c1.selectbox(
        label="Select number of clusters",
        options=list(pam_n_cl_obj.res_n_cl_obj_dict.keys())  # list(range(min_n_cl, max_n_cl + 1))
    )
    pam_obj_to_show: pam.ClMetrics = pam_n_cl_obj.res_n_cl_obj_dict[n_cl_slt]

    if kms_true:
        if isinstance(kms_n_cl_obj, pam.ClMetrics):
            kms_obj_to_show = kms_n_cl_obj
        elif type_check(kms_n_cl_obj, pam.ClSelect):
            kms_obj_to_show = kms_n_cl_obj.res_n_cl_obj_dict[n_cl_slt]
        else:
            raise Exception(f"Wrong data type: {type(kms_n_cl_obj)}\n should be: 'pam.ClSelect | pam.ClMetrics'")
    else:
        kms_obj_to_show = None

    metric_names = list(pam_obj_to_show.samples_metrics_df.columns)
    samples_metric_slt = c2.selectbox(
        label="Select samples metric", options=metric_names
    )

    show_cl_m_obj_res(samples_metric_slt, pam_obj_to_show, kms_obj_to_show, pam_name, kms_name)


def show_cl_m_obj_res(
        samples_metric: str,
        pam_obj_to_show: pam.ClMetrics,
        kms_obj_to_show: pam.ClMetrics,
        pam_name: str,
        kms_name: str
):
    kms_true = kms_obj_to_show is not None

    if kms_true:
        s_tab, g_tab, d_tab, c_tab = st.tabs(["Samples graphs", "Clusters graphs", "Data tables", "Crosstab"])
    else:
        s_tab, g_tab, d_tab = st.tabs(["Samples graphs", "Clusters graphs", "Data tables"])
        c_tab = st.container()

    with s_tab:
        def rtn_chart(clm_obj, metric_slt):
            sorted_df = clm_obj.samples_metrics_df.sort_values(by=["Labels", metric_slt])
            """
            print("pam_app_line_415")
            print(sorted_df[metric_slt])
            data = sorted_df[metric_slt].to_frame().reset_index()
            data["Samples"] = data["Samples"].apply(lambda x: f"p({x})")
            print(data["Samples"])
            return px.bar(data, y="Samples", color="Labels", x=metric_slt, barmode='group')
            """
            return set_up_h_bar_chart(sorted_df[metric_slt], flip_color=True)

        def show_samples_charts(name, clm_obj, metric_slt):
            st.caption(name)
            st.plotly_chart(rtn_chart(clm_obj, metric_slt))

        if kms_true:
            c1, c2 = st.columns(2)
            with c1:
                show_samples_charts(pam_name, pam_obj_to_show, samples_metric)
            with c2:
                show_samples_charts(kms_name, kms_obj_to_show, samples_metric)
        else:
            show_samples_charts(pam_name, pam_obj_to_show, samples_metric)

    with g_tab:
        def show_cluster_charts(name, clm_obj):
            st.caption(name)
            data_df = clm_obj.cluster_metrics_df.apply(lambda x: x / x.max())
            st.plotly_chart(set_up_h_bar_chart(data_df, axis=1, flip_color=True))

        if kms_true:
            c1, c2 = st.columns(2)
            with c1:
                show_cluster_charts(pam_name, pam_obj_to_show)
            with c2:
                show_cluster_charts(kms_name, kms_obj_to_show)
        else:
            show_cluster_charts(pam_name, pam_obj_to_show)

    with d_tab:
        c1, c2 = st.columns(2)
        c1.caption(pam_name)
        c2.caption(kms_name)
        c1.dataframe(pam_obj_to_show.cluster_metrics_df)
        c2.dataframe(kms_obj_to_show.cluster_metrics_df) if kms_true else None
        c1.dataframe(pam_obj_to_show.samples_metrics_df)
        c2.dataframe(kms_obj_to_show.samples_metrics_df) if kms_true else None

    if kms_true:
        with c_tab:
            ct_res = pam.print_pd_crosstab(labels=pam_obj_to_show.labels, target_labels=kms_obj_to_show.labels)
            ct_max = ct_res.max()
            ct_sum = ct_res.sum()
            ct_stat = pd.DataFrame([ct_max, ct_sum, ct_max/ct_sum], index=["max", "sum", "percent"])
            print(ct_res)
            # ct_res = pd.concat([ct_res, ct_stat])])
            st.dataframe(ct_res, use_container_width=True)
            st.dataframe(ct_stat, use_container_width=True)


def all_metrics_data_charts(n_cl_obj):
    pam_1, pam_2, pam_3, pam_4 = st.tabs(
        ("DataFrame", "Line Charts", "min_max_scaled charts", "Standardized charts")
    )
    with pam_1:
        st.dataframe(n_cl_obj.n_cl_metrics)
    with pam_2:
        st.plotly_chart(multiline_chart_from_df_columns(
            df=n_cl_obj.n_cl_metrics,
            x_name="k-number of clusters",
            title="PAM metrics"
        ))
    with pam_3:
        st.plotly_chart(multiline_chart_from_df_columns(
            df=n_cl_obj.n_cl_metrics,
            apply_func=min_max_scale_sr,
            x_name="k-number of clusters",
            title="PAM metrics"
        ))
    with pam_4:
        st.plotly_chart(multiline_chart_from_df_columns(
            df=n_cl_obj.n_cl_metrics,
            apply_func=standardize_sr,
            x_name="k-number of clusters",
            title="PAM metrics"
        ))


def show_3d_plots(data_df: pd.DataFrame, labels_sr: pd.Series, select_redux: Literal["PCA", "LDA"], name: str = None):
    # if not list_val_df it returns the df unchanged
    # pam_data = explode_list_val_df(pam_n_cl_obj.data)
    if select_redux == "LDA":
        n_cl = len(labels_sr.unique())
        n_ft = data_df.shape[1]
        lda_bl = 3 > min([n_ft, n_cl - 1])
        if lda_bl:
            st.info(fr"""
                    LDA can not be performed.
                     - n_clusters: {n_cl} 
                     - n_ft: {n_ft}
                    """)
            return

    fig1 = set_up_3d_graph_data(
        data=data_df,
        labels=labels_sr,
        select_redux=select_redux
    )
    st.plotly_chart(fig1, use_container_width=True, theme="streamlit")
    if name:
        st.caption(f"{name} graph")


def dim_redux_3d_plots_target_labels_comp(data_obj: ClRes):
    input_obj = data_obj.input_obj
    cl_stg_obj_dict = data_obj.cl_stg_obj_dict

    n_cl_method_key = st.selectbox(
        label="Select centroid_clustering method", options=list(cl_stg_obj_dict.keys()), index=len(cl_stg_obj_dict.keys()) - 1
    )
    n_cl_s_obj = cl_stg_obj_dict[n_cl_method_key]

    if input_obj.target_labels is None:
        st.warning("Target labels have not been specified")
        return

    dim_reduction_3d_plots(
        pam_n_cl_obj=n_cl_s_obj.n_cl_obj,
        kms_n_cl_obj=input_obj.target_labels,
        pam_name=n_cl_method_key,
        kms_name="Target labels"
    )


def dim_reduction_3d_plots(
        pam_n_cl_obj: pam.ClSelect,
        kms_n_cl_obj: pam.ClSelect | pd.Series | None = None,
        pam_name=None,
        kms_name=None
):
    pam_name = "K-medoids" if pam_name is None else pam_name
    kms_name = "K-means" if kms_name is None else kms_name

    col_1, col_2 = st.columns((1, 1))
    with col_1:
        select_n_cl = st.selectbox(label="Number of n_clusters", options=pam_n_cl_obj.labels_df.columns)
        con1 = st.container()
    with col_2:
        select_redux = st.selectbox(label="Dimension reduction method", options=["PCA", "LDA"])
        con2 = st.container()

    def show_3d_plots_p(n_cl_obj, name):
        # if not list_val_df it returns the df unchanged
        # pam_data = explode_list_val_df(pam_n_cl_obj.data)
        show_3d_plots(
            data_df=n_cl_obj.data,
            labels_sr=n_cl_obj.labels_df[select_n_cl],
            select_redux=select_redux,
            name=name
        )

    if kms_n_cl_obj is not None:
        with con1:
            show_3d_plots_p(pam_n_cl_obj, pam_name)
        with con2:
            if isinstance(kms_n_cl_obj, pd.Series):
                show_3d_plots(
                    data_df=pam_n_cl_obj.data,
                    labels_sr=kms_n_cl_obj,
                    select_redux=select_redux,
                    name="Original labels"
                )
            else:
                show_3d_plots_p(kms_n_cl_obj, kms_name)
    else:
        show_3d_plots_p(pam_n_cl_obj, pam_name)


def create_n_clm_plotly_charts_single_curve(n_cl_obj: pam.ClSelect):
    metrics = multiselect_submit(
        label="Select metrics",
        options=n_cl_obj.metrics_for_fitting,
        default=["Silhouette"],
        label_above=True
    )
    for metric in metrics:
        st.plotly_chart(n_cl_obj.n_cl_m_fig[metric])

    st.dataframe(n_cl_obj.m_func_fit_difs)


def cluster_tab_results(
        pam_n_cl_obj: pam.ClSelect | ClSettings | ClRes,
        kms_n_cl_obj: pam.ClSelect | ClSettings | None = None,
        pam_name="K-medoids",
        kms_name="K-means",
        **kwargs
):
    print("\n\npamp_line_1163\n\ncluster_tab_results\n\n")
    kms_true = kms_n_cl_obj is not None

    if not isinstance(pam_n_cl_obj, pam.ClSelect):
        try:
            pam_name = pam_n_cl_obj.cl_name
            pam_n_cl_obj = pam_n_cl_obj.n_cl_obj
            if kms_true:
                kms_name = kms_n_cl_obj.cl_name
                kms_n_cl_obj = kms_n_cl_obj.n_cl_obj
        except AttributeError:
            pass

    menu_options = [
            "Fitting curve graphs",
            f"{pam_name} metrics",
            ] + ([f"{kms_name} metrics"] if kms_true else []) + [
            "3d plots",
            "Specific k-clusters results",
            "Single fitting curve graph check",
        ]

    with st.sidebar:
        slt_list = multiselect_submit(
            label="Render content menu",
            options=menu_options,
            default=["Fitting curve graphs", f"{pam_name} metrics"],
            label_above=True
        )

    with st.expander(label="Unified k-cluster score"):

        def change_cl_slt_for_delta():
            if r_delta_key not in st.session_state:
                return

            pam_n_cl_obj.selecting_num_of_clusters(relative_error=st.session_state[r_delta_key])
            kms_n_cl_obj.selecting_num_of_clusters(relative_error=st.session_state[r_delta_key]) if kms_true else None

        c1, c2 = st.columns(2)

        r_delta, r_delta_key = dec_key(
            c1.toggle, label="Relative delta", value=False, on_change=change_cl_slt_for_delta
        )
        flip = c2.toggle(label="Vertical view", value=False)
        if not flip:
            st.dataframe(pam_n_cl_obj.metrics_weights.to_frame().T, use_container_width=True)

            st.dataframe(
                pd.DataFrame(pam_n_cl_obj.n_cl_score, columns=[pam_name]).T, use_container_width=True
            )
            st.dataframe(
                pd.DataFrame(kms_n_cl_obj.n_cl_score, columns=[kms_name]).T, use_container_width=True
            ) if kms_true else None
        else:
            if kms_true:
                c1.dataframe(pd.DataFrame(
                    [pam_n_cl_obj.n_cl_score, kms_n_cl_obj.n_cl_score],
                    index=[f"{pam_name}", f"{kms_name}"]
                ).T, use_container_width=True)
            else:
                c1.dataframe(pam_n_cl_obj.n_cl_score.to_frame(), use_container_width=True)

            c2.dataframe(pam_n_cl_obj.metrics_weights, use_container_width=True)

    if "Fitting curve graphs" in slt_list:
        with st.expander(label="Graphs of fitting curves on n_cluster metrics", expanded=True):
            create_n_cl_metrics_plotly_multy_line_charts(pam_n_cl_obj, kms_n_cl_obj)

    if "Specific k-clusters results" in slt_list:
        with st.expander(label="See results for specified number of n_clusters"):
            specific_k_clusters_results(pam_n_cl_obj, kms_n_cl_obj, pam_name, kms_name)

    if f"{pam_name} metrics" in slt_list:
        with st.expander(label=f"{pam_name} metrics"):
            all_metrics_data_charts(pam_n_cl_obj)

    if f"{kms_name} metrics" in slt_list:
        with st.expander(label=f"{kms_name} metrics"):
            all_metrics_data_charts(kms_n_cl_obj)

    if "3d plots" in slt_list:
        with st.expander(label="Dimension reduction 3d plot"):
            dim_reduction_3d_plots(pam_n_cl_obj, kms_n_cl_obj, pam_name, kms_name)

    if "Single fitting curve graph check" in slt_list:
        with st.expander("See fitting results for single curve"):
            if kms_true:
                n_cl_ob_toggle = st.toggle(label="Show k-means instead of k-medoids", value=False)
                n_cl_obj = kms_n_cl_obj if n_cl_ob_toggle else pam_n_cl_obj
            else:
                n_cl_obj = pam_n_cl_obj
            create_n_clm_plotly_charts_single_curve(n_cl_obj)
