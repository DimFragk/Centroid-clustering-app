import numpy as np
import streamlit as st

from typing import Callable
from functools import partial

# from centroid_clustering.src import clustering_selection as pam
from centroid_clustering import clustering_selection as pam

from utils.general_functions import def_var_value_if_none, rtn_dict_update, type_check
import utils.streamlit_functions as Ui
import utils.streamlit_app_stracture as sts

from .app_classes import PAMInput, ClRes, ClStgKmeans, ClStgFasterPam, ClStgCustomKMedoids
from .result_pages import dim_redux_3d_plots_target_labels_comp, specific_k_cluster_target_labels_comp
from .result_pages import cluster_tab_results
from .input_options import from_file_to_pam_input_obj, create_random_data


def general_cluster_tab_settings_inside_form(
        n_cl_s_obj: ClStgFasterPam | ClStgCustomKMedoids,
        dec_key: Callable,
        container_1,
        container_2,
        norm_format_dict: dict,
        dist_metrics_options: list[str] | None = None,
):

    c1 = container_1
    c2 = container_2

    name_of_change, cl_name_key = dec_key(
        func=c1.text_input,
        label="Add a name for this centroid_clustering",
        value=f"{n_cl_s_obj.alg_name}_:_v1.1"
    )
    (min_n_cl, max_n_cl), min_max_key = dec_key(
        func=c1.select_slider,
        label="Select min and max number of n_clusters",
        options=list(range(2, 31)),
        value=[n_cl_s_obj.min_n_cl, n_cl_s_obj.max_n_cl]
    )

    cl_s_obj_d_opt = def_var_value_if_none(
        value_passed=dist_metrics_options, default=list(pam.DistFunction.dist_func_1d_dict.keys())
    )

    slt_dist, dist_metric_key = dec_key(
        func=c2.selectbox,
        label="Select distance function",
        options=cl_s_obj_d_opt,
        # gdss_cl_obj.dist_metric_list,
        index=cl_s_obj_d_opt.index(n_cl_s_obj.dist_metric)
    )
    p_norm, norm_ord_key = dec_key(
        func=c2.select_slider,
        label="Define p-norm",
        options=list(norm_format_dict.keys()),
        format_func=norm_format_dict.get,
        value=n_cl_s_obj.norm_ord,
        help="Only used if a norm distance function is selected. Defines the norm order for the difference vector"
    )

    return dict(
        cl_name_key=cl_name_key,
        min_max_key=min_max_key,
        dist_metric_key=dist_metric_key,
        norm_ord_key=norm_ord_key
    )


def cluster_tab_setting_inside_form(
        n_cl_s_obj: ClStgFasterPam | ClStgFasterPam | ClStgKmeans,
        placement: bool, input_settings_func: Callable = None
):
    if placement is None:
        return

    if callable(getattr(n_cl_s_obj, "cluster_tab_setting_inside_form", None)):
        return n_cl_s_obj.cluster_tab_setting_inside_form(
            placement=placement, input_settings_func=input_settings_func
        )
    # cl_m_obj: pam.ClMetrics = mf.get_dict_0key_val(n_cl_obj.res_n_cl_obj_dict)
    dec_key = Ui.init_dec_key(
        father_key=f"Inside: {n_cl_s_obj.cl_name}{n_cl_s_obj.alg_name}",
        direct_st_val=False
    )

    norm_format_dict = {
        -np.inf: "min (-inf)",
        -2: "^-2",
        -1: "^-1",
        1/2: "^1/2 (sqrt)",
        1: "sum(abs()) (l1 norm)",
        2: "euclidean (l2 norm)",
        3: "l3 norm",
        4: "l4 norm",
        np.inf: "max (inf norm)"
    }

    t1, t2 = st.tabs([
        Ui.latext_size("Clustering settings", size=4),
        Ui.latext_size(f"{n_cl_s_obj.alg_name} settings", size=4)
    ])

    if placement:
        c1 = t1.container()
        c2 = t1.container()
        t1.divider()
        c3 = t1.container()

        # c4 = c45 = t2.container()
        # t2.divider()
        # c6 = c5 = t2.container()
    else:
        # f_col_1, f_col_2, e2, f_col_3, f_col_4 = st.columns([10, 10, 2, 10, 10])
        if input_settings_func is not None:
            c1, c2, c3 = t1.columns((1, 1, 1))
        else:
            c1, c2 = t1.columns(2)
            c3 = c2
        # c45 = t2.container()
        # c4, c5, c6 = t2.columns((1, 1, 1))
    stg_dict_keys = general_cluster_tab_settings_inside_form(
        n_cl_s_obj=n_cl_s_obj,
        dec_key=dec_key,
        container_1=c1,
        container_2=c2,
        norm_format_dict=norm_format_dict,
        # dist_metrics_options=list(pam.DistFunction.dist_func_1d_dict.keys())
    )

    if callable(getattr(n_cl_s_obj, "cl_tab_stg_inside_form", None)):
        with t2:
            m_stg_dict_keys = n_cl_s_obj.cl_tab_stg_inside_form(
                placement=placement, dec_key=dec_key, norm_format_dict=norm_format_dict
            )
            stg_dict_keys.update(m_stg_dict_keys)

    # TODO: TRY TO USE FOR GDSS_OBJ
    if input_settings_func is not None:
        """
        input_data, input_data_key = dec_key(
            func=c2.selectbox,
            label="Input data for centroid_clustering",
            options=gdss_cl_obj.data_n_cl_str_list,
            index=gdss_cl_obj.data_n_cl_str_list.index(cl_s_obj.data_n_cl_str)
        )

        m_idx_p_opt_res, m_idx_p_opt_res_key = dec_key(
            func=c2.toggle,
            label="Run centroid_clustering with post-opt dm data",
            value=cl_s_obj.m_idx,
            help="Multiple post optimization solution vector points for each decision maker,\n"
                 "every point is an post optimization solution. The result of the centroid_clustering is fuzzy,\n"
                 "because points from the same dm can end up in different clusters.\n"
                 "The number of post-opt points is the double the number of criteria",
        )
        """
        with c3:
            i_stg_dict_keys = input_settings_func(dec_key=dec_key)
            stg_dict_keys.update(i_stg_dict_keys)

    return stg_dict_keys


def cluster_tab_setting_form_and_results(
        data_obj: PAMInput | ClRes,
        show_results_fn: Callable,
        show_settings=True,
        placement=False
):
    print("pamp_line_547")
    print(st.session_state.settings_placement)
    print(type(data_obj))
    print(type(data_obj).__name__)

    if type_check(data_obj, ClRes):   # isinstance(data_obj, ClRes):
        input_obj = data_obj.input_obj
        cl_stg_obj_dict = data_obj.cl_stg_obj_dict
    elif type_check(data_obj, PAMInput):
        input_obj = data_obj
        cl_stg_obj_dict = {}
    else:
        raise Exception(f"False input 'data_obj' of type {type(data_obj)}\n correct types are: 'PAMInput' | 'ClRes'")

    options_dict = {
        "Custom k-medoids default settings": ClStgCustomKMedoids(
            cl_name="Custom k-medoids default settings",
            data=input_obj.data
        ),
        "K-means default settings": ClStgKmeans(
            cl_name="K-means default settings",
            data=input_obj.data
        ),
        "FasterPam default settings": ClStgFasterPam(
            cl_name="FasterPam default settings",
            data=input_obj.data
        )
    }
    print("pamp_line_661")
    print(cl_stg_obj_dict.keys())
    print(options_dict.keys())
    options_dict = rtn_dict_update(options_dict, cl_stg_obj_dict, copy=True)

    many_cl = len(cl_stg_obj_dict) > 1
    if many_cl:
        c1, c2, c3 = st.columns([3, 3, 1], vertical_alignment="bottom")
    else:
        c1, c3 = st.columns([6, 1], vertical_alignment="bottom")
        c2 = c1

    n_cl_method_key = c1.selectbox(
        label="Select centroid_clustering method", options=list(options_dict.keys()), index=len(options_dict.keys())-1
    )
    n_cl_s_obj = options_dict[n_cl_method_key]

    n_cl_method_comp_key = None
    if many_cl:
        comp_options = list(filter(lambda x: n_cl_method_key != x, list(cl_stg_obj_dict.keys())))
        n_cl_method_comp_key = c2.selectbox(
            label="Comparison centroid_clustering method", options=comp_options
        )

    show_settings = c3.toggle("Show settings", value=show_settings, key="Toggle_show_settings")

    cont = st.container()

    if n_cl_s_obj.n_cl_obj is not None:
        kms_n_cl_s_obj = cl_stg_obj_dict[n_cl_method_comp_key] if n_cl_method_comp_key else None
        show_results_fn(
            pam_n_cl_obj=n_cl_s_obj.n_cl_obj,
            kms_n_cl_obj=kms_n_cl_s_obj.n_cl_obj if n_cl_method_comp_key else None,
            pam_name=n_cl_s_obj.cl_name,
            kms_name=kms_n_cl_s_obj.cl_name if n_cl_method_comp_key else None
        )

    if not show_settings:
        return

    with cont:
        form_cl = Ui.create_form_to_sidebar_if(
            s_state_bool=placement,
            form_key="Clustering settings"
        )

    with form_cl:
        stg_dict_keys = cluster_tab_setting_inside_form(
            n_cl_s_obj=n_cl_s_obj, placement=placement
        )

        print("pamp_line_590")
        print(stg_dict_keys.keys())

        def st_f(key):
            print("pamp_line_594")
            print(stg_dict_keys.keys())
            print(stg_dict_keys[key])
            return st.session_state[stg_dict_keys.pop(key)]

        def set_up_func(d_obj: PAMInput | ClRes):
            # NEEDS THE INPUT "d_obj: PAMInput" BECAUSE OF THE "sts.submit_set_up" FUNCTION

            print("pam_app_line_548")
            print(stg_dict_keys)

            cl_name = st_f("cl_name_key")
            new_n_cl_s_obj = n_cl_s_obj.set_up(
                cl_name=cl_name,
                data=d_obj.data,
                **dict(zip(["min_n_cl", "max_n_cl"], st_f("min_max_key"))),
                **{k[:-4]: st.session_state[stg_dict_keys[k]] for k in stg_dict_keys}
            )

            if type_check(data_obj, PAMInput):
                cl_res_obj = ClRes(input_obj=input_obj, cl_stg_obj_dict={cl_name: new_n_cl_s_obj})
                return cl_res_obj, data_obj.f_name
            else:
                data_obj.cl_stg_obj_dict[cl_name] = new_n_cl_s_obj

        def submit_fn():
            if type_check(data_obj, PAMInput):
                sts.submit_set_up(rtn_method_obj_func=set_up_func, method_input=input_obj)
            else:
                set_up_func(input_obj)

        if st.form_submit_button(
                label="Run Clustering",
                type="primary",
                on_click=submit_fn
        ):
            print("pam_app_line_576")
            print(stg_dict_keys)
        else:
            return None, None


def home_page():
    st.header("Welcome to the centroid clustering app", divider="rainbow")
    st.markdown(
        r"""
        The goal of the centroid based clustering app, is to support the selection
        of the "best" among many different clustering versions/attempts, on the same input data. 
        
        There are 2 main decision parameters: 
        - The clustering method (including settings parameters)
        - The number of clusters (k)
        
        The project source code can be found in ([github](https://github.com/DimFragk/Centroid-clustering-app))
        """
    )
    st.divider()
    st.subheader("Clustering algorithms", divider="grey")
    st.markdown(
        r"""
        ###### The app has implemented 2 centroid_clustering methods:
        - k-medoids
        - k-means
        
        ###### The k-medoids centroid_clustering algorithms used are:  
        1. 'fasterpam': One of the pam algorithms from the python library "kmedoids"
        ([pypi.org](https://pypi.org/project/kmedoids/)) 
        ([github](https://github.com/kno10/python-kmedoids)) 
        2. 'Custom kmedoids': My attempt at creating a k-medoids centroid_clustering algorithm, it is included in this project 
        
        ###### The k-means centroid_clustering algorithm used is:
        1. 'KMeans': from the python library scikit-learn
        ([Documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html))
        """
    )
    st.divider()
    st.subheader("General Methodology", divider="grey")
    st.write(
        """
        For selecting the best centroid_clustering algorithm, the app runs the selected clustering algorithm 
        for every number of clusters from 'k = 2' to 'k = 20', so 19 total clustering versions/attempts. 
        
        Then it evaluates each centroid_clustering using 4 metrics:
        - Inertia
        - Distortion
        - Silhouette score
        - calinski harabasz index
        
        After the evaluation, an approximate polynomial interpolation (fitting function)
        is selected to represent the expected value of each metric 
        (for polynomial orders less than half of the data points).
        The goal is to find the maximum difference of the fitting function and each metric.
        
        Finally, a combined clustering index is calculated by the weighted sum of relative differences, 
        for each centroid_clustering with different 'k'.
        
        This process is repeated when selecting a different clustering algorithm, 
        and the results are shown for 2 centroid_clustering algorithms at a time, for easier comparisons.    
        """
    )
    st.divider()
    st.subheader("System input data", divider="grey")
    st.write(
        """
        The system input data are:
        - The data matrix, with samples for row and features/criteria for columns (N X M)
        - The labels array (or target labels) (Optional), 
            with values representing the cluster each sample belongs to (N X 1)
        - The centers matrix (optional), 
            the samples or imaginary points pre-selected as the cluster target centers (k X M)
            
        There are 3 data options for running the app:
        - With a template excel file (.xls, .xlsx).
        - With a '.csv' file
        - With 3 random random data generation options for demo/testing
        """
    )

    c1, c2 = st.columns((1, 4), vertical_alignment="top")
    with c2.expander("Template file guide"):
        st.write(
            """
            - If there are no target labels and centers, 
            the first sheet of the excel, or the sheet with name 'Samples', 
            is considered as the samples data. 
            - The first row and columns of the sheet is always considered as index.
            - For the input to work, do NOT modify the names of the template sheet, 
            only delete unused sheets 'Labels' or 'Centers.
            - If the centers are selected from the samples and the 'Centers' sheet is does not exist, 
            centers are inferred from the labels unique values as the samples index.
            """
        )

    win_f_path = """excel_templates_clustering/clustering_input_template.xlsx"""

    with open(win_f_path, "rb") as template_file:
        template_byte = template_file.read()
        c1.download_button(
            label="Download excel template",
            data=template_byte,
            file_name="template.xlsx",
            mime='application/octet-stream'
        )

    st.divider()


def main():
    page_tabs_data = [
        sts.PageTab(
            tab_name="Clustering results",
            tab_page=partial(cluster_tab_setting_form_and_results, show_results_fn=cluster_tab_results),
            tab_icon="diagram-3",
            tab_setting=None,
        ),
        sts.PageTab(
            tab_name="3d plot comparison with target labels",
            tab_page=dim_redux_3d_plots_target_labels_comp,
            tab_icon="transparency",
            tab_setting=None,
        ),
        sts.PageTab(
            tab_name="Metrics comparison with target labels",
            tab_page=specific_k_cluster_target_labels_comp,
            tab_icon="arrows-angle-contract",
            tab_setting=None,
        ),
    ]

    sts.main(
        page_tabs_data=page_tabs_data,
        from_file_to_input_obj=from_file_to_pam_input_obj,
        random_data_gen=create_random_data,
        input_obj_set_up=partial(cluster_tab_setting_form_and_results, show_results_fn=cluster_tab_results),
        home_page=home_page,
        r_xl_type="bytes_io",
        submit_set_up_needed=False
    )


if __name__ == "__main__":
    st.set_page_config(
        page_title="Clustering app",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            'Get Help': "",
            'Report a bug': "",
            'About': "Welcome to the Large Group Decision Support System (LGDSS) app!"
        }
    )
    main()

