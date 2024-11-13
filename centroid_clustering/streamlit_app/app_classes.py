import pandas as pd
import streamlit as st

from dataclasses import dataclass, field
from typing import Callable
from functools import partial

from centroid_clustering import clustering_selection as pam
from centroid_clustering.clustering_metrics import DistFunction, ClMetrics
import centroid_clustering.custom_k_medoids as ckm
from utils import streamlit_functions as Ui


@dataclass(slots=True)
class ClSettings:
    cl_name: str
    data: pd.DataFrame

    dists_norm: int = 1
    min_n_cl: float = 2
    max_n_cl: float = 20

    dist_metric: str = "euclidean"
    norm_ord: int = 2

    n_cl_obj: pam.ClSelect = field(default=None, repr=False)
    dist_func_obj: pam.DistFunction = field(default=None, repr=False)
    cl_m_slt: pam.ClMetrics = field(init=False, default=None, repr=False)

    cl_m_func: Callable = field(init=False, default=None, repr=False)

    def __post_init__(self):
        if self.n_cl_obj is not None:
            self.cl_m_func = self.n_cl_obj.clm_obj_func
            self.cl_m_slt = self.n_cl_obj.selected_clm_obj()


@dataclass(slots=True)
class ClStgKmeans(ClSettings):
    max_iter: int = 300
    algorithm: str = "lloyd"

    alg_name: str = field(init=False, default="k-means")

    def cluster_tab_setting_inside_form(
            self,
            placement: bool,
            input_settings_func: Callable = None
    ):
        dec_key = Ui.init_dec_key(
            father_key=f"Inside: {self}",
            direct_st_val=False
        )

        t1, t2 = st.tabs([
            Ui.latext_size("Clustering settings", size=4),
            Ui.latext_size(f"{self.alg_name} settings", size=4)
        ])

        if placement:
            c1 = c2 = t1.container()
            t1.divider()
            c0 = t1.container()
            c3 = t2.container()
            c4 = t2.container()
        else:
            if input_settings_func is not None:
                c1, c0 = t1.columns(3)
                c2 = c1
            else:
                c1, c2 = t1.columns(2)
                c0 = t1
            c3, c4 = t2.columns(2)

        stg_dict_keys = {}

        name_of_change, stg_dict_keys["cl_name_key"] = dec_key(
            func=c1.text_input,
            label="Add a name for this centroid_clustering",
            value=f"{self.alg_name}_:_v1.1"
        )
        (min_n_cl, max_n_cl), stg_dict_keys["min_max_key"] = dec_key(
            func=c2.select_slider,
            label="Select min and max number of n_clusters",
            options=list(range(2, 31)),
            value=[self.min_n_cl, self.max_n_cl]
        )

        max_v, stg_dict_keys["max_iter_key"] = dec_key(
            c3.number_input, label="Max number of iterations", max_value=1000, value=self.max_iter
        )

        algorithm, stg_dict_keys["algorithm_key"] = dec_key(
            c4.selectbox,
            label="K-means algorithm to use",
            options=["lloyd", "elkan"],
            index=["lloyd", "elkan"].index(self.algorithm)
        )

        if input_settings_func is not None:
            with c0:
                i_stg_dict_keys = input_settings_func(dec_key=dec_key)
                stg_dict_keys.update(i_stg_dict_keys)

        return stg_dict_keys

    @classmethod
    def set_up(cls, cl_name, data, max_iter, min_n_cl=2, max_n_cl=20, **kwargs):
        dist_func_obj=DistFunction(
            dist_metric="norm^p",
            norm_order=2,
            cache_points=data
        )
        n_cl_obj = pam.ClSelect(
            data=data,
            cl_metrics_obj_func=partial(pam.cl_metrics_set_up_for_kms_obj, dist_func_obj=dist_func_obj, **kwargs),
            n_iter=max_iter,
            min_n_cl=min_n_cl,
            max_n_cl=max_n_cl
        )
        return cls(
            cl_name=cl_name,
            data=data,
            n_cl_obj=n_cl_obj,
            max_iter=max_iter,
            min_n_cl=min_n_cl,
            max_n_cl=max_n_cl,
            **kwargs
        )


@dataclass(slots=True)
class ClStgFasterPam(ClSettings):
    max_iter: int = 100
    init: any = "random"
    random_state: any = None

    alg_name: str = field(init=False, default="FasterPam")

    @classmethod
    def set_up(cls, cl_name, data, max_iter=20, min_n_cl=2, max_n_cl=20, **kwargs):
        dist_func_obj=DistFunction(
            dist_metric=kwargs.pop("dist_metric"),
            norm_order=kwargs.pop("norm_ord"),
            cache_points=data
        )
        n_cl_obj = pam.ClSelect(
            data=data,
            cl_metrics_obj_func=partial(pam.cl_metrics_set_up_for_faster_pam, dist_func_obj=dist_func_obj, **kwargs),
            n_iter=max_iter,
            min_n_cl=min_n_cl,
            max_n_cl=max_n_cl
        )
        return cls(
            cl_name=cl_name,
            data=data,
            n_cl_obj=n_cl_obj,
            max_iter=max_iter,
            min_n_cl=min_n_cl,
            max_n_cl=max_n_cl,
            **kwargs
        )

    def cl_tab_stg_inside_form(self, placement: bool, dec_key, norm_format_dict):
        max_v, max_iter_key = dec_key(
            st.number_input, label="Max number of iterations", max_value=1000, value=self.max_iter
        )
        return dict(max_iter_key=max_iter_key)


@dataclass(slots=True)
class ClStgCustomKMedoids(ClSettings):
    im_mid_enabled: bool = False
    n_cp_cand: int = 0
    im_mid_type: str = "g_median"
    # cand_options: Optional[dict[str, int | bool | None]] = None

    iter_cp_comb: int = 200     # 500
    max_iter: int = 30  # 50

    n_init: int = 1     # 3
    st_p_method: str = "convex_hull"
    dim_redux: int = None

    alg_name: str = field(init=False, default="Custom k-medoids")

    @classmethod
    def set_up(cls, cl_name, data, max_iter=20, min_n_cl=2, max_n_cl=20, **kwargs):
        dist_func_obj = DistFunction(
            dist_metric=kwargs.pop("dist_metric"),
            norm_order=kwargs.pop("norm_ord"),
            cache_points=data
        )

        cand_options = {
            "n_cp_cand": kwargs.pop("n_cp_cand"),
            "im_mid_type": kwargs.pop("im_mid_type")
        }
        if not kwargs.pop("im_mid_enabled"):
            cand_options = None

        def cl_m_c_kmedoids(data, n_clusters, max_iter):
            return ClMetrics.from_k_medoids_obj(ckm.Kmedoids(
                data=data,
                n_clusters=n_clusters,
                max_iter=max_iter,
                dist_func_obj=dist_func_obj,
                cand_options=cand_options,
                **kwargs
            ))

        n_cl_obj = pam.ClSelect(
            data=data,
            cl_metrics_obj_func=cl_m_c_kmedoids,
            n_iter=max_iter,
            min_n_cl=min_n_cl,
            max_n_cl=max_n_cl
        )
        return cls(
            cl_name=cl_name,
            data=data,
            n_cl_obj=n_cl_obj,
            max_iter=max_iter,
            min_n_cl=min_n_cl,
            max_n_cl=max_n_cl,
            **kwargs
        )

    def cl_tab_stg_inside_form(
            self, placement: bool, dec_key, norm_format_dict
    ):
        def placement_fn(n):
            if placement:
                return (st.container(), ) * n
            else:
                return st.columns(n, gap="medium", vertical_alignment="bottom")

        c4, c5 = placement_fn(2)

        dists_norm, dists_norm_key = dec_key(
            func=c4.select_slider,
            label="p-norm (order) to aggregate distances",
            options=list(norm_format_dict.keys()),
            format_func=lambda x: norm_format_dict.get(x),
            value=self.dists_norm
        )
        iter_cp_comb, iter_cp_comb_key = dec_key(
            func=c5.slider,
            label="Max number of center-point sets checked per iteration",
            min_value=50,
            max_value=5000,
            step=5,
            value=self.iter_cp_comb
        )

        st.divider()
        c4, c5, c6 = placement_fn(3)

        n_init, n_init_key = dec_key(
            func=c4.slider,
            label="Number of independent runs from different starting points",
            min_value=1,
            max_value=10,
            step=1,
            value=self.n_init
        )

        dim_redux_dict = {None: "No operation"}
        dim_redux_dict.update({i: str(f"Dim: {i}") for i in range(3, self.data.shape[1])})
        print("\n")
        print(dim_redux_dict)
        print(list(dim_redux_dict.keys()).index(self.dim_redux))
        dim_redux, dim_redux_key = dec_key(
            func=c5.select_slider,
            label="Dimension reduction with PCA",
            options=list(dim_redux_dict.keys()),
            format_func=dim_redux_dict.get,
            value=self.dim_redux
        )

        st_p_method, st_p_method_key = dec_key(
            func=c6.selectbox,
            label="Select center points initialization method",
            options=[None, "convex_hull", "kmeans++"],
            format_func=lambda x: "Random" if x is None else x,
            index=[None, "convex_hull", "kmeans++"].index(self.st_p_method)
        )

        st.divider()
        c4, c5, c6 = placement_fn(3)

        im_mid_enabled, im_mid_enabled_key = dec_key(
            func=c4.toggle,
            label="Enable center point approximations",
            value=self.im_mid_enabled
        )

        cp_cand_dict = {0: "No candidates", 1: "1", 3: "3", 5: "5", 10: "10", 20: "20", 50: "50"}

        im_mid_fn_list: list[str] = list(ckm.PamCore.im_mid_funcs_dict.keys())
        im_mid_type, im_mid_type_key = dec_key(
            func=c6.selectbox,
            label="Imaginary center point approximation method (mean value types)",
            options=im_mid_fn_list,
            index=im_mid_fn_list.index(self.im_mid_type)
        )

        cp_cand, n_cp_cand_key = dec_key(
            func=c5.select_slider,
            label="Candidates close to mean",
            options=list(cp_cand_dict.keys()),
            format_func=cp_cand_dict.get,
            value=list(cp_cand_dict.keys()).index(self.n_cp_cand),
            # value=self.n_cp_cand if isinstance(self.n_cp_cand, int) else None
            help="""
                    The closest 'n' candidates to the imaginary center will sorted according to their cluster center cost.
                    If no candidates are selected, then all points are sorted directly according to their distance 
                    from the imaginary center.
                    """
        )

        return dict(
            dists_norm_key=dists_norm_key,
            iter_cp_comb_key=iter_cp_comb_key,
            n_init_key=n_init_key,
            st_p_method_key=st_p_method_key,
            dim_redux_key=dim_redux_key,
            im_mid_enabled_key=im_mid_enabled_key,
            im_mid_type_key=im_mid_type_key,
            n_cp_cand_key=n_cp_cand_key
        )


@dataclass
class PAMInput:
    f_name: str
    data: pd.DataFrame
    target_labels: pd.Series = field(default=None)
    center_points: pd.DataFrame = field(default=None)


@dataclass
class ClRes:
    input_obj: PAMInput
    cl_stg_obj_dict: dict[str, ClStgCustomKMedoids | ClStgKmeans | ClStgFasterPam]
