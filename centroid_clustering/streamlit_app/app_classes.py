import pandas as pd
import streamlit as st

from dataclasses import dataclass, field
from typing import Callable
from functools import partial

from centroid_clustering import clustering_selection as pam
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
        n_cl_obj = pam.ClSelect(
            data=data,
            cl_metrics_obj_func=partial(pam.cl_metrics_set_up_for_kms_obj, **kwargs),
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
        n_cl_obj = pam.ClSelect(
            data=data,
            cl_metrics_obj_func=partial(pam.cl_metrics_set_up_for_faster_pam, **kwargs),
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
    n_cp_cand: int = None
    g_median: bool = False
    p_mean_ord: bool = None
    # cand_options: Optional[dict[str, int | bool | None]] = None
    iter_cp_comb: int = 200     # 500
    max_iter: int = 30  # 50

    n_init: int = 1     # 3
    st_p_method: str = "convex_hull"

    alg_name: str = field(init=False, default="Custom k-medoids")

    @property
    def cand_options(self):
        return {
            "n_cp_cand": self.n_cp_cand,
            "g_median": self.g_median,
            "p_mean_ord": self.p_mean_ord
        }

    @classmethod
    def set_up(cls, cl_name, data, max_iter=20, min_n_cl=2, max_n_cl=20, **kwargs):

        n_cl_obj = pam.ClSelect(
            data=data,
            cl_metrics_obj_func=partial(pam.cl_metrics_set_up_for_k_medoids, **kwargs),
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
        if placement:
            c4 = st.container()
            st.divider()
            c6 = c5 = st.container()
        else:
            c4, c5, c6 = st.columns((1, 1, 1))

        dists_norm, dists_norm_key = dec_key(
            func=c4.select_slider,
            label="p-norm (order) to aggregate distances",
            options=list(norm_format_dict.keys()),
            format_func=lambda x: norm_format_dict.get(x),
            value=self.dists_norm
        )
        iter_cp_comb, iter_cp_comb_key = dec_key(
            func=c4.slider,
            label="Max number of center-point sets checked per iteration",
            min_value=50,
            max_value=5000,
            step=5,
            value=self.iter_cp_comb
        )

        cp_cand_dict = {None: "No candidates", 1: "1", 3: "3", 5: "5", 10: "10", 20: "20", 50: "50"}

        cp_cand, n_cp_cand_key = dec_key(
            func=c5.select_slider,
            label="Candidates close to mean",
            options=cp_cand_dict.keys(),
            format_func=cp_cand_dict.get,
            value=self.n_cp_cand if isinstance(self.n_cp_cand, int) else None
        )

        p_mean_ord, p_mean_ord_key = dec_key(
            func=c5.slider,
            label="Imaginary mid point approximation with power mean",
            min_value=-2.0, max_value=4.0, step=0.5, help="Value -1: Harmonic mean, 0: Geometric mean",
            value=self.p_mean_ord
        )
        g_median, g_median_key = dec_key(
            func=c6.toggle,
            label="Imaginary mid point approximation with geometric median",
            value=self.g_median,
            help="""Overwrites previous setting of power mean 
                    and uses the geometric median for approximating 
                    the medoid of the clusters"""
        )
        return dict(
            dists_norm_key=dists_norm_key,
            iter_cp_comb_key=iter_cp_comb_key,
            n_cp_cand_key=n_cp_cand_key,
            p_mean_ord_key=p_mean_ord_key,
            g_median_key=g_median_key
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
