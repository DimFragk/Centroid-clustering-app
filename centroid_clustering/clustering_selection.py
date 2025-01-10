from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import load_iris, make_blobs
from scipy.optimize import curve_fit
from scipy import odr
import plotly.graph_objects as go
from kmedoids import fasterpam

import math
from functools import partial
from typing import Callable, Protocol

import utils.general_functions as mf
import utils.math_functions as gmf
import utils.pandas_functions as gpd
import utils.data_processing as gdp
import utils.visualization_functions as gvp

from centroid_clustering.custom_k_medoids import Kmedoids, k_medoids_range
from centroid_clustering.clustering_metrics import DistFunction, ClMetrics

# from __future__ import annotations
# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)


class ClMetricsRange(Protocol):
    def run_cl_metrics_obj_for_n_cl(self, min_n_cl: int, max_n_cl: int) -> dict[str, ClMetrics]:
        ...


@dataclass
class CKMedoidsRange:
    custom_k_medoids_set_up_func: Callable[[int, list | np.ndarray | pd.Index | pd.MultiIndex], Kmedoids]

    def run_cl_metrics_obj_for_n_cl(self, min_n_cl, max_n_cl):
        ckm_dict = k_medoids_range(
            set_up_k_medoids=self.custom_k_medoids_set_up_func,
            min_n_cl=min_n_cl,
            max_n_cl=max_n_cl,
            ascending=False
        )
        clm_ckm_obj_dict = {f"Cl({key})": ClMetrics.from_k_medoids_obj(ckm_obj) for key, ckm_obj in ckm_dict.items()}
        return clm_ckm_obj_dict


class ClSelect:
    MetricsTree = mf.MetricsTree
    TreeWeights = MetricsTree(
        metric_name="all",
        percent=1.0,
        sub_obj_list=[
            MetricsTree(
                metric_name="dec_val_m",
                percent=0.4,
                sub_obj_list=[
                    MetricsTree(
                        metric_name="Distortion",
                        percent=0.5
                    ),
                    MetricsTree(
                        metric_name="inertia",
                        percent=0.5,
                        sub_obj_list=[
                            MetricsTree(
                                metric_name="Inertia",
                                percent=0.5
                            ),
                            MetricsTree(
                                metric_name="Method Inertia",
                                percent=0.5
                            )
                        ]
                    )
                ]
            ),
            MetricsTree(
                metric_name="inc_val_m",
                percent=0.6,
                sub_obj_list=[
                    MetricsTree(
                        metric_name="silhouette",
                        percent=0.6,
                        sub_obj_list=[
                            MetricsTree(
                                metric_name="Silhouette.sklearn",
                                percent=1 / 3
                            ),
                            MetricsTree(
                                metric_name="Simplified Silhouette",
                                percent=1 / 3,
                            ),
                            MetricsTree(
                                metric_name="Silhouette",
                                percent=1 / 3,
                            )
                        ]
                    ),
                    MetricsTree(
                        metric_name="cal.har.d_metric",
                        percent=0.4,
                        sub_obj_list=[
                            MetricsTree(
                                metric_name="Cal.Har.d_metric",
                                percent=0.5
                            ),
                            MetricsTree(
                                metric_name="Cal.Har.sklearn",
                                percent=0.5
                            )
                        ]
                    )
                ]
            ),
        ]
    )
    """
    metrics_w = pd.Series({
        "Distortion": 0.20,  # 0.15
        "Inertia": 0.10,  # 0.075
        "Method Inertia": 0.10,  # 0.075
        "Silhouette.sklearn": 0.175,  # 0.4
        "Simplified Silhouette": 0.175,
        "Cal.Har.d_metric": 0.25,  # 0.15
        # "Cal.Har.sklearn": 0.15,  # 0.15
    })
    """

    increasing_val_metric_names = [
        "Silhouette.sklearn",
        "Simplified Silhouette",
        "Silhouette",
        "Cal.Har.d_metric",
        "Cal.Har.sklearn",
    ]
    decreasing_val_metric_names = [
        "Distortion",
        "Inertia",
        "Method Inertia",
    ]

    def __init__(
            self,
            cl_metrics_k_gen: ClMetricsRange = None,
            cl_metrics_obj_func: Callable = None,
            min_n_cl=2, max_n_cl=10,
            target_labels=None,
            metrics_weights: list | dict | pd.Series | None = None,
            **clm_func_kwargs
    ):
        self.clm_obj_func: Callable[[int], ClMetrics] | None = None
        self.run_cl_metrics_obj_for_n_cl: Callable[[int, int], dict[str, ClMetrics]] = None
        """
        self.clm_obj_func = cl_metrics_obj_func if dist_func_obj is None else partial(
            cl_metrics_obj_func, dist_func_obj=dist_func_obj
        )
        """
        self.set_up_functions(cl_metrics_k_gen, cl_metrics_obj_func, clm_func_kwargs)

        # self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data, dtype="float32")
        self.min_n_cl = min_n_cl
        self.max_n_cl = max_n_cl

        self.target_labels = target_labels

        self.res_n_cl_obj_dict: dict[str | tuple, ClMetrics] = {}

        self.labels_df: pd.DataFrame | None = None

        self.n_cl_metrics = pd.DataFrame()

        self.set_up_n_cl_run(min_n_cl=self.min_n_cl, max_n_cl=self.max_n_cl)

        self.metrics_weights: pd.Series | None = None
        self.inc_val_metric_names: list[str] | None = None
        self.dec_val_metric_names: list[str] | None = None
        self.metrics_for_fitting: list[str] | None = None
        self.set_up_metrics_weights(metrics_weights)

        self.n_cl_m_fit: pd.DataFrame | None = None
        self.n_cl_m_delta: pd.DataFrame | None = None
        self.n_cl_m_r_delta: pd.DataFrame | None = None
        self.m_func_fit_difs: pd.DataFrame | None = None
        self.n_cl_m_fig: dict[str, go.Figure] = {}
        # self.create_curve_fit_data_for_n_cl_metrics()
        self.create_pol_fit_data_for_n_cl_metrics(n_cl_metrics=self.n_cl_metrics[self.metrics_for_fitting])

        self.n_cl_score: pd.Series | None = None
        self.selected_n_cl_str: str | None = None
        self.selected_n_cl: int | None = None
        self.selecting_num_of_clusters()

    def __repr__(self):
        print(self.labels_df.value_counts())
        print(self.n_cl_metrics)
        clm_name = mf.get_dict_0key_val(self.res_n_cl_obj_dict).Cl_Method.__repr__()

        clm_f = getattr(self.clm_obj_func, "__name__", __default=None)
        clm_f = clm_f if clm_f is not None else getattr(self.clm_obj_func, "name", __default=None)
        return f"ClSelect obj(clustering_f={clm_name})\n\nlabels={self.labels_df.head(10)}\n\nclm_obj_func: {clm_f})"

    def set_up_functions(self, cl_metrics_k_gen, cl_metrics_obj_func, clm_func_kwargs):
        print("pam_line_154")
        print(clm_func_kwargs)
        if cl_metrics_k_gen is None and cl_metrics_obj_func is None:
            raise Exception(
                "One of the below input parameters must be defined:"
                "\n - 'cl_metrics_range_func'"
                "\n - 'cl_metrics_obj_func'"
            )
        elif cl_metrics_k_gen is not None:
            self.clm_obj_func = None if cl_metrics_obj_func is None else partial(
                cl_metrics_obj_func, **clm_func_kwargs
            )
            self.run_cl_metrics_obj_for_n_cl = cl_metrics_k_gen.run_cl_metrics_obj_for_n_cl
        elif cl_metrics_obj_func is not None:
            self.clm_obj_func = lambda k: cl_metrics_obj_func(n_clusters=k, **clm_func_kwargs)
            self.run_cl_metrics_obj_for_n_cl = partial(
                self.run_cl_metrics_obj_for_n_cl_default, self.clm_obj_func
            )

    def set_up_metrics_weights(self, metrics_weights=None):
        metrics_names = self.n_cl_metrics.dropna().columns
        if metrics_weights is None or isinstance(metrics_weights, list):
            if isinstance(metrics_weights, list):
                metrics_names = self.n_cl_metrics[metrics_names.isin(list)].columns
            m_names, m_weights = self.TreeWeights.weights(list(metrics_names), non_zero=False)
            print("pam_line_229")
            print(m_weights)
            print(m_names)
            metrics_w = pd.Series(data=m_weights, index=m_names)
        elif isinstance(metrics_weights, pd.Series):
            metrics_w = gdp.scale_to_percent(
                pd.Series(metrics_weights, index=metrics_names).dropna(),
                norm_ord=1
            )
        elif isinstance(metrics_weights, dict):
            metrics_w = gdp.scale_to_percent(
                pd.Series(mf.filter_dict_by_keys(metrics_weights, list(metrics_names))),
                norm_ord=1
            )
        else:
            raise Exception(f"Incompatible data type: {type(metrics_weights)}")

        # return metrics_w.loc[metrics_w.index.isin(self.n_cl_metrics.columns)]
        self.metrics_weights = metrics_w
        self.inc_val_metric_names = list(metrics_w.index.intersection(self.increasing_val_metric_names))
        self.dec_val_metric_names = list(metrics_w.index.intersection(self.decreasing_val_metric_names))
        self.metrics_for_fitting = list(self.metrics_weights.index)

    @staticmethod
    def merge_dfs_and_sort(df_old, df_new, axis=1, ascending=True):
        if isinstance(df_old, (pd.DataFrame, pd.Series)) and df_old.empty:
            return df_new
        elif df_old is None:
            return df_new

        df = gpd.merge_dfs(df_old=df_old, df_new=df_new, axis=axis)
        # TODO: Examine if there are multiple runs with the same number of clusters and if it is okk
        return mf.sort_str_num_index(df=df, axis=axis, ascending=ascending)

    @classmethod
    def set_up_n_cl_metrics(cls, res_n_cl_obj_dict, n_cl_metrics_old, calc_elbow=False):
        # n_cl_metrics_new = cls.concat_dfs_axis_and_sort(n_cl_metrics_old, n_cl_metrics, axis=0)

        n_cl_metrics = mf.extract_series_from_obj_dict(res_n_cl_obj_dict, var_name="metrics_sr", axis=0)

        if isinstance(n_cl_metrics_old, pd.DataFrame) and not n_cl_metrics_old.empty:
            n_cl_metrics = cls.merge_dfs_and_sort(
                df_old=n_cl_metrics_old[list(n_cl_metrics.columns)],
                df_new=n_cl_metrics,
                axis=0
            )

        if calc_elbow:
            n_cl_metrics["Inertia elbow"] = cls.calc_inertia_angles(n_cl_metrics["Inertia"])
            n_cl_metrics["M_Inertia elbow"] = cls.calc_inertia_angles(n_cl_metrics["Method Inertia"])

        return n_cl_metrics

    def set_up_n_cl_run(self, min_n_cl: int, max_n_cl: int):
        res_n_cl_obj_dict = self.run_cl_metrics_obj_for_n_cl(min_n_cl=min_n_cl, max_n_cl=max_n_cl)

        labels_df = mf.extract_series_from_obj_dict(res_n_cl_obj_dict, var_name="labels").T

        print("pam_Line_1144")
        print(pd.DataFrame([getattr(obj, "labels").rename(key) for key, obj in res_n_cl_obj_dict.items()]).T)
        print(labels_df)
        # self.labels_df = mf.concat_dfs_axis_and_sort(self.labels_df, new_labels, axis=1)
        self.labels_df = self.merge_dfs_and_sort(df_old=self.labels_df, df_new=labels_df, axis=1)

        self.n_cl_metrics = self.set_up_n_cl_metrics(
            res_n_cl_obj_dict=res_n_cl_obj_dict, n_cl_metrics_old=self.n_cl_metrics, calc_elbow=False
        )
        print("pam_line_2652")
        print(self.n_cl_metrics)

        self.res_n_cl_obj_dict.update(res_n_cl_obj_dict)

    def run_change_in_n_cl(self, new_min_n_cl: int, new_max_n_cl: int, rerun_all=False):
        if rerun_all:
            self.set_up_n_cl_run(min_n_cl=new_min_n_cl, max_n_cl=new_max_n_cl)
            self.min_n_cl = new_min_n_cl
            self.max_n_cl = new_max_n_cl

        if new_min_n_cl < self.min_n_cl:
            self.set_up_n_cl_run(min_n_cl=new_min_n_cl, max_n_cl=self.min_n_cl-1)
            self.min_n_cl = new_min_n_cl

        if new_max_n_cl > self.max_n_cl:
            self.set_up_n_cl_run(min_n_cl=self.max_n_cl+1, max_n_cl=new_max_n_cl)
            self.max_n_cl = new_max_n_cl

        # self.create_curve_fit_data_for_n_cl_metrics()
        self.create_pol_fit_data_for_n_cl_metrics(n_cl_metrics=self.n_cl_metrics[self.metrics_for_fitting])

        self.selecting_num_of_clusters()

    @staticmethod
    def print_crosstab_for_n_cl(kmedoids_res_n_cl_dict, target_labels):
        if target_labels is None:
            return

        print("\n\nCrosstab results of ClSelect object:\n\n")
        for n_cl in kmedoids_res_n_cl_dict:
            print(f"{print_pd_crosstab(kmedoids_res_n_cl_dict[n_cl].labels, target_labels)}\n\n")

    @staticmethod
    def run_cl_metrics_obj_for_n_cl_default(clm_obj_func: Callable[[int], ClMetrics], min_n_cl=2, max_n_cl=10):
        return {
            f"Cl({n_cl})": clm_obj_func(n_cl)
            for n_cl in range(min_n_cl, max_n_cl + 1)
        }

    @staticmethod
    def curve_functions_for_fitting(
            flt_key_list: list | str | None = None,
            not_selected=False, del_selected=False):

        curve_func_dict = {
            "exp_1": lambda x, a, b: a*np.exp((x - 0) * b) - 0,

            "xpn_1": lambda x, a, b, c: a * (x ** b),
            "xpn_2": lambda x, a, b, c: a * (x ** np.log(b)) + c * x,

            "pol_3": lambda x, a, b, c, d, e:
            a * ((x - d) ** 3) + b * ((x - d) ** 2) + c * (x - d) + e,
            "pol_4": lambda x, a, b, c:
            a * ((x - 0) ** 4) + b * ((x - 0) ** 2) + c,
            "pol_6": lambda x, a, b, c, d:
            a * ((x - c) ** 4) + b * ((x - 0) ** 3) + d,

            "ln__1": lambda x, a, b: b * np.log(x) ** a - 0,
            "ln__3": lambda x, a, b: b * np.log(x - 0) - a,

            "dl_30": lambda x, a, b, c: a * x ** (b / x),
            "dl_31": lambda x, a, b, c: a * x ** (b / (x + c)),

            "dl_60": lambda x, a, b: 1 * x ** (a / x ** b),

            "dl_71": lambda x, a, b, c: b * x ** (a / x),
            "dl_72": lambda x, a, b, c: b * (x - c) ** (a / x),
            "dl_73": lambda x, a, b, c: b * x ** (a / (x - c)),

            "dl102": lambda x, a, b, c, d: (x ** c + a * x) / (b + x),
            "dl103": lambda x, a, b, c, d: (x ** (c / (x - d)) + a * x) / (b + x)
        }

        if not_selected is True:
            no_slt_curves = {
                "dl100": lambda x, a, b, c: a * x ** c / (b + x),
                "dl101": lambda x, a, b, c, d: (a * x ** c + x) / (b + x),
                "pol_2": lambda x, a, b, c, d: a * ((x - c) ** 2) + b * (x - c) + d,
                "dl_10": lambda x, a, b: a * x * b ** x,
                "dl_21": lambda x, a, b, c: a * np.exp(b / x),
                "dl_50": lambda x, a, b: np.exp(a / x ** b),
                "dl_80": lambda x, a, b, c, d: x ** (a / b ** x),
            }
            curve_func_dict.update(no_slt_curves)

        if del_selected is True:
            del_slt_curves = {
                "exp_2": lambda x, a, b, c: a * np.exp((x - c) * b) - 0,
                "pol_5": lambda x, a, b, c: a * ((x - 0) ** 4) + b * ((x - c) ** 3),
                "dl_40": lambda x, a, b, c: c * np.exp(a / x ** 2 + b),
                "dl_81": lambda x, a, b, c, d: x ** (a / b ** x - c),
                "dl_90": lambda x, a, b: a * x / (b + x),
                "exp_4": lambda x, a, b: a ** ((x - 0) * b) - 0,  # VERY BAD approximation
                # "exp_3": lambda x, a, b, c: pam_n_cl_obj.exponential_func(x, a, b, 0, c),
                # "exp_5": lambda x, a, b, c: a ** ((x - 0) * b) - c,
                # "exp_6": lambda x, a, b, c: pam_n_cl_obj.exponent_func(x, a, b, c, 0),
                # "exp_7": lambda x, a, b, c: c * a ** ((x - 0) * b) - 0,     # THE SAME AS "exp_1"
                # "exp_8": lambda x, a, b, c, d: c * a ** ((x - d) * b) - 0,  # THE SAME AS "exp_1"
                # "exp_9": lambda x, a, b, c, d: c * a ** ((x - 0) * b) - d,
                # "exp_10": lambda x, a, b, c, d, e: c * a ** ((x - e) * b) - d,
                # "pol_7": lambda x, a, b, c, d: a * ((x - 0) ** 4) + b * ((x - c) ** 3) + d,
                # "ln__2": lambda x, a, b, c: b * np.log(x - a) - c,
                # "ln__4": lambda x, a, b: b * np.log(x - a) - 0,
                # "dl_11": lambda x, a, b, c: a*(x-c)*b**(x-c),
                # "dl_12": lambda x, a, b, c: a * x * b ** x + c,
                # "dl_22": lambda x, a, b, c: a * np.exp(b / (x+c)),
                # "dl_40": lambda x, a, b, c: c * np.exp(a * x**2 + b),     # VERY BAD approximation
                # "dl_61": lambda x, a, b, c: (x - c) ** (a / x ** b),
                # "dl_62": lambda x, a, b, c: 1 * x ** (a / (x - c) ** b),
                # "dl_91": lambda x, a, b, c: a * x / (b + x ** c),
            }
            curve_func_dict.update(del_slt_curves)

        if flt_key_list is None:
            return curve_func_dict
        elif isinstance(flt_key_list, str):
            if flt_key_list in curve_func_dict.keys():
                return curve_func_dict[flt_key_list]
        elif isinstance(flt_key_list, list):
            flt_in_keys = list(filter(lambda x: x in curve_func_dict.keys(), flt_key_list))
            return mf.filter_dict_by_keys(curve_func_dict, flt_in_keys)

    # NOT USED
    def create_curve_fit_data_for_n_cl_metrics(self, curve_f_keys: list | str | None = None):
        self.n_cl_m_fit, self.n_cl_m_delta, self.m_func_fit_difs = self.find_best_curve_fit_for_n_cl_metrics(
            curve_funcs_dict=self.curve_functions_for_fitting(flt_key_list=curve_f_keys, not_selected=False),
            n_cl_metrics=self.n_cl_metrics[self.metrics_for_fitting],  # self.n_cl_metrics.iloc[1:-1],
            min_x=self.min_n_cl,
            max_x=self.max_n_cl
        )

    def create_pol_fit_data_for_n_cl_metrics(self, n_cl_metrics):
        x_data = list(range(self.min_n_cl, self.max_n_cl + 1))
        max_pol_ord = len(x_data) // 2

        self.n_cl_m_fit = pd.DataFrame(index=n_cl_metrics.index, columns=n_cl_metrics.columns)
        self.n_cl_m_delta = pd.DataFrame(index=n_cl_metrics.index, columns=n_cl_metrics.columns)
        self.n_cl_m_r_delta = pd.DataFrame(index=n_cl_metrics.index, columns=n_cl_metrics.columns)
        self.m_func_fit_difs = pd.DataFrame(index=list(range(3, max_pol_ord)), columns=n_cl_metrics.columns)

        # TODO: FOR LESS THAN 5 CL CREATE AN OTHER PROCESS
        for metric in n_cl_metrics.columns:
            (
                self.n_cl_m_fit[metric],
                self.n_cl_m_delta[metric],
                self.n_cl_m_r_delta[metric],
                self.m_func_fit_difs[metric],
                self.n_cl_m_fig[metric]
            ) = self.find_best_pol_curve_fit_to_metric(
                x_data=x_data, metric=n_cl_metrics[metric], max_pol_ord=max_pol_ord
            )

    # NOT USED
    @classmethod
    def find_single_curve_fit_to_metric(cls, func_to_fit, metric, n_cl_metrics, min_x=None, max_x=None):
        min_x, max_x = cls.def_min_max_if_none(min_max_dif=len(n_cl_metrics.index) - 1, min_x=min_x, max_x=max_x)
        x_data = list(range(min_x, max_x + 1))

        curve_f_res, opt_func = cls.curve_fit_single_metric(
            curve_func=func_to_fit, metric=n_cl_metrics[metric], x_data=x_data
        )

        return curve_f_res, n_cl_metrics[metric] - curve_f_res

    @staticmethod
    def def_min_max_if_none(min_max_dif, min_x, max_x):
        if min_x is None:
            if max_x is None:
                min_x = 0
                max_x = min_max_dif
            else:
                min_x = 0
                max_x = min(max_x, min_max_dif)
        else:
            if max_x is None:
                min_x = min_x
                max_x = min_x + min_max_dif
            else:
                min_x = min_x
                max_x = min_x + min(max_x - min_x, min_max_dif)

        return min_x, max_x

    @classmethod
    def find_best_curve_fit_for_n_cl_metrics(cls, curve_funcs_dict, n_cl_metrics, min_x=None, max_x=None):
        min_x, max_x = cls.def_min_max_if_none(min_max_dif=len(n_cl_metrics.index) - 1, min_x=min_x, max_x=max_x)
        x_data = list(range(min_x, max_x+1))

        n_cl_m_fit = pd.DataFrame(index=n_cl_metrics.index, columns=n_cl_metrics.columns)
        n_cl_m_delta = pd.DataFrame(index=n_cl_metrics.index, columns=n_cl_metrics.columns)
        min_sqrt_fit_dist = pd.DataFrame(index=curve_funcs_dict.keys(), columns=n_cl_metrics.columns)
        """
        print("pam_line_2918")
        print(n_cl_metrics)
        """
        for metric in n_cl_metrics.columns:
            n_cl_m_fit[metric], n_cl_m_delta[metric], min_sqrt_fit_dist[metric] = cls.find_best_curve_fit_for_metric(
                metric_sr=n_cl_metrics[metric], curve_funcs_dict=curve_funcs_dict, x_data=x_data
            )

        return n_cl_m_fit, n_cl_m_delta, min_sqrt_fit_dist

    @staticmethod
    def curve_fit_single_metric(curve_func: callable, metric: pd.Series, x_data):
        """
        print("pam_line_2927")
        print(metric)
        print(x_data)
        """

        try:
            popt = curve_fit(curve_func, x_data, metric.values)[0]
        except TypeError:
            return None, None
        except RuntimeError:
            return None, None

        def opt_func(x):
            return curve_func(x, *popt)

        curve_f_res = pd.Series(x_data, index=metric.index).apply(opt_func)

        with pd.option_context('mode.use_inf_as_na', True):
            if curve_f_res.isna().any():
                return None, None

        return curve_f_res, opt_func

    @staticmethod
    def find_best_pol_curve_fit_to_metric(x_data, metric, max_pol_ord, relative_error=False, fig_x_num=None):
        x = np.array(x_data)
        y = metric.values
        fig_x_num = 38 if fig_x_num is None else fig_x_num
        x_b = np.linspace(x[0], x[-1], num=fig_x_num)  # 'num' is double the default k=19 points, can be set to anything

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=x, y=y, name="Input data"))

        p_y_list = []
        p_y_xb_list = []
        poly_dist = []
        poly_area_dif = []

        pol_ord_to_fit = list(range(3, max_pol_ord))
        for i in pol_ord_to_fit:

            odr_obj = odr.ODR(odr.Data(x, y), odr.polynomial(i))
            output = odr_obj.run()  # running ODR fitting
            poly = np.poly1d(output.beta[::-1])

            p_y_xb = poly(x_b)
            p_y_xb_list += [p_y_xb]

            p_y = poly(x)
            p_y_list += [p_y]

            sq_dif = sum(((y - p_y) / p_y)**2) if relative_error else sum((y - p_y)**2)
            poly_dist += [sq_dif]

            area_dif = gmf.trapezoid_area_fn_diff(y_1=y, y_2=p_y, dx=1)
            poly_area_dif += [area_dif]

            print(f"line_615\n area_diff: {area_dif}\n min_")
            print(area_dif)
            print(sq_dif)
            fig.add_trace(go.Scatter(
                x=x_b, y=p_y_xb,
                name=f"{i}ord polynomial ODR", mode="lines",
                line=dict(color=f"#{i%10}{i%10}{i%10}700")   # f"#{i%9}5{i%9}700")
            ))

        value_d, poss_d = mf.n_min(poly_dist, n=len(poly_dist) // 5 + 1)
        p_res_d_mean_xb = np.array(p_y_xb_list)[poss_d].mean(axis=0)

        value_a, poss_a = mf.n_min(poly_area_dif, n=len(poly_area_dif) // 5 + 1)
        p_res_a_mean_xb = np.array(p_y_xb_list)[poss_a].mean(axis=0)

        d_min = poly_dist.index(min(poly_dist))
        a_min = poly_area_dif.index(min(poly_area_dif))

        fig.add_trace(go.Scatter(
            x=x_b, y=p_y_xb_list[d_min], name="min dist", mode="lines", line=dict(color=f"#F20094")
        ))
        fig.add_trace(go.Scatter(
            x=x_b, y=p_y_xb_list[a_min], name="min area", mode="lines", line=dict(color=f"#821800")
        ))
        fig.add_trace(go.Scatter(
            x=x_b, y=p_res_d_mean_xb, name="mean of 4 best dist", mode="lines", line=dict(color=f"#0075F2")
        ))
        fig.add_trace(go.Scatter(
            x=x_b, y=p_res_a_mean_xb, name="mean of 4 best area", mode="lines", line=dict(color=f"#0B7D5F")
        ))

        print("pam_line_3534")
        print(p_y_xb_list)
        print(poly_dist)
        print(poss_d)
        print(value_d)
        print("Line 3455")
        print(pd.DataFrame([poly_area_dif, poly_dist], index=["Area", "Dist"], columns=pol_ord_to_fit).T)

        p_res_d_mean = np.array(p_y_list)[poss_d].mean(axis=0)
        p_res_d_delta_mean = y - p_res_d_mean
        p_res_d_sum_mean = y + p_res_d_mean
        p_res_d_r_delta_mean = 2 * p_res_d_delta_mean / p_res_d_sum_mean

        return (
            p_res_d_mean,
            p_res_d_delta_mean,
            p_res_d_r_delta_mean,
            poly_dist,
            fig
        )

    @classmethod
    def find_best_curve_fit_for_metric(cls, metric_sr, curve_funcs_dict, x_data):
        min_squared_fit_dist = {}
        min_squared_fit = np.inf
        best_curve_f_res = pd.Series()
        best_curve_delta = pd.Series()
        best_c_f_key = None
        for c_f_key in curve_funcs_dict:
            """
            print("pam line 2388", c_f_key)
            try:
                popt = curve_fit(curve_funcs_dict[c_f_key], x_data, metric_sr)[0]
            except TypeError:
                continue
            except RuntimeError:
                continue
            curve_f_res = pd.Series(x_data, index=metric_sr.index).apply(
                lambda x: curve_funcs_dict[c_f_key](x, *popt)
            )
            curve_delta = metric_sr - curve_f_res
            """
            curve_f_res, opt_func = cls.curve_fit_single_metric(curve_funcs_dict[c_f_key], metric_sr, x_data=x_data)
            if curve_f_res is None:
                continue

            curve_delta = metric_sr - curve_f_res
            min_sqrt_fit = np.sum(curve_delta**2)
            min_squared_fit_dist[c_f_key] = min_sqrt_fit

            if min_sqrt_fit < min_squared_fit:
                min_squared_fit = min_sqrt_fit
                best_c_f_key = c_f_key
                best_curve_f_res = curve_f_res
                best_curve_delta = curve_delta

        min_sqrt_fit_dist = pd.Series(min_squared_fit_dist).sort_values(ascending=True)

        if not min_sqrt_fit_dist.index[0] == best_c_f_key:
            raise Exception("Minimum values don't match")

        return best_curve_f_res, best_curve_delta, min_sqrt_fit_dist

    @classmethod
    def calc_inertia_angles(cls, total_inertia):
        if len(total_inertia.index) < 2:
            return pd.Series(None, index=total_inertia.index)

        inertia_angles = pd.Series()
        # key_list = mf.sort_string_list_based_on_number_inside(list(n_cl_s_obj_dict.keys()))
        # inertia_angles = inertia_angles.sort_index(axis=0, key=mf.return_int_number_of_string, ascending=True)
        inertia_angles.loc[total_inertia.index[0]] = None

        for i, val in enumerate(total_inertia.index[1:-1]):
            inert_v1 = total_inertia.iloc[i]
            inert_v2 = total_inertia.iloc[i + 1]
            inert_v3 = total_inertia.iloc[i + 2]
            """
            print("$$$$$$")
            print(i+1)
            print(inert_v1)
            print(inert_v2)
            print(inert_v3)
            print(inert_v2 - inert_v1)
            print(inert_v3 - inert_v2)
            """
            inertia_angles.loc[val] = cls.calc_angle_of_tangents(inert_v2 - inert_v1, inert_v3 - inert_v2)

        inertia_angles.loc[total_inertia.index[-1:][0]] = None

        return inertia_angles

    @staticmethod
    def calc_angle_of_tangents(tang1, tang2):
        # angle_difference = math.degrees(math.atan(tang1)) - math.degrees(math.atan(tang2)) # The same as below
        angle_difference = math.atan(tang1) - math.atan(tang2)
        # print(math.degrees(angle_difference))
        gama_angle = 180 + math.degrees(angle_difference)
        """
        print("###")
        print(math.degrees(math.atan(tang1)))
        print(math.degrees(math.atan(tang2)))
        print(math.degrees(angle_difference))
        print(gama_angle)
        """
        return gama_angle

    def min_max_scale_n_cl_delta(self):
        return pd.concat([
            self.n_cl_m_delta[["Distortion", "Inertia", "Method Inertia"]].apply(
                lambda x: - (x - x.max()) / (x.max() - x.min())
            ),
            self.n_cl_m_delta[["Silhouette", "Cal.Har.d_metric", "Cal.Har.sklearn"]].apply(
                lambda x: (x - x.min()) / (x.max() - x.min())
            )
        ], axis=1)

    def standardize_n_cl_delta(self):
        return pd.concat([
            self.n_cl_m_delta.loc[
                :, self.n_cl_m_delta.columns.isin(self.dec_val_metric_names)
            ].apply(
                lambda x: - (x - x.mean()) / x.std()
            ),
            self.n_cl_m_delta.loc[
                :, self.n_cl_m_delta.columns.isin(self.inc_val_metric_names)
            ].apply(
                lambda x: (x - x.mean()) / x.std()
            )
        ], axis=1)

    def standardize_metric_delta(self, sr):
        if sr.name in self.inc_val_metric_names:
            return gdp.standardize_sr(sr)
        elif sr.name in self.dec_val_metric_names:
            return -gdp.standardize_sr(sr)
        else:
            return pd.Series(None, index=sr.index, name=sr.name)

    def change_metric_sr_monotony(self, sr):
        if sr.name in self.inc_val_metric_names:
            return sr
        elif sr.name in self.dec_val_metric_names:
            return -sr
        else:
            return pd.Series(None, index=sr.index, name=sr.name)

    def selecting_num_of_clusters(self, relative_error=False):
        print("pam_line_807")
        print(self.n_cl_metrics)
        print(self.n_cl_m_delta)
        print(self.n_cl_m_r_delta)

        if relative_error:
            n_cl_m_delta_scaled = self.n_cl_m_r_delta.apply(self.change_metric_sr_monotony).dropna()
        else:
            # n_cl_m_delta_scaled = self.standardize_n_cl_delta()
            n_cl_m_delta_scaled = self.n_cl_m_delta.apply(self.standardize_metric_delta).dropna()

        self.n_cl_score = n_cl_m_delta_scaled.dot(self.metrics_weights).sort_values(ascending=False)

        self.selected_n_cl_str = self.n_cl_score.index[0]
        self.selected_n_cl = mf.return_int_number_of_string(self.selected_n_cl_str)

    def selected_clm_obj(self) -> ClMetrics:
        return self.res_n_cl_obj_dict[self.selected_n_cl_str]

    @property
    def cl_m_slt(self):
        return self.selected_clm_obj()


def print_pd_crosstab(labels, target_labels):
    if isinstance(target_labels, (list, dict, np.ndarray)):
        target_labels = pd.Series(target_labels)

    if isinstance(labels, (list, dict, np.ndarray)):
        labels = pd.Series(labels)

    target_labels = target_labels.rename("Target labels")
    labels = labels.rename("labels predicted")

    crosstab_res = pd.crosstab(index=labels, columns=target_labels)

    # print(f"\n\npam_line_810\n---\n{crosstab_res}\n---\n\n")

    return crosstab_res


def load_iris_dataset():
    data_obj = load_iris(as_frame=True)
    data = data_obj.data
    target_of_cl = data_obj.target

    return data, target_of_cl


def create_data_fit_for_clustering(random_state: int | None = None, n_features=5):
    def partial_make_blobs(n_samples, centers, cluster_std, center_box):
        return make_blobs(
            n_samples=n_samples,
            n_features=n_features,
            centers=centers,
            cluster_std=cluster_std,
            center_box=center_box,
            shuffle=True,
            random_state=random_state,
            return_centers=True
        )

    # TODO: Use number of samples

    feature_names = [f"Feature_{i}" for i in range(1, n_features + 1)]

    data_1, labels_1, center_1 = partial_make_blobs(
        n_samples=500,  # 500
        centers=5,  # 4
        cluster_std=0.4,  # 1
        center_box=(0, 2),  # (-10.0, 10.0)
    )

    data_2, labels_2, center_2 = partial_make_blobs(
        n_samples=300,  # 500
        centers=5,  # 4
        cluster_std=0.2,  # 1
        center_box=(0.5, 2.5),  # (-10.0, 10.0)
    )

    data_3, labels_3, center_3 = partial_make_blobs(
        n_samples=200,  # 500
        centers=5,  # 4
        cluster_std=0.1,  # 1
        center_box=(1, 2),  # (-10.0, 10.0)
    )

    data_4, labels_4, center_4 = partial_make_blobs(
        n_samples=[
            20, 30, 100, 50, 40, 150, 60,
            150, 50, 10, 100, 70, 30, 90, 80
        ],  # 500
        centers=None,  # 14
        cluster_std=0.15,  # 1
        center_box=(0, 1.5),  # (-10.0, 10.0)
    )
    data_list = [data_1, data_2, data_3, data_4]

    def lb_fn(lb_sr, bl: int):
        return np.array([f"b({bl + 1}) : c{j + 1}" for j in lb_sr])

    def cp_fn(centers, bl):
        return pd.DataFrame(
            centers,
            index=[f"b({bl + 1}) : c{j + 1}" for j in range(len(centers))],
            columns=feature_names
        )

    labels_list = [lb_fn(labels_1, 1), lb_fn(labels_2, 2), lb_fn(labels_3, 3), lb_fn(labels_4, 4)]
    centers_list = [cp_fn(center_1, 1), cp_fn(center_2, 2), cp_fn(center_3, 3), cp_fn(center_4, 4)]

    points = pd.DataFrame(np.vstack(data_list), columns=feature_names)
    labels = pd.Series(np.hstack(labels_list))
    cps = pd.concat(centers_list)

    print("pam_line_908")
    print(data_1)
    print(labels_1)
    print(center_1)
    print("___")
    print(points)
    print(labels)
    print(cps)

    return points, labels, cps


def cl_metrics_set_up_for_kms_obj(
        data, n_clusters, max_iter, dist_func_obj: DistFunction | None = None,
        **kwargs
):
    dist_func_obj = mf.def_var_value_if_none(dist_func_obj, def_func=lambda: DistFunction(
        dist_metric="norm^p",
        norm_order=2,
        cache_points=data
    ))

    return ClMetrics.from_sklearn_k_means_obj(
        kmeans_obj=KMeans(n_clusters=n_clusters, max_iter=max_iter, **kwargs),
        data=data,
        dist_func_obj=dist_func_obj,
    )


def cl_metrics_set_up_for_k_medoids(
        data, n_clusters, max_iter=40,
        dist_metric="euclidean", norm_ord=None, dist_func_obj: DistFunction | None = None,
        dists_norm=1,
        n_init=3, starting_points=None, im_st_points=None, dim_redux=None, st_p_method="convex_hull",
        n_cp_cand=5, g_median=True, p_mean_ord=None, iter_cp_comb=200
) -> ClMetrics:
    dist_func_obj = mf.def_var_value_if_none(dist_func_obj, def_func=lambda: DistFunction(
        dist_metric=dist_metric,
        norm_order=norm_ord,
        cache_points=data
    ))

    return ClMetrics.from_k_medoids_obj(Kmedoids(
        data=data,
        n_clusters=n_clusters,
        max_iter=max_iter,
        dists_norm=dists_norm,
        dist_func_obj=dist_func_obj,
        n_init=n_init,
        dim_redux=dim_redux,
        starting_points=starting_points,
        im_st_points=im_st_points,
        st_p_method=st_p_method,
        cand_options={"n_cp_cand": n_cp_cand, "g_median": g_median, "p_mean_ord": p_mean_ord},
        iter_cp_comb=iter_cp_comb
    ))


def cl_metrics_set_up_for_faster_pam(
        data, n_clusters, max_iter=100,
        dist_metric="euclidean", norm_ord=None, dist_func_obj: DistFunction | None = None, dists_norm=1,
        **kwargs
) -> Callable[[], ClMetrics]:
    dist_func_obj = mf.def_var_value_if_none(dist_func_obj, def_func=lambda: DistFunction(
        dist_metric=dist_metric,
        norm_order=norm_ord,
        cache_points=data
    ))

    return ClMetrics.from_faster_pam_obj(
        f_pam_obj=fasterpam(
            diss=dist_func_obj.distance_func_cache_all(),
            medoids=n_clusters,
            max_iter=max_iter,
            **kwargs
        ),
        data=data,
        dist_func_obj=dist_func_obj,
        dists_norm=dists_norm    # CHECK default AGAIN: 1 OR 2
    )


def main():
    target_labels = None
    # data = load_weights_excel_data()
    # data, target_labels = load_iris_dataset()
    # data = geeks_for_geeks_example_elbow_data()
    data, target_labels, cps = create_data_fit_for_clustering(random_state=np.random.randint(low=0, high=100))
    # data = data.set_axis([f"p_{i+1}" for i in range(len(data.index))])
    # data = data.set_axis([f"DM_{i + 1}" for i in range(len(data.index))])
    # data = data.set_axis([f"crit_{i + 1}" for i in range(len(data.columns))], axis=1)

    """
    kmedoids_res = Kmedoids(data, n_clusters=5, max_iter=10)
    kmedoids_res.print_all_tables_per_iter()
    print(kmedoids_res)
    """

    # ClSelect.calc_angle_of_tangents(5-10, 3-5)
    # ClSelect.calc_angle_of_tangents(-math.tan(math.pi/3), -math.tan(math.pi/6))

    """
    kms_n_cl_obj = ClSelect(
        return_cl_metrics_kms_obj, data,
        min_n_cl=2, max_n_cl=10, n_iter=10, target_labels=target_labels)
    print(kms_n_cl_obj)
    """

    # kmedoids = test_pam_default_dist_func_settings(data, test_only_first=True)

    n_cl_obj = ClSelect(
        cl_metrics_obj_func=partial(cl_metrics_set_up_for_k_medoids, n_cp_cand=None),
        data=data, min_n_cl=2, max_n_cl=15, n_iter=100
    )
    print(n_cl_obj.n_cl_metrics["Silhouette"])
    cl_m_slt = n_cl_obj.selected_clm_obj()
    gvp.set_up_3d_graph_data(data, cl_m_slt.labels, "PCA").show()
    gvp.set_up_3d_graph_data(data, target_labels, "PCA").show()
    gvp.set_up_3d_graph_data(data, cl_m_slt.labels, "LDA").show()
    gvp.set_up_3d_graph_data(data, target_labels, "LDA").show()

    # cl_m_kms_obj = return_cl_metrics_kms_obj(data, 8, 10)
    # print(cl_m_kms_obj)


if __name__ == "__main__":
    main()
