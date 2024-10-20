import pandas as pd
import numpy as np
from numba import njit, prange
from dataclasses import dataclass
from typing import Optional, Callable

from sklearn.metrics import silhouette_samples, calinski_harabasz_score
from sklearn.preprocessing import LabelEncoder

from scipy.spatial.distance import pdist, cdist, squareform
from scipy.spatial.distance import braycurtis, correlation, cosine, jensenshannon
from scipy.spatial.distance import canberra, chebyshev
from scipy.spatial.distance import sqeuclidean, euclidean, cityblock
from scipy.spatial import distance_matrix

from kmedoids import KMedoidsResult

from utils import general_functions as mf
import utils.math_functions as gmf
import utils.pandas_functions as gpd
from utils.general_functions import print_execution_time


# import utils.data_processing as gdp
# import utils.visualization_functions as gvp
# import Read_Excel as rxl


class DistFunction:

    dist_func_1d_dict = {
        "euclidean": euclidean,  # p = 2
        "sqeuclidean": sqeuclidean,  # p = 2, norm^2
        "cityblock": cityblock,  # p = 1
        "chebyshev": chebyshev,  # inf-norm (max)
        "minkowski (p-norm)": lambda x, y, norm_ord: np.linalg.norm(x - y, ord=norm_ord),
        "p-norm^p": lambda x, y, norm_ord: np.sum(abs(x - y) ** norm_ord),
        "braycurtis": braycurtis,
        "canberra": canberra,
        "correlation": correlation,
        "cosine": cosine,
        "jensenshannon": jensenshannon
    }

    def __init__(self, dist_metric="euclidean", norm_order=None,
                 dist_func: Callable = None, vector_dist_func: Callable = None,
                 cache_points: pd.DataFrame = None, points_weight_matrix: pd.DataFrame = None):

        self.norm_order = norm_order
        self.dist_func_0d = vector_dist_func
        self.dist_func_1d = None
        self.dist_func_2d = None
        self.pdist_func = None

        self.dist_metric = dist_metric
        self.distance_metric_set_up(dist_metric, norm_order, vector_dist_func)

        self.cache_points = cache_points
        self.pw_matrix = points_weight_matrix
        self.dists_matrix_cache = None

        self.cache_dists = cache_points is not None
        if self.cache_dists:
            self.init_cache(self.cache_points)

        self.dist_func = mf.def_var_value_if_none(value_passed=dist_func, default=self.default_dist_func())

        self.cache_complete = False

        self.dist_func_cache = self.distance_func_cache_all

        if self.cache_dists:
            self.distance_func_cache_all()

    def __repr__(self):
        print(f"Metric selected: {self.dist_metric}")
        print(f"Norm order: {self.norm_order}")
        print(f"Distance funtion for vectors (1d) passed (bool): {self.dist_func is None}")
        print(f"Cache distances (bool): {self.cache_dists is True}")
        return "DistFunction obj"

    def df_to_cache(self, df):
        self.cache_points = df
        self.init_cache(df)

    def delete_cache(self):
        self.init_cache(self.dists_matrix_cache)
        self.cache_complete = False

    def init_cache(self, points):
        self.dists_matrix_cache = pd.DataFrame(np.nan, index=points.index, columns=points.index)

    def distance_metric_set_up(self, metric_passed: str, norm_order: int, vector_dist_func: Callable):
        if metric_passed is None and norm_order is None and vector_dist_func is None:
            raise Exception("At least 1 of the 3 must be passed: metric, norm_order, dist_func_0d")

        if vector_dist_func is not None:
            self.dist_metric = f"custom 1d-func metric: {metric_passed}" if metric_passed else "custom 1d-func metric"
            self.dist_func_1d_dict[self.dist_metric] = self.dist_func_0d
            self.dist_func_1d = lambda x, y: self.np_array_matrix_vector_dist(x, y, dist_func_1d=self.dist_func_0d)
            self.dist_func_2d = lambda x, y: self.np_array_matrix_dist_func(x, y, dist_func_1d=self.dist_func_0d)
            self.pdist_func = self.p_dist_func_from(dist_func_0d=self.dist_func_0d)
            return

        if metric_passed == "euclidean":
            self.dist_metric = "euclidean"
            self.dist_func_0d = lambda x, y: euclidean(x, y)
            self.dist_func_1d = lambda x, y: np.sqrt(np.sum((x - y) ** 2, axis=1))
            self.dist_func_2d = lambda x, y: cdist(x, y, self.dist_metric)
            self.pdist_func = self.p_dist_func_scipy(dist_metric=self.dist_metric)
            return

        if norm_order is not None or metric_passed in [
            "norm", "pnorm", "p_norm", "p-norm", "minkowski",
            "minkowski (norm)", "minkowski (p_norm)", "minkowski (p-norm)"
        ]:
            self.dist_metric = "minkowski (p-norm)"
            self.norm_order = mf.def_var_value_if_none(value_passed=norm_order, default=2)
            self.dist_func_0d = lambda x, y: np.linalg.norm(x - y, ord=norm_order)
            self.dist_func_1d = lambda x, y: np.linalg.norm(x - y, ord=norm_order, axis=1)
            self.dist_func_2d = lambda x, y: distance_matrix(x, y, p=norm_order)
            self.pdist_func = self.p_dist_func_scipy(dist_metric="minkowski", p=self.norm_order)
            return

        if metric_passed in ["sqeuclidean", "euclidean^2", "norm^2", "p_norm^2", "p-norm^2"]:
            self.dist_metric = "sqeuclidean"
            self.norm_order = 2
            self.dist_func_0d = sqeuclidean
            self.dist_func_1d = lambda x, y: np.sum(np.abs(x - y) ** 2, axis=1)
            self.dist_func_2d = lambda x, y: cdist(x, y, self.dist_metric)
            self.pdist_func = self.p_dist_func_scipy(dist_metric=self.dist_metric)
            return

        if metric_passed in ["p-norm^p", "p_norm^p", "p-norm^p", "norm^p"]:
            self.dist_metric = "p-norm^p"
            self.norm_order = mf.def_var_value_if_none(value_passed=norm_order, default=2)
            self.dist_func_0d = lambda x, y: np.sum(np.abs(x - y) ** self.norm_order)
            self.dist_func_1d = lambda x, y: np.sum(np.abs(x - y) ** self.norm_order, axis=1)
            self.dist_func_2d = gmf.gen_cust_dist_func(
                kernel_inner=lambda a, b: (a - b) ** self.norm_order,
                kernel_outer=lambda acc: acc,
            )
            self.pdist_func = self.p_dist_func_jit(
                kernel_inner=lambda a, b: (a - b) ** self.norm_order,
                kernel_outer=lambda acc: acc
            )
            return

        if metric_passed in self.dist_func_1d_dict.keys():
            self.dist_func_0d = self.dist_func_1d_dict[self.dist_metric]
            self.dist_func_1d = lambda x, y: self.np_array_matrix_vector_dist(x, y, dist_func_1d=self.dist_func_0d)
            self.dist_func_2d = lambda x, y: cdist(x, y, self.dist_metric)
            self.pdist_func = self.p_dist_func_scipy(dist_metric=self.dist_metric)
            return

        if metric_passed in [
            'braycurtis', 'canberra', 'chebychev', 'chebyshev', 'cheby', 'cheb', 'ch', 'cityblock', 'cblock',
            'cb', 'c', 'correlation', 'co', 'cosine', 'cos', 'dice', 'euclidean', 'euclid', 'eu', 'e',
            'hamming', 'hamm', 'ha', 'h', 'minkowski', 'mi', 'm', 'pnorm', 'jaccard', 'jacc', 'ja', 'j',
            'jensenshannon', 'js', 'kulsinski', 'kulczynski1', 'mahalanobis', 'mahal', 'mah', 'rogerstanimoto',
            'russellrao', 'seuclidean', 'se', 's', 'sokalmichener', 'sokalsneath', 'sqeuclidean', 'sqe',
            'sqeuclid', 'yule'
        ]:
            self.dist_metric = metric_passed
            self.dist_func_0d = lambda x, y: cdist([x], [y], metric_passed)[0][0]
            self.dist_func_1d = lambda x, y: cdist([y], x, metric_passed)[0]
            self.dist_func_2d = lambda x, y: cdist(x, y, metric_passed)
            self.pdist_func = self.p_dist_func_scipy(dist_metric=self.dist_metric)
            return

        raise Exception(f"Metric passed ({metric_passed}) has not been supported")

    def default_dist_func(self):
        # A property-like function that return a func
        # It also calculates weighted distances if a weight matrix is passed

        func = self.distance_function

        if self.pw_matrix is None:
            return func

        def wrapper(x: pd.DataFrame | pd.Series, y: pd.DataFrame | pd.Series):
            res = func(x, y)
            if isinstance(res, pd.DataFrame):
                return res * self.pw_matrix.loc[res.index, res.columns]
            elif isinstance(res, pd.Series):
                return res * self.pw_matrix.loc[res.index, res.name]
            else:
                return res * self.pw_matrix.loc[res.name, res.name]

        return wrapper

    @staticmethod
    def convert_to_pandas(convert_to_pd=True):
        def real_decorator(func):
            # if function takes any arguments can be added like this...(*args, **kwargs)
            def wrapper(*args, **kwargs):
                if len(args) == 3:
                    matrix_a = args[1]
                    matrix_b = args[2]
                elif len(args) == 1:
                    matrix_a = kwargs["matrix_a"]
                    matrix_b = kwargs["matrix_b"]
                elif len(args) == 2:
                    matrix_a = args[1]
                    matrix_b = kwargs["matrix_b"]
                else:
                    raise Exception(f"To many arguments\n: {args}")

                a_is_df = isinstance(matrix_a, pd.DataFrame)
                b_is_df = isinstance(matrix_b, pd.DataFrame)
                a_is_sr = isinstance(matrix_a, pd.Series)
                b_is_sr = isinstance(matrix_b, pd.Series)
                a_is_pd = a_is_df or a_is_sr
                b_is_pd = b_is_df or b_is_sr

                kwargs["matrix_a"] = matrix_a.values if a_is_pd else matrix_a
                kwargs["matrix_b"] = matrix_b.values if b_is_pd else matrix_b

                res = func(args[0], **kwargs)
                if a_is_pd and b_is_pd and convert_to_pd:
                    if a_is_df:
                        if b_is_df:
                            return pd.DataFrame(data=res, index=matrix_a.index, columns=matrix_b.index)
                        else:
                            return pd.Series(data=res, index=matrix_a.index, name=matrix_b.name)
                    elif b_is_df:
                        return pd.Series(data=res, name=matrix_a.name, index=matrix_b.index)

                return res

            return wrapper
        return real_decorator

    def np_dist_func(self, matrix_a: np.ndarray, matrix_b: np.ndarray):
        if len(matrix_a.shape) > 1:
            if len(matrix_b.shape) > 1:
                return self.dist_func_2d(matrix_a, matrix_b)
            else:
                return self.dist_func_1d(matrix_a, matrix_b)
        else:
            if len(matrix_b.shape) > 1:
                return self.dist_func_1d(matrix_b, matrix_a)
            else:
                return self.dist_func_0d(matrix_a, matrix_b)

    def distance_function_new(
            self,
            matrix_a: np.ndarray | pd.DataFrame | pd.Series,
            matrix_b: np.ndarray | pd.DataFrame | pd.Series
    ):
        a_is_np = isinstance(matrix_a, np.ndarray)
        b_is_np = isinstance(matrix_b, np.ndarray)

        if a_is_np:
            if b_is_np:
                return self.np_dist_func(matrix_a, matrix_b)
            elif isinstance(matrix_b, pd.DataFrame):
                if len(matrix_a.shape) > 1:
                    return self.dist_func_2d(matrix_a, matrix_b.values)
                else:
                    return self.dist_func_1d(matrix_a, matrix_b.values)
            elif isinstance(matrix_b, pd.Series):
                if len(matrix_a.shape) > 1:
                    return self.dist_func_1d(matrix_b.values, matrix_a)
                else:
                    return self.dist_func_0d(matrix_a, matrix_b.values)
            else:
                raise Exception(f"False matrix_b input type: {type(matrix_b)}")
        elif isinstance(matrix_a, pd.DataFrame):
            if b_is_np:
                if len(matrix_b.shape) > 1:
                    return self.dist_func_2d(matrix_a.values, matrix_b)
                else:
                    return self.dist_func_1d(matrix_a.values, matrix_b)
            elif isinstance(matrix_b, pd.DataFrame):
                return pd.DataFrame(
                    self.dist_func_2d(matrix_a.values, matrix_b.values),
                    index=matrix_a.index,
                    columns=matrix_b.columns
                )
            elif isinstance(matrix_b, pd.Series):
                return pd.Series(
                    self.dist_func_1d(matrix_a.values, matrix_b.values),
                    index=matrix_a.index,
                    name=matrix_b.name
                )
            else:
                raise Exception(f"False matrix_b input type: {type(matrix_b)}")
        elif isinstance(matrix_a, pd.Series):
            if b_is_np:
                if len(matrix_b.shape) > 1:
                    return self.dist_func_1d(matrix_b, matrix_a.values)
                else:
                    return self.dist_func_0d(matrix_a.values, matrix_b)
            if isinstance(matrix_b, pd.DataFrame):
                return pd.Series(
                    self.dist_func_1d(matrix_b.values, matrix_a.values),
                    index=matrix_b.index,
                    name=matrix_a.name
                )
            if isinstance(matrix_b, pd.Series):
                return self.dist_func_0d(matrix_a.values, matrix_b.values)
            else:
                raise Exception(f"False matrix_b input type: {type(matrix_b)}")
        else:
            raise Exception(f"False matrix_a input type: {type(matrix_a)}")

    @staticmethod
    def matrix_dims(matrix):
        if len(matrix.shape) > 1:
            shape = matrix.shape[:2]
            is_matrix = True
        else:
            shape = (1, matrix.shape[0])
            is_matrix = False
        return is_matrix, shape

    @convert_to_pandas(convert_to_pd=True)
    def distance_function(self, matrix_a: np.ndarray, matrix_b: np.ndarray):
        """
        The distance function for matrices and vectors
        :param matrix_a: np.ndarray
        :param matrix_b: np.ndarray

        :return: If input is pandas,
        the decorator 'convert_to_pandas' converts the result to pandas
        """
        a_is_matrix, shape_a = self.matrix_dims(matrix_a)
        b_is_matrix, shape_b = self.matrix_dims(matrix_b)

        if shape_a[1] != shape_b[1]:
            raise Exception(f"Columns of A and B do not match\nshapes:\nA({matrix_a.shape})\nB({matrix_b.shape})")

        if a_is_matrix:
            return self.dist_func_2d(matrix_a, matrix_b) if b_is_matrix else self.dist_func_1d(matrix_a, matrix_b)
        else:
            return self.dist_func_1d(matrix_b, matrix_a) if b_is_matrix else self.dist_func_0d(matrix_a, matrix_b)

    @staticmethod
    def np_array_matrix_dist_func(matrix_a: np.ndarray, matrix_b: np.ndarray, dist_func_1d):
        idx_len = len(matrix_a)
        col_len = len(matrix_b)
        dists_array = np.empty(shape=[idx_len, col_len], dtype=np.float32)

        for i in range(idx_len):
            for j in range(col_len):
                dists_array[i, j] = dist_func_1d(
                    matrix_a[i], matrix_b[j]
                )
        return dists_array

    @staticmethod
    def np_array_matrix_vector_dist(matrix_a, vector_b, dist_func_1d):
        total_dist_pers = len(matrix_a)
        dists_array = np.empty(shape=total_dist_pers, dtype=np.float32)

        for i in range(total_dist_pers):
            dists_array[i] = dist_func_1d(
                matrix_a[i], vector_b
            )
        return dists_array

    # NOT USED
    def calc_points_and_single_cp_dist(
            self, points_df: pd.DataFrame | np.ndarray, center: pd.Series | np.ndarray
    ):
        if isinstance(points_df, pd.DataFrame):
            return points_df.apply(
                lambda x: self.dist_func_0d(other_point=x, center_point=center),
                axis=1
            )

        return np.array([
            self.dist_func_0d(row, center) for row in points_df
        ])

    @staticmethod
    def p_dist_func_scipy(dist_metric, **kwargs):
        def p_d_func(matrix):
            return pd.DataFrame(
                squareform(pdist(matrix, dist_metric, **kwargs)),
                index=matrix.index,
                columns=matrix.index
            )
        return p_d_func

    @staticmethod
    def p_dist_func_from(dist_func_0d: Callable):
        dist_func_0d = njit(dist_func_0d, fastmath=True, inline='always')

        @njit(fastmath=True, parallel=True)
        def p_d_func(matrix: np.ndarray):
            n = matrix.shape[0]
            total_dist_pers = (n * (n - 1)) // 2
            dists_array = np.empty(shape=total_dist_pers, dtype=np.float32)
            k = 0
            for i in prange(n - 1):
                for j in range(i + 1, n):
                    dists_array[k] = dist_func_0d(
                        matrix[i], matrix[j]
                    )
                    k += 1
                # print(f"{k} / {total_dist_pers}")
            return dists_array

        def p_d_f(matrix):
            return pd.DataFrame(
                squareform(p_d_func(matrix.values)), index=matrix.index, columns=matrix.index
            )

        return p_d_f

    @staticmethod
    def p_dist_func_jit(kernel_inner: Callable, kernel_outer: Callable):
        d_func = gmf.gen_cust_p_dist_func(
            kernel_inner=kernel_inner,
            kernel_outer=kernel_outer
        )

        def func(matrix):
            return pd.DataFrame(
                squareform(d_func(matrix)),
                index=matrix.index,
                columns=matrix.columns
            )
        return func

    def distance_func_cache_all(self, matrix_a=None, matrix_b=None, cache=True):
        if matrix_a is None and matrix_b is None and self.cache_complete:
            print("pam_line_611")
            print("Precomputed distance matrix requested")
            return self.dists_matrix_cache

        if matrix_a is None:
            matrix_a = self.cache_points

        if matrix_b is None:
            matrix_b = matrix_a

        if not cache:
            return self.dist_func(matrix_a=matrix_a, matrix_b=matrix_b)

        if isinstance(matrix_a, pd.DataFrame):
            index_idx = matrix_a.index
        elif isinstance(matrix_a, pd.Series):
            index_idx = matrix_a.name
        else:
            raise Exception("False data type entry")

        if isinstance(matrix_b, pd.DataFrame):
            col_idx = matrix_b.index
        elif isinstance(matrix_b, pd.Series):
            col_idx = matrix_b.name
        else:
            raise Exception("False data type entry")

        if not self.cache_complete:
            print("pam_line_638")
            print("Cache is not complete!")
            print(f"Distance metric {self.dist_metric}")
            print(f"Number of np.nun values: {self.num_of_nan_in_cache()}")

            self.dists_matrix_cache = self.pdist_func(self.cache_points)

            self.check_if_all_dists_are_cached()

        return self.dists_matrix_cache.loc[index_idx, col_idx].copy()

    def check_if_all_dists_are_cached(self):
        self.cache_complete = all(self.dists_matrix_cache.notna().apply(all))
        # self.dist_func_cache = lambda a, b, bl=self.cache_dists: self.distance_func_cache_all(a, b, bl)
        print(f"pam line 544\n cache complete: {self.cache_complete}")

    def num_of_nan_in_cache(self):
        return self.dists_matrix_cache.apply(
            lambda x: x.isna().sum()
        ).sum()

    def check_cache_compatibility(self, data_points):
        row_check = len(self.dists_matrix_cache.loc[data_points.index].index) == len(data_points.index)
        col_check = len(self.cache_points[data_points.columns].columns) == len(data_points.columns)
        return row_check and col_check


@dataclass(slots=True, frozen=True)
class ClMGroupData:
    labels: pd.Series
    dists_matrix: pd.DataFrame
    cl_cp_costs_df: pd.Series
    global_cp_cost: pd.Series
    global_medoid: pd.Series
    cl_method_name: str
    nun_of_cl: int
    exec_name: Optional[str] = None

    def rtn_subgroup_obj(self, idx: pd.Index | pd.MultiIndex, idx_lvl_1: list | None = None):
        if isinstance(idx, pd.MultiIndex):
            df_idx = self.cl_cp_costs_df.index
            """
            print("hcl_line_28")
            print(self.cl_cp_costs_df.index)
            print(idx)
            print(idx.nlevels)
            print(df_idx.nlevels)
            print(df_idx.get_level_values(0))
            print(df_idx.get_level_values(df_idx.nlevels-2))
            print(idx.get_level_values(0))
            """

            lvl1 = list(df_idx.get_level_values(df_idx.nlevels-2).isin(idx.get_level_values(0)))
            lvl2 = list(df_idx.get_level_values(df_idx.nlevels-1).isin(idx.get_level_values(1)))
            bool_lvl = [lvl1[i] and lvl2[i] for i in range(len(lvl1))]
        else:
            df_idx = self.cl_cp_costs_df.index
            bool_lvl = list(df_idx.get_level_values(df_idx.nlevels-2).isin(idx))

        return ClMGroupData(
            labels=mf.set_up_df_to_index(df=self.labels, idx=idx, idx_lvl_1=idx_lvl_1),
            dists_matrix=mf.set_up_df_to_index(df=self.dists_matrix, idx=idx, idx_lvl_1=idx_lvl_1),
            cl_cp_costs_df=self.cl_cp_costs_df.loc[bool_lvl],
            global_cp_cost=mf.set_up_df_to_index(df=self.global_cp_cost, idx=idx, idx_lvl_1=idx_lvl_1),
            global_medoid=self.global_medoid,
            cl_method_name=self.cl_method_name,
            nun_of_cl=self.nun_of_cl,
            exec_name=f"{self.exec_name} (flt)",
        )


class ClMetrics:
    def __init__(
            self, data: pd.DataFrame, cl_method, labels: pd.Series, center_points: pd.DataFrame | None,
            dist_func, dist_metric: str, dists_matrix: pd.DataFrame, dists_p_norm: int,
            inertia=None, clusters_df=None, cps_dist_matrix=None,
            cl_name: str | None = "Clustering Metrics object", medoid_le_dict: dict | None = None):

        self.cl_name = cl_name
        self.data = data
        self.Cl_Method = cl_method
        self.labels = labels
        # self.labels_n = labels.apply(lambda x: list(labels.unique()).index(x))
        self.cps_df = self.data.groupby(self.labels).mean() if center_points is None else center_points
        self.dist_func = dist_func
        self.dist_metric = dist_metric
        self.dists_matrix = dists_matrix.loc[self.data.index, self.data.index]

        self.dists_p_norm = dists_p_norm
        self.inertia = inertia
        self.medoid_le_dict = medoid_le_dict

        if center_points is None:
            self.cps_df = self.data.groupby(self.labels).mean()

        self.global_cp_cost = self.dists_matrix.apply(lambda x: np.linalg.norm(x, ord=self.dists_p_norm))

        self.n_cl = len(self.cps_df.index)
        self.no_im_cp_bool = all(self.cps_df.index.isin(self.labels.index))
        print("pam_line_2038")
        print(self.no_im_cp_bool)
        print(self.medoid_le_dict)
        """
        print("pam_line_1640")
        self.print_data_passed()
        """

        if cps_dist_matrix is not None:
            self.cps_dist_matrix = cps_dist_matrix
        elif self.no_im_cp_bool:
            self.cps_dist_matrix = self.dists_matrix.loc[self.data.index, self.cps_df.index]
        elif medoid_le_dict is not None:
            medoid_idx = pd.Index([t for k, t in self.medoid_le_dict.items()])
            self.cps_dist_matrix = self.dists_matrix.loc[self.data.index, medoid_idx]
            self.cps_dist_matrix = self.cps_dist_matrix.set_axis(self.medoid_le_dict.keys(), axis=1)
        else:
            self.cps_dist_matrix = self.dist_func(self.data, self.cps_df, False)

        self.clusters_df = mf.def_var_value_if_none(clusters_df, def_func=self.create_clusters_df)
        """
        print("pam_line_1825")
        print(self.clusters_df)
        print(self.create_clusters_df(concat_method=True))
        print(self.create_clusters_df(concat_method=False))
        """
        self.cl_cp_costs_df = self.create_clusters_cp_costs_df()
        """
        print("pam_line_2054")
        print(self.cl_cp_costs_df.sort_index())
        self.cl_cp_costs_df = self.create_clusters_cp_costs_df_old()
        print(self.cl_cp_costs_df.sort_index())
        """

        # TODO: For k-medoids ClDistMetrics is already computed, need to avoid merged multi-index
        self.ClDist = ClDistMetrics(
            data=self.data,
            labels=self.labels,
            center_points=self.cps_df,
            dist_func=self.dist_func,
            dist_metric=self.dist_metric,
            norm_ord=self.dists_p_norm,
            clusters_df=self.clusters_df,
            cps_dist_matrix=self.cps_dist_matrix,
            medoid_le_dict=self.medoid_le_dict
        )
        # self.cl_counts = self.ClDist.cl_counts.loc[list(self.cps_df.index)].set_axis(self.cps_df.index)
        self.cl_counts = self.ClDist.cl_counts.reindex(self.cps_df.index)

        self.sil_score = None
        self.sil_samples = None
        self.sil_clusters = pd.Series()
        self.sil_samples_spl = self.calc_silhouette_samples(simplified=True)
        self.set_up_silhouette_score()

        self.separation_cl: pd.Series | None = None
        self.global_medoid: pd.Series | None = None
        self.cal_har_index = None
        self.ch_index_skl_mod = None
        self.calinski_harabasz_index()
        self.sklearn_calinski_harabasz_modified()

        self.mean_cl_dists: pd.DataFrame | None = None
        self.calc_mean_cl_dists_from_all_points()

        self.cluster_metrics_df = self.create_cluster_metrics_df()
        self.samples_metrics_df = self.create_sample_metrics_df()

        self.metrics_sr = pd.Series({
            "Distortion": self.ClDist.distortion,  # 0.15
            "Inertia": self.ClDist.inertia,  # 0.075
            "Method Inertia": self.inertia,  # 0.075
            "Silhouette.sklearn": self.sil_score,  # 0.4
            "Simplified Silhouette": self.sil_samples_spl.mean(),
            "Silhouette": self.calc_silhouette_samples(simplified=False).mean(),
            "Cal.Har.d_metric": self.cal_har_index,  # 0.15
            "Cal.Har.sklearn": self.ch_index_skl_mod,  # 0.15
        }, name=f"n_cl({self.n_cl})")

    def print_data_passed(self):
        print("\n---\n\tClMetrics obj passed data")
        print(f"\nData:\n", self.data)
        print(f"\nLabels\n", self.labels)
        print(f"\nCenter points\n", self.cps_df)
        print(f"\nDistance metric\n", self.dist_metric)
        print("---")

    def __repr__(self):
        print(self.cluster_metrics_df)
        print(self.cluster_metrics_df.mean())
        return "ClMetrics obj"

    @classmethod
    def from_k_medoids_obj(cls, kmedoids_obj):
        return cls(
            data=kmedoids_obj.data,
            cl_method=kmedoids_obj,
            labels=kmedoids_obj.labels,
            center_points=kmedoids_obj.medoids,  # kmedoids_obj.res_cps_df,
            clusters_df=kmedoids_obj.clusters_df,
            dist_func=kmedoids_obj.dist_func_cache,
            dist_metric=kmedoids_obj.DistFunc.dist_metric,
            dists_matrix=kmedoids_obj.DistFunc.distance_func_cache_all(),
            dists_p_norm=kmedoids_obj.dists_norm_ord,
            inertia=kmedoids_obj.inertia,
            cl_name="Custom k-medoids"
        )

    @classmethod
    def from_sklearn_k_means_obj(cls, kmeans_obj, data, dist_func_obj: DistFunction | None = None):
        if dist_func_obj is None:
            dist_func_obj = DistFunction(
                dist_metric="sqeuclidean",
                cache_points=pd.DataFrame(data)
            )

        kms_fit = kmeans_obj.fit(data)
        im_cp_names = [f"im_cp({i + 1})" for i in range(len(kms_fit.cluster_centers_))]

        return cls(
            data=data,
            cl_method=kmeans_obj,
            labels=pd.Series(kms_fit.labels_, index=data.index).apply(lambda x: f"im_cp({x + 1})"),
            center_points=pd.DataFrame(data=kms_fit.cluster_centers_, index=im_cp_names, columns=data.columns),
            dist_metric=dist_func_obj.dist_metric,
            dist_func=dist_func_obj.dist_func_cache,
            dists_matrix=dist_func_obj.distance_func_cache_all(),
            dists_p_norm=1,
            inertia=kms_fit.inertia_,
            clusters_df=None,
            cps_dist_matrix=pd.DataFrame(kms_fit.transform(data), index=data.index, columns=im_cp_names),
            cl_name="K-means"
        )

    @classmethod
    def from_faster_pam_obj(cls, f_pam_obj: KMedoidsResult, data, dist_func_obj: DistFunction, dists_norm=2):
        def tuple_name_f(tpl):
            if not isinstance(tpl, tuple):
                return f"Medoid({tpl})"
            return f"Medoid({mf.tuple_to_text(tpl, sep='|')})"

        le = LabelEncoder()
        le.classes_ = np.array([tuple_name_f(t) for t in data.iloc[f_pam_obj.medoids].index])
        labels_inv = pd.Series(le.inverse_transform(f_pam_obj.labels), index=data.index)
        medoids = pd.DataFrame(data.iloc[f_pam_obj.medoids].values, index=list(le.classes_), columns=data.columns)
        labels_c = pd.Series(f_pam_obj.labels, index=data.index).apply(
            lambda x: tuple_name_f(data.iloc[f_pam_obj.medoids[x]].name)
        )
        print("pam_line_2165")
        print(list(labels_inv))
        print(list(labels_c))
        print(list(labels_inv) == list(labels_c))
        print(data.iloc[f_pam_obj.medoids])
        print(medoids)

        return cls(
            data=data,
            cl_method=f_pam_obj,
            labels=labels_inv,
            center_points=medoids,
            dist_metric=dist_func_obj.dist_metric,
            dist_func=dist_func_obj.dist_func_cache,
            dists_matrix=dist_func_obj.distance_func_cache_all(),
            dists_p_norm=dists_norm,
            inertia=float(f_pam_obj.loss),
            cl_name="Faster_PAM",
            medoid_le_dict=dict(zip(list(le.classes_), list(data.iloc[f_pam_obj.medoids].index)))
        )

    @classmethod
    def from_target_labels(
            cls,
            data: pd.DataFrame,
            target_labels: pd.Series,
            center_points: pd.DataFrame | None = None,
            dist_metric="sqeuclidean",
            dists_p_norm: int | float = 1,
            **kwargs
    ):
        dist_func_obj = DistFunction(dist_metric=dist_metric, cache_points=data, **kwargs)
        return cls(
            data=data,
            cl_method=None,
            labels=target_labels,
            center_points=center_points,
            dist_func=dist_func_obj.dist_func_cache,
            dist_metric=dist_func_obj.dist_metric,
            dists_matrix=dist_func_obj.dist_func_cache(),
            dists_p_norm=dists_p_norm,
        )

    def rtn_clm_g_data_obj(self, name=None):
        return ClMGroupData(
            labels=self.labels,
            dists_matrix=self.dists_matrix,
            cl_cp_costs_df=self.cl_cp_costs_df,
            global_cp_cost=self.global_cp_cost,
            global_medoid=self.global_medoid,
            cl_method_name=self.cl_name,
            exec_name=name,
            nun_of_cl=self.n_cl
        )

    def create_clusters_df_dict(self):
        return {cp: cl.drop("merged_idx_pd", axis=1) for cp, cl in self.ClDist.clusters_gby}

    def create_clusters_df(self, concat_method=False):
        if concat_method:
            return pd.concat(
                [self.data.loc[self.labels[self.labels == cp].index] for cp in self.cps_df.index],
                keys=self.cps_df.index
            ).sort_index()

        return gpd.add_1st_lvl_index_to_df(self.data, self.labels, index_name="medoid").sort_index()

    @mf.print_execution_time
    def create_clusters_cp_costs_df_old(self):
        cl_dms_idx = [self.labels[self.labels == cp].index for cp in self.cps_df.index]
        return pd.concat(
            [self.dists_matrix.loc[cl_idx, cl_idx].mean() for cl_idx in cl_dms_idx],
            keys=self.cps_df.index
        )

    @mf.print_execution_time
    def create_clusters_cp_costs_df(self):
        # labels = self.labels.loc[self.dists_matrix.index].values
        labels = self.labels.values

        m_idx = isinstance(self.cps_df.index, pd.MultiIndex)
        cps_idx = self.cps_df.index.to_flat_index() if m_idx else self.cps_df.index
        """
        print("pam_m_line_777")
        print(self.labels)
        print(self.dists_matrix)
        print(self.cps_df)
        """

        d_matrix = self.dists_matrix.values

        cl_cost = self.create_cluster_cps_cost(labels, cps_idx, d_matrix)
        """
        cl_cost = []
        for k in np.arange(len(labels)):
            for cp in cps_idx:
                if labels[k] != cp:
                    continue

                cl_dists_sum = 0
                counter = 0
                for j in np.arange(len(labels)):
                    if labels[j] != cp:
                        continue
                    counter += 1
                    cl_dists_sum += d_matrix[k, j]

                cl_cost += [cl_dists_sum / counter]
        """

        m_idx_df = self.labels.index.to_frame()
        m_idx_df.insert(0, "medoids", self.labels.tolist())
        return pd.Series(cl_cost, index=pd.MultiIndex.from_frame(m_idx_df))

    @staticmethod
    def create_cluster_cps_cost(labels, cps_idx, d_matrix):
        cl_cost = []
        id_order = []

        sort_id = np.arange(len(labels))
        for i in range(len(cps_idx)):
            idx = labels == cps_idx[i]
            cl_cost += np.mean(d_matrix[np.ix_(idx, idx)], axis=0).tolist()
            id_order += sort_id[idx].tolist()

        cl_cost_n = [0] * len(labels)
        for i in range(len(labels)):
            cl_cost_n[id_order[i]] = cl_cost[i]

        return cl_cost_n


    def calc_silhouette_samples(self, simplified=True):
        if simplified:
            point_cl_mean_dists = self.cps_dist_matrix
        else:
            """
            print("pam_line_2102")
            point_cl_mean_dists = self.calc_mean_cl_dists_from_all_points_old().T
            print(point_cl_mean_dists)
            point_cl_mean_dists = self.calc_mean_cl_dists_from_all_points()
            print(point_cl_mean_dists)
            """
            if self.mean_cl_dists is None:
                self.calc_mean_cl_dists_from_all_points()
            point_cl_mean_dists = self.mean_cl_dists

        name = "Silhouette samples simplified" if simplified else "Silhouette samples"
        sil_sample = self.calc_silh_samples_from_d_matrix(d_matrix=point_cl_mean_dists, name=name)
        return sil_sample

    @mf.print_execution_time
    def calc_mean_cl_dists_from_all_points_old(self):
        point_cl_mean_dists = pd.DataFrame(index=self.cps_df.index, columns=self.labels.index, dtype="float32")
        for cl_name in self.cps_df.index:
            point_cl_mean_dists.loc[cl_name] = self.dists_matrix.loc[
                                               :,
                                               self.clusters_df.xs(cl_name, level=0).index
                                               ].apply(lambda x: x.drop(x.name)).mean(axis=1, skipna=True)
        return point_cl_mean_dists

    @mf.print_execution_time
    def calc_mean_cl_dists_from_all_points(self):
        pt_idx = self.labels.index
        cl_idx = self.cps_df.index
        p_cl_mean_d = np.empty([len(pt_idx), len(cl_idx)], dtype=np.float32)
        for i, cl_name in enumerate(cl_idx):
            dists_matrix = self.dists_matrix.loc[:, self.clusters_df.xs(cl_name, level=0).index]
            in_cl = dists_matrix.index.isin(dists_matrix.columns).astype("int")
            p_cl_mean_d[:, i] = np.sum(dists_matrix.values, axis=1) / (len(dists_matrix.columns) - in_cl)
        self.mean_cl_dists = pd.DataFrame(p_cl_mean_d, index=pt_idx, columns=cl_idx)

    @mf.print_execution_time
    def calc_silh_samples_from_d_matrix(self, d_matrix: pd.DataFrame, name="Silhouette samples"):
        """d_matrix is a matrix with dms as index and clusters as columns"""
        p_cl_mean_d = d_matrix.values
        """
        le = LabelEncoder()
        le.classes_ = np.array(d_matrix.columns)
        labels_c = le.transform(self.labels.values)

        print("pam_line_2421")
        print(list(d_matrix.columns))
        print(np.unique(list(self.labels.values)))
        print(np.unique(list(labels_c)))
        print(d_matrix)

        sil_sample = np.empty(shape=len(self.labels.index), dtype=np.float32)
        for i, cp in enumerate(labels_c):
            p_dists = p_cl_mean_d[i]
            a = p_dists[cp]
            b = np.min(np.delete(p_dists, cp))

            sil_sample[i] = (b - a) / max(b, a)
        """
        """
        labels_c = list(self.labels.values)
        for j, col in enumerate(d_matrix.columns):
            for i, row in enumerate(self.labels):
                if row == col:
                    labels_c[i] = j
        """

        sil_sample = np.empty(shape=len(self.labels.index), dtype=np.float32)
        j = 0   # useless, just for not underlining it
        for i, cp in enumerate(list(self.labels.values)):
            for j, col in enumerate(d_matrix.columns):
                if cp == col:
                    break
            p_dists = p_cl_mean_d[i]
            a = p_dists[j]
            b = np.min(np.delete(p_dists, j))

            sil_sample[i] = (b - a) / max(b, a)

        return pd.Series(data=sil_sample, index=self.labels.index, name=name, dtype="float32")


    def sklearn_silhouette_samples(self, data: pd.DataFrame, labels: pd.Series, dist_metric=None, dists_matrix=None):

        print("pam_line_2008")
        print(dists_matrix)
        print(labels)

        if isinstance(labels.iloc[0], tuple):
            labels = labels.loc[dists_matrix.index].apply(lambda x: dists_matrix.index.get_loc(x))
        # print(labels)
        if dists_matrix is not None:
            return pd.Series(
                silhouette_samples(
                    X=dists_matrix.reset_index(drop=True),
                    labels=labels.reset_index(drop=True),
                    metric="precomputed"),
                index=self.data.index,
                name="Silhouette samples sklearn"
            )
        else:
            return pd.Series(
                silhouette_samples(
                    X=data.values,
                    labels=labels,
                    metric=dist_metric),
                index=data.index,
                name="Silhouette samples sklearn"
            )

    def set_up_silhouette_score(self):
        self.sil_samples = self.sklearn_silhouette_samples(
            self.data, self.labels, dist_metric=self.dist_metric, dists_matrix=self.dists_matrix
        )
        """
        print("pam line 1693")
        print(self.data)
        print(self.labels)
        print(self.sil_samples)
        self.sil_samples = self.calc_silhouette_samples()
        print(self.sil_samples)
        """
        self.sil_clusters = self.silhouette_clusters_from_samples(self.sil_samples)
        """
        self.sil_clusters = pd.concat(
            [self.sil_samples, self.labels.rename("merged_idx_pd")], axis=1
        ).groupby(["merged_idx_pd"]).mean().squeeze().rename("cl_silhouette")
        """
        self.sil_score = self.sil_samples.mean()

    def silhouette_clusters_from_samples(self, sil_samples, name="Silhouette clusters"):
        return pd.Series(
            {cp: sil_samples.loc[self.clusters_df.loc[cp].index].mean() for cp in self.cps_df.index}
        ).rename(name)

    def calinski_harabasz_index(self, dists_p_norm: int | None = None):
        """
        cohesion = np.sum([
            (self.dist_func(self.clusters_df.loc[cp], self.cps_df.loc[cp]).abs().pow(self.dists_p_norm)).squeeze().sum()
            for cp in self.cps_df.index
        ])"""
        if dists_p_norm is None:
            dists_p_norm = self.dists_p_norm

        cohesion = self.ClDist.wcs_of_dists.sum()

        if self.dists_matrix is None:
            self.dists_matrix = self.dist_func(self.data, self.data)

        """
        print("pam_line_2317")
        print(self.no_im_cp_bool)
        print(self.data.loc[global_medoid_idx])
        print(self.cps_df)
        print(self.dist_func(self.data.loc[global_medoid_idx], self.data.mean(), bl=False))
        print("---")
        print(self.dist_func(
            self.cps_df, self.data.loc[global_medoid_idx], bl=self.no_im_cp_bool
        ).pow(2))
        print("---")
        """

        global_medoid = self.data.loc[self.global_cp_cost.idxmin()]

        if self.no_im_cp_bool:
            separation_cl = self.cps_dist_matrix.loc[global_medoid.name, self.cps_df.index].rename("Cl separation")
        else:
            separation_cl = self.dist_func(
                self.cps_df, global_medoid, False
            ).rename("Cl separation")

        separation = np.sum(separation_cl.pow(dists_p_norm) * self.cl_counts)     # .reindex(self.cps_df.index)))

        n_cl = len(self.cl_counts.index)
        sep_weight = n_cl - 1
        coh_weight = len(self.labels.index) - n_cl

        self.separation_cl = separation_cl
        self.global_medoid = global_medoid
        self.cal_har_index = (separation / sep_weight) / (cohesion / coh_weight)

    def sklearn_calinski_harabasz_modified(self, dists_p_norm: int | None = None):
        """
            Compute the Calinski and Harabasz score.

            It is also known as the Variance Ratio Criterion.

            The score is defined as ratio of the sum of between-cluster dispersion and
            of within-cluster dispersion.

            Read more in the :ref:`User Guide <calinski_harabasz_index>`.

            Parameters
            ----------
            :param dists_p_norm:

            X : array-like of shape (n_samples, n_features)
                A list of ``n_features``-dimensional data points. Each row corresponds
                to a single data point.

            labels : array-like of shape (n_samples,)
                Predicted labels for each sample.

            Returns
            -------
            score : float
                The resulting Calinski-Harabasz score.

            References
            ----------
            [1] `T. Calinski and J. Harabasz, 1974. "A dendrite method for cluster
               analysis". Communications in Statistics
               <https://www.tandfonline.com/doi/abs/10.1080/03610927408827101>`_

            """

        if dists_p_norm is None:
            dists_p_norm = self.dists_p_norm

        x_data = self.data.values
        cl_centers = self.cps_df.values

        le = LabelEncoder()
        le.classes_ = self.cps_df.index.values
        labels = le.transform(self.labels)

        n_samples, _ = x_data.shape
        n_labels = len(le.classes_)

        # check_number_of_labels(n_labels, n_samples)

        # extra_disp, intra_disp = 0.0, 0.0
        extra_disp = 0.0
        intra_disp = self.ClDist.wcs_of_dists.sum()

        # mean = np.mean(X, axis=0)

        global_medoid = self.data.loc[self.global_cp_cost.idxmin()]
        mean = global_medoid.values

        separation_cl = np.empty(shape=n_labels, dtype=np.float32)
        for k in range(n_labels):
            cluster_k = x_data[labels == k]
            # mean_k = np.mean(cluster_k, axis=0)
            mean_k = cl_centers[k]
            # extra_disp += len(cluster_k) * np.sum((mean_k - mean) ** 2)
            separation_cl[k] = self.dist_func(mean_k, mean, False)
            print("pam_line_1060")
            print(separation_cl[k])
            extra_disp += len(cluster_k) * separation_cl[k] ** dists_p_norm
            # intra_disp += np.sum((cluster_k - mean_k) ** 2)

        self.global_medoid = global_medoid
        self.separation_cl = pd.Series(separation_cl, index=self.cps_df.index, name="Cl separation")
        self.ch_index_skl_mod = (
            1.0
            if intra_disp == 0.0
            else extra_disp * (n_samples - n_labels) / (intra_disp * (n_labels - 1.0))
        )

    def sklearn_calinski_harabasz(self):
        try:
            return calinski_harabasz_score(
                self.data,
                labels=self.labels.apply(lambda y: list(self.labels.unique()).index(y))
            )
        except ValueError:
            return None

    def create_cluster_metrics_df(self):
        cl_series_list = [
            self.cl_counts,
            self.sil_clusters,
            self.silhouette_clusters_from_samples(self.sil_samples_spl, name="Silhouette simplified"),
            self.ClDist.distortion_cl,
            self.ClDist.inertia_cl,
            self.separation_cl
        ]
        return gpd.m_idx_to_tuple_idx_pd(pd.concat([
            cl_sr.reindex(self.ClDist.inertia_cl.index)
            for cl_sr in cl_series_list
        ], axis=1)).rename_axis("Medoid" if self.no_im_cp_bool else "Imaginary mid")

    def create_sample_metrics_df(self):
        s_data = gpd.m_idx_to_tuple_idx_pd(pd.concat(
            [
                self.sil_samples,
                self.sil_samples_spl,
                self.cps_dist_matrix.min(axis=1).rename("Min cl medoid"),
                self.mean_cl_dists.min(axis=1).rename("Mean cl distance"),
                self.dist_func(self.data, self.global_medoid).rename("Distance from global medoid"),
                self.labels.rename("Labels")
            ],
            axis=1
        ))

        s_data = s_data.set_axis(pd.Index(s_data.index.to_series().apply(lambda x: f"s({x})").values, name="Samples"))
        s_data = s_data.set_index(keys="Labels", drop=True, append=True)
        return s_data


class ClDistMetrics:
    def __init__(self, data: pd.DataFrame, labels: pd.Series, center_points: pd.DataFrame,
                 dist_func: callable, dist_metric: str | None, norm_ord=2,
                 clusters_df=None, cps_dist_matrix=None, medoid_le_dict: dict | None = None):

        self.data = data
        self.labels = labels
        self.cps_df = center_points
        self.dist_func = dist_func
        self.dist_metric = dist_metric
        self.norm_ord = norm_ord
        self.medoid_le_dict = medoid_le_dict

        self.no_im_cp_bool = all(self.cps_df.index.isin(self.labels.index))

        self.cl_counts = labels.value_counts()

        self.cps_dist_matrix = mf.def_var_value_if_none(
            value_passed=cps_dist_matrix, def_func=lambda: self.dist_func(self.data, self.cps_df, self.no_im_cp_bool)
        )

        print("pam_line_2732")
        print(self.cps_dist_matrix)

        if clusters_df is not None:
            self.clusters_gby = clusters_df.groupby(level=0)
        else:
            # self.clusters_gby = self.calc_clusters_groupby_label(self.data, self.labels)
            self.clusters_gby = self.data.groupby(by=self.labels)
        """
        self.wcs_of_dists = self.calc_clusters_sum_dists_pow_norm(
            clusters_gby=self.clusters_gby,
            cps_df=self.cps_df,
            dist_func=self.cps_dists_f_cache,
            norm_ord=self.norm_ord,
            cache=True
        )"""

        self.wcs_of_dists = self.calc_clusters_sum_dists_pow_norm(
            data=self.data,
            labels=self.labels,
            cps=self.cps_df,
            cps_dist_func=self.cps_dists_f_cache,
            dists_norm=self.norm_ord,
            cache=True
        )

        self.distortion_cl = (self.wcs_of_dists / self.cl_counts).rename("Distortion cl")

        self.distortion = self.calc_distortion_from(
            cluster_distortion=self.distortion_cl,
            cl_counts=self.cl_counts
        )

        self.inertia_cl = self.wcs_of_dists
        self.inertia_cl_total = self.inertia_cl.sum()
        self.inertia = self.calc_inertia(
            data=self.data,
            cps_df=self.cps_df,
            dist_func=self.cps_dists_f_cache,
            cache=True,
            norm_ord=self.norm_ord
        )

    def __repr__(self):
        print("ClDistMetrics obj stats")
        stats_df = pd.concat([self.cl_counts, self.inertia_cl, self.distortion_cl], axis=1)
        print(stats_df)
        print(stats_df.mean())
        print("total inertia")
        print(self.inertia)
        print("total distortion")
        print(self.distortion)
        return "ClDistMetrics obj"

    @classmethod
    def from_pam_core_obj(cls, pam_core_obj, labels=None):
        return cls(
            data=pam_core_obj.points,
            labels=mf.def_var_value_if_none(value_passed=labels, default=pam_core_obj.res_labels),
            center_points=pam_core_obj.res_cps_df,
            dist_func=pam_core_obj.dist_func,
            dist_metric=None,
            norm_ord=pam_core_obj.dists_p_norm,
            # clusters_df=pam_core_obj.res_clusters_df
        )

    def cps_dists_f_cache(self, a, b, bl=True):
        if bl:
            if isinstance(a, pd.DataFrame):
                if isinstance(b, pd.DataFrame):
                    return self.cps_dist_matrix.loc[a.index, b.index]
                else:
                    return self.cps_dist_matrix.loc[a.index, b.name]
            else:
                if isinstance(b, pd.DataFrame):
                    return self.cps_dist_matrix.loc[a.name, b.index]
                else:
                    return self.cps_dist_matrix.loc[a.name, b.name]
        else:
            return self.dist_func(a, b, bl)

    @staticmethod
    def calc_clusters_sum_dists_pow_norm(data, labels, cps, cps_dist_func, dists_norm, cache=True):
        return pd.Series([
            cps_dist_func(data.loc[labels == cp], cps.loc[cp]).abs().pow(dists_norm).sum()
            for cp in cps.index
        ], index=cps.index, name=f"WCS^{dists_norm})")

    @staticmethod
    def calc_clusters_sum_dists_pow_norm_old(clusters_gby, cps_df, dist_func, norm_ord, cache=True):
        # Within Cluster Sum of Powers
        print("pam_line_2640")
        print(cps_df)
        return clusters_gby.apply(
            lambda cl: np.sum(
                np.abs(dist_func(
                    cl.droplevel(level=0, axis=0),
                    cps_df.loc[cl.name],
                    bl=cache
                )) ** norm_ord
            )
        ).squeeze().rename(f"WCS^{norm_ord})")

    # NOT USED
    @staticmethod
    def calc_clusters_sum_dists_pow_norm_other_new_old(clusters_gby, cps_df, dist_func, norm_ord=2, cache=True):
        # Within Cluster Sum of Powers
        wcs_s = np.empty(shape=len(cps_df.index), dtype=np.float32)
        norm_bool = not norm_ord == 2
        wcs_p = np.empty(shape=len(cps_df.index), dtype=np.float32) if norm_bool else wcs_s
        for i, (g_name, g_df) in enumerate(clusters_gby):
            cl_d_array = np.abs(dist_func(
                g_df.droplevel(level=0, axis=0),
                cps_df.loc[g_name],
                bl=cache
            ))
            wcs_p[i] = np.sum(cl_d_array ** norm_ord)
            if norm_bool:
                wcs_s[i] = np.sum(cl_d_array ** 2)

        return (
            pd.Series(wcs_p, index=cps_df.index, name=f"WCS^{norm_ord}"),
            pd.Series(wcs_s, index=cps_df.index, name="WCS^2")
        )

    @staticmethod
    def calc_distortion_from(cluster_distortion, cl_counts):
        # sum(cluster_distortion * cl_counts / cl_counts.sum())
        return sum(cluster_distortion * cl_counts) / cl_counts.sum()

    @classmethod
    def calc_inertia(cls, data, cps_df, dist_func, norm_ord=2, cache=True):
        """
        return np.sum(np.min(cls.apply_dot_difference_function(
            matrix_1=data,
            matrix_2=cps_df,
            function=lambda x: cls.calc_row_norm_dist_squared(x, order=order)),
            axis=1))
        """
        return np.sum(np.abs(np.min(dist_func(data, cps_df, bl=cache), axis=1) )**norm_ord)

    def aprox_distortion_from_inertia(self):
        return self.inertia / len(self.data.index)

    def aprox_distortion_from_cl_inertia(self):
        return self.inertia_cl.sum() / len(self.data.index)

    # NOT USED
    @staticmethod
    def apply_dot_difference_function(matrix_1, matrix_2, function):
        # results have matrix_1 rows and matrix_2 columns
        return pd.concat(
            [
                matrix_1.apply(lambda row: function(row - matrix_2.loc[cp]), axis=1)
                for cp in matrix_2.index
            ],
            axis=1
        )

    # NOT USED
    @classmethod
    def calc_cluster_distortion_from(cls, clusters_gby, cps_df, dist_func, norm_ord, cache=True):
        """
        return clusters_gby.apply(lambda cl: np.mean(cls.calc_row_norm_dist_squared(cl - cps_df.loc[cl.name])))\
            .squeeze().rename("cl_distortion")
        """
        return clusters_gby.apply(
            lambda cl: np.mean(
                np.abs(dist_func(
                    cl.droplevel(level=0, axis=0),
                    cps_df.loc[cl.name],
                    cache
                ))**norm_ord
            )
        ).squeeze().rename("cl_distortion")

