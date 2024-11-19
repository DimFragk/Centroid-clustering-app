from collections.abc import Callable

import pandas as pd
import numpy as np
# from scipy.spatial.distance import cdist
# from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull
# from scipy.stats import pmean
from numba import njit

import math
from time import time, process_time
from itertools import combinations
from typing import Literal

from utils import general_functions as mf
import utils.math_functions as gmf
import utils.pandas_functions as gpd
import utils.data_processing as gdp
# import utils.visualization_functions as gvp
from utils.general_functions import def_var_value_if_none

from centroid_clustering.clustering_metrics import DistFunction, ClDistMetrics


class Kmedoids:
    st_p_method_options = ["convex_hull", "kmeans++", "rand_min_cost", "pam_build", "pam_build_iter", "random"]
    n_cl_search_strategy_options = ["best_n_cl", "break_desc", "break_desc_ex"]
    def __init__(
            self,
            data: pd.DataFrame,
            n_clusters: int,
            n_init: int = 1,
            dim_redux: int = None,
            st_p_method: Literal["convex_hull", "kmeans++", "rand_min_cost", "pam_build", "pam_build_iter"] | str = "pam_build_iter",
            starting_points: list[int | str | tuple[int, str]] = None,
            im_st_points: np.ndarray | pd.DataFrame | pd.Series | list = None,
            dist_func_obj: DistFunction | pd.DataFrame = None,
            dist_metric: str = "euclidean",
            norm_ord: int = 2,
            dists_norm: int = 1,
            cand_options: dict[str, (int, str, bool)] = None,
            max_iter: int = 200,
            iter_cp_comb: int = 500,
            mid_point_approx: bool = True,
            break_comb_loop: bool = True,
            n_cl_search_strategy: Literal["best_n_cl", "break_desc", "break_desc_ex"] = "break_desc_ex"
    ):

        self.data = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
        self.num_of_clusters = n_clusters

        self.n_init = n_init
        self.dim_redux = dim_redux

        self.st_p_method = st_p_method
        self.starting_points = starting_points
        self.im_st_points = im_st_points

        self.dists_norm_ord = dists_norm
        self.cand_options = cand_options

        self.max_num_of_iter = max_iter
        self.iter_cp_comb = iter_cp_comb

        self.mid_point_approx = mid_point_approx
        self.break_comb_loop = break_comb_loop
        self.n_cl_search_strategy = n_cl_search_strategy

        self.num_of_dimensions = len(self.data.columns)
        self.num_of_points = len(self.data.index)

        self.points: pd.DataFrame | None = None
        self.points_no_dupl: pd.DataFrame | None = None
        self.DistFunc: DistFunction | None = None

        self.data_set_up(dim_redux=dim_redux, dist_func_obj=dist_func_obj, dist_metric=dist_metric, norm_ord=norm_ord)

        self.idx_min_dist_to_im_st_p = pd.Index([])
        self.candidate_st_p: pd.DataFrame | None = None
        self.st_p_df_list = []

        self.set_up_iter_starting_points(print_init_cost=True)

        self.starting_points_df = pd.DataFrame()
        self.PamCore: PamCore | None = None
        self.cost_metrics = pd.DataFrame()

        self.select_best_pam_of_st_points()

        self.labels = pd.Series()
        self.medoids = pd.DataFrame()
        self.data_df = pd.DataFrame()
        self.clusters_df = pd.DataFrame()
        self.inertia: float | None = None

        self.set_up_res_from_pam_core(pam_core=self.PamCore)

    def print_starting_points(self):
        print("Line 246 pam")
        print(f"Number of clusters: {self.num_of_clusters}")
        print(f"Starting data_pd passed: \n{self.starting_points}")
        print(f"Candidates selected: \n{list(self.candidate_st_p.index)}")
        print(f"DMs picked for starting point: \n{list(self.starting_points_df.index)}")
        print(self.starting_points_df)

    def print_cl_type(self):
        print("\n\n---\n\tClustering set up\n")
        print(f"Num of points: {self.num_of_points}")
        print(f"Num of dimensions: {self.num_of_dimensions}")
        print(f"Dimensions after reduction: {len(self.points.columns)}, redux: {self.dim_redux}")
        print(f"Max number of iterations (loops): {self.max_num_of_iter}")
        print("---")
        print(f"1d distance function passed (bool): {self.DistFunc.dist_func_1d is None}")
        print(f"Distance metric: {self.DistFunc.dist_metric}")
        print(f"Norm order: {self.DistFunc.norm_order}")
        print("---")
        print("Candidate center points options")
        print(f"Max number of candidate cluster medoids: {self.PamCore.n_cp_cand}")
        print(f"Imaginary center point method: {self.PamCore.im_mid_type}")
        print("---")
        print(f"Starting points selection method: {self.st_p_method}")
        print(f"Starting medoids passed: {self.starting_points}")
        print(f"Imaginary starting points passed: {self.im_st_points}")
        print(f"Candidate starting medoids: {self.candidate_st_p}")
        print(f"Starting medoids: {self.starting_points_df}")
        print("\nEnd of centroid_clustering report---")

    def print_selected(self):
        """
        result_dict = self.iter_data[self.iter_run_count - 1]
        print(result_dict["centers"])
        print(result_dict["n_clusters"])
        """
        self.print_cl_type()
        print("---")
        print(self.medoids)
        print(self.labels)
        print("---")

    def print_test_random_st_p_selection(self):
        self.starting_points = ["DM_17"]
        self.candidate_st_p = self.points
        self.starting_points_df = self.select_starting_points()
        self.print_starting_points()

        self.starting_points = ["DM_4"]
        self.candidate_st_p = self.points.loc[["DM_9", "DM_10"]]
        self.starting_points_df = self.select_starting_points()
        self.print_starting_points()

        self.starting_points = ["DM_17", "DM_4"]
        self.candidate_st_p = self.points.loc[["DM_9", "DM_16", "DM_17"]]
        self.starting_points_df = self.select_starting_points()
        self.print_starting_points()

        self.starting_points = ["DM_17", "DM_4", "DM_283", "DM_4", "DM_5", "DM_6"]
        self.candidate_st_p = self.points.loc[["DM_9", "DM_16", "DM_17"]]
        self.starting_points_df = self.select_starting_points()
        self.print_starting_points()

        self.starting_points = ["DM_17", "DM_4", "DM_283", "DM_4", "DM_5", "DM_6"]
        self.candidate_st_p = self.points.loc[["DM_9", "DM_16", "DM_17"]]
        self.starting_points_df = self.choose_n_priority_no_replace(
            first=self.points.loc[self.points.index.isin(self.starting_points)],
            second=self.candidate_st_p,
            third=self.points,
            num=self.num_of_clusters)
        self.print_starting_points()

        self.starting_points = ["DM_13", "DM_433", "DM_4"]
        self.starting_points = [3]
        self.starting_points_df = self.select_starting_points()
        self.print_starting_points()

        self.starting_points = [16]
        self.starting_points_df = self.select_starting_points()
        self.print_starting_points()

        self.starting_points = [16, 3]
        self.starting_points_df = self.select_starting_points()
        self.print_starting_points()

        self.starting_points = [16, 3, 123, 235, 123, 1244, 3, 16, 5]
        self.starting_points_df = self.select_starting_points()
        self.print_starting_points()

    def print_init_config_cost(self):
        print("Initial starting points configuration cost")
        print(f"Initialization method: {self.st_p_method}")
        print(f"Starting points: {self.st_p_method}")
        print(self.st_p_df_list)
        print(f"Cost:")
        print(self.calc_init_cp_cost(self.st_p_df_list))

    def __repr__(self):
        # self.print_all_iter_per_table()
        # self.print_all_tables_per_iter()
        self.print_selected()
        return "Kmedoids object"

    def set_up_res_from_pam_core(self, pam_core=None):
        pam_core = mf.def_var_value_if_none(pam_core, default=self.PamCore)

        cps = pam_core.res_cps_df
        self.labels = pam_core.labels
        self.data_df = pd.DataFrame(self.data, index=self.labels.index)
        self.medoids = self.data_df.loc[cps.index]
        self.clusters_df = gpd.add_1st_lvl_index_to_df(self.data_df, self.labels, index_name="medoid").sort_index()
        self.inertia = pam_core.final_config_cost

    @mf.print_execution_time
    def select_best_pam_of_st_points(self):
        # self.print_test_random_st_p_selection()
        pam_core_list = []
        inertia_list = []
        distortion_list = []
        config_cost_list = []
        for st_p_df in self.st_p_df_list:
            print("pam_line_270")
            print(f"starting points \n{st_p_df.index}")
            # print(self.points.loc[st_p_df.index].drop_duplicates())
            # print(f"num of np.nan: ", self.DistFunc.num_of_nan_in_cache())

            pam_core_res = PamCore.from_k_medoids_obj(
                k_medoids_obj=self,
                starting_medoids=st_p_df,
                dist_func=self.DistFunc.distance_func_cache_all
            )
            pam_core_list += [pam_core_res]
            inertia_list += [pam_core_res.CDMetrics.inertia]
            distortion_list += [pam_core_res.CDMetrics.distortion]
            config_cost_list += [pam_core_res.final_config_cost]

        """
        inertia_list = list(map(lambda x: x.CDMetrics.inertia, pam_core_list))
        distortion_list = list(map(lambda x: x.CDMetrics.distortion, pam_core_list))
        config_cost_list = list(map(lambda x: x.final_config_cost, pam_core_list))
        """

        print("pam_line_286")
        print("Best results set:")
        print(f"\t-> inertia:\tset({np.argmin(inertia_list) + 1})")
        print(f"\t-> distortion:\tset({np.argmin(distortion_list) + 1})")
        print(f"\t-> configuration cost:\tset({np.argmin(config_cost_list) + 1})")
        print(f" Configuration costs:\n{config_cost_list}")
        print(f" Distortions:\n{distortion_list}")
        print(f" Inertia:\n{inertia_list}")

        best_idx = int(np.argmin(config_cost_list))
        print(f"Best configuration cost starting set\n{self.st_p_df_list[best_idx].index}")
        print(pam_core_list[best_idx].CDMetrics)

        self.cost_metrics = pd.Series(
            {"config_cost": config_cost_list, "inertia": inertia_list, "distortion": distortion_list}
        )
        self.starting_points_df = self.st_p_df_list[best_idx]
        self.PamCore: PamCore = pam_core_list[best_idx]

    @mf.print_execution_time
    def data_set_up(
            self,
            dim_redux: int = None,
            dist_func_obj: DistFunction=None,
            dist_metric=None,
            norm_ord=None,
    ):

        data_df = pd.DataFrame(self.data, dtype="float32")

        if dim_redux is not None:
            # TODO PCoA method for all dist functions
            data_df = gdp.apply_pca(dataframe=data_df, remaining_dim=dim_redux)

        if dist_func_obj is None:
            print("New distance function is calculated")
            self.DistFunc = DistFunction(
                dist_metric=dist_metric,
                norm_order=norm_ord,
                vector_dist_func=None,
                cache_points=data_df
            )
        elif isinstance(dist_func_obj, DistFunction):
            if not dist_func_obj.check_cache_compatibility(data_df):
                print("ckm_line_76")
                print("Creating new cache")
                print(f"Points index: {data_df.index}")
                print(f"Old cache index: {dist_func_obj.dists_matrix_cache.index}")
                dist_func_obj.df_to_cache(df=data_df, calc_dists_matrix=True)
            else:
                print("ckm_line_82")
                print("DistFunc object cache is compatible")

            self.DistFunc = dist_func_obj

        elif isinstance(dist_func_obj, pd.DataFrame) and dist_func_obj.shape[0] == dist_func_obj.shape[1]:
            print("Distances matrix passed")
            self.DistFunc = DistFunction.set_up_from_dist_matrix(
                dist_func_obj, dist_metric=dist_metric, norm_order=norm_ord
            )

        self.points = data_df
        self.points_no_dupl = data_df.drop_duplicates()

    def select_st_p_from_convex_hull(self, dim=3):
        """
        if self.dim_redux is not None:
            # If dimension reduction is applied to "self.data_pd", create convex_hull of same dimensions
            convex_hull = ConvexHull(mf.apply_pca(dataframe=self.data_pd, remaining_dim=self.dim_redux))
        elif dim is not None:
            # If dimension are NOT reduced, this "dim" option creates convex_hull of dimensions given
            convex_hull = ConvexHull(mf.apply_pca(dataframe=self.data_pd, remaining_dim=dim))
        else:
            convex_hull = ConvexHull(self.data_pd)
        """
        convex_hull = ConvexHull(gdp.apply_pca(dataframe=self.points_no_dupl, remaining_dim=dim))

        labels = pd.Series(0, index=self.points_no_dupl.index, name="label")
        labels.iloc[convex_hull.vertices] = 1
        """
        print(convex_hull.data_pd.shape)
        print(convex_hull.vertices)
        print(convex_hull.simplices.shape)
        print(convex_hull.simplices)
        print(convex_hull.area)
        print(convex_hull.volume)
        print(merged_idx_pd.loc[merged_idx_pd.isin([1])])
        # fig1 = mf.set_up_3d_graph_data(data=self.data_pd, merged_idx_pd=merged_idx_pd, select_redux="PCA")
        # fig1.show()
        """
        return self.points_no_dupl.loc[labels.isin([1])]

    def k_means_plus_plus_st_p_method(self, st_p_idx: pd.Index | None = None, random_state=42):
        flag = (st_p_idx is None or (isinstance(st_p_idx, pd.Index | pd.MultiIndex) and st_p_idx.empty))
        center_points = [0] if flag else list(self.points.index.get_indexer(st_p_idx))

        num_of_points = self.num_of_clusters - len(center_points) + 1
        int_idx = self.plus_plus(
            ds=self.points.values, k=num_of_points, cps_int_idx=center_points, random_state=random_state
        )
        return self.points.iloc[int_idx]

    @staticmethod
    def plus_plus(ds, k, cps_int_idx: list | None = None, random_state=42, rtn_idx=True):
        """
        Create cluster centroids using the k-means++ algorithm.
        Parameters
        ----------
        ds : numpy array
            The dataset to be used for centroid initialization.
        k : int
            The desired number of clusters for which centroids are required.
        Returns
        -------
        centroids : numpy array
            Collection of k centroids as a numpy array.
        Inspiration from here: https://stackoverflow.com/questions/5466323/how-could-one-implement-the-k-means-algorithm
        """

        if cps_int_idx:
            centroids = [ds[i] for i in cps_int_idx]
            int_idx = cps_int_idx
        else:
            centroids = [ds[0]]
            int_idx = [0]

        np.random.seed(random_state)
        for _ in range(1, k):
            dist_sq = np.array([min([np.inner(c - x, c - x) for c in centroids]) for x in ds])
            probs = dist_sq / dist_sq.sum()
            cumulative_probs = probs.cumsum()
            r = np.random.rand()

            for j, p in enumerate(cumulative_probs):
                if r < p:
                    i = j
                    break

            centroids.append(ds[i])
            int_idx.append(i)

        if rtn_idx:
            return int_idx
        else:
            return np.array(centroids),

    def random_min_cost_st_p_method(self, idx_slt=None):
        min_cost = np.inf
        min_int_idx_cp_list = None
        dist_matrix = self.DistFunc.dists_matrix_cache.values

        if idx_slt is None:
            rand_int_gen = lambda : np.random.choice(self.num_of_points, size=self.num_of_clusters, replace=False)
        else:
            n_point = self.num_of_clusters - len(idx_slt)
            id_list = list(idx_slt)
            rand_int_gen = lambda : list(np.random.choice(self.num_of_points, size=n_point, replace=False)) + id_list

        for _ in range(20):
            int_idx = rand_int_gen()
            r_cost = np.sum(np.min(
                np.abs(dist_matrix[int_idx]), axis=0
            ) ** self.dists_norm_ord)
            if min_cost > r_cost:
                min_cost = r_cost
                min_int_idx_cp_list = int_idx
        return self.points.iloc[min_int_idx_cp_list]

    @mf.print_execution_time
    def pam_build_iter_st_p_method(self, st_p_idx: pd.Index | None = None):
        dist_matrix_df = self.DistFunc.distance_func_cache_all()
        flag = False
        if st_p_idx is None or (isinstance(st_p_idx, pd.Index | pd.MultiIndex) and st_p_idx.empty):
            first_cp = dist_matrix_df.sum().idxmin()
            d_nearest = dist_matrix_df[first_cp].values.copy()
            center_points = [first_cp]
        else:
            d_nearest = dist_matrix_df[st_p_idx].min(axis=1).values.copy()
            center_points = list(st_p_idx)
            if self.num_of_clusters < 7 or len(st_p_idx) < self.num_of_clusters / 2:
                flag = True

        dist_matrix = dist_matrix_df.values
        num_of_point = self.num_of_clusters - len(center_points)
        center_points = self.built_n_points_with_min_cost(num_of_point, dist_matrix, d_nearest, center_points)

        if flag:
            return self.points.loc[center_points]

        cost_1 = self.calc_comb_cp_cost(center_points, dist_matrix_df, self.dists_norm_ord)

        poss = self.num_of_clusters * 2 // 3
        new_cps = center_points[-poss:]
        d_nearest = dist_matrix_df[new_cps].min(axis=1).values.copy()
        num_of_point = self.num_of_clusters - len(new_cps)
        new_center_points = self.built_n_points_with_min_cost(num_of_point, dist_matrix, d_nearest, new_cps)

        cost_2 = self.calc_comb_cp_cost(new_center_points, dist_matrix_df, self.dists_norm_ord)

        """
        print("ckm_line_393_test")
        cost = self.calc_comb_cp_cost(center_points, dist_matrix_df, self.dists_norm_ord)
        print(center_points)
        print(cost)
        min_cp_comb = None
        min_cost = cost
        for i in range(1, self.num_of_clusters):
            print(center_points[-i:])
            new_cps = center_points[-i:]
            d_nearest = dist_matrix_df[new_cps].min(axis=1).values.copy()
            num_of_point = self.num_of_clusters - len(new_cps)
            cps = self.built_n_points_with_min_cost(num_of_point, dist_matrix, d_nearest, new_cps)
            print(cps)
            cost = self.calc_comb_cp_cost(cps, dist_matrix_df, self.dists_norm_ord)
            print(cost)
            if min_cost > cost:
                min_cost = cost
                min_cp_comb = cps

        center_points = min_cp_comb
        print(f"min_cost: {min_cost}")
        min_cp_comb = None
        min_cost = cost
        for i in range(1, self.num_of_clusters):
            print(center_points[-i:])
            new_cps = center_points[-i:]
            d_nearest = dist_matrix_df[new_cps].min(axis=1).values.copy()
            num_of_point = self.num_of_clusters - len(new_cps)
            cps = self.built_n_points_with_min_cost(num_of_point, dist_matrix, d_nearest, new_cps)
            print(cps)
            cost = self.calc_comb_cp_cost(cps, dist_matrix_df, self.dists_norm_ord)
            print(cost)
            if min_cost > cost:
                min_cost = cost
                min_cp_comb = cps

        print(f"min_cost: {min_cost}")
        """

        cps = new_center_points if cost_1 > cost_2 else center_points
        return self.points.loc[cps]

    @mf.print_execution_time
    def pam_build_st_p_method(self, st_p_idx: pd.Index | None = None):
        dist_matrix = self.DistFunc.distance_func_cache_all()

        if st_p_idx is None or (isinstance(st_p_idx, pd.Index | pd.MultiIndex) and st_p_idx.empty):
            first_cp = dist_matrix.sum().idxmin()
            d_nearest = dist_matrix[first_cp].values.copy()
            dist_matrix = dist_matrix.values
            center_points = [first_cp]
        else:
            d_nearest = dist_matrix[st_p_idx].min(axis=1).values.copy()
            dist_matrix = dist_matrix.values
            center_points = list(st_p_idx)

        num_of_point = self.num_of_clusters - len(center_points)
        center_points = self.built_n_points_with_min_cost(num_of_point, dist_matrix, d_nearest, center_points)

        return self.points.loc[center_points]

    @staticmethod
    # @mf.print_execution_time
    @njit
    def built_n_points_with_min_cost(
            nun_of_points: int,
            dist_matrix: np.ndarray,
            d_nearest: np.ndarray,
            center_points: list
    ):
        for _ in range(nun_of_points):
            min_cost = 0
            min_cp = None
            for cp in range(len(dist_matrix)):
                if cp in center_points:
                    continue
                idx = dist_matrix[cp] < d_nearest
                cp_cost = np.sum(dist_matrix[cp][idx] - d_nearest[idx])
                if min_cost > cp_cost:
                    min_cost = cp_cost
                    min_cp = cp

            # Update nearest distances
            center_points += [min_cp]
            min_cp_dists = dist_matrix[int(min_cp)]
            """
            d_nearest = np.array(
                [np.min([min_cp_dists[i], d_nearest[i]]) for i in range(len(d_nearest))]
            )
            """
            for i in range(len(d_nearest)):
                val1 = min_cp_dists[i]
                val2 = d_nearest[i]
                if val1 < val2:
                    d_nearest[i] = val1

        return center_points

    def reverse_pam_build_from_many_center_points(self, idx_slt):
        dist_matrix_df = self.DistFunc.dists_matrix_cache
        n_p = len(idx_slt) - self.num_of_clusters
        min_cost = np.inf
        min_cps = None
        id_list = list(idx_slt)
        for i in range(n_p):
            for i in range(len(id_list)):
                cps = id_list[:i] + id_list[i + 1:]
                cost = self.calc_comb_cp_cost(cps, dist_matrix_df, self.dists_norm_ord)
                if cost < min_cost:
                    min_cost = cost
                    min_cps = cps
            id_list = min_cps

        return self.points.loc[min_cps]

    @staticmethod
    def calc_comb_cp_cost(cp_comb, dist_matrix, dists_norm_ord=1):
        return np.sum(np.min(
            np.abs(dist_matrix.loc[cp_comb].values), axis=0
        ) ** dists_norm_ord)

    def calc_init_cp_cost(self, cps_df_list):
        dist_matrix = self.DistFunc.distance_func_cache_all()
        return [
            self.calc_comb_cp_cost(cp_comb.index, dist_matrix, self.dists_norm_ord)
            for cp_comb in cps_df_list
        ]

    @mf.print_execution_time
    def set_up_iter_starting_points(self, print_init_cost=False):
        idx_slt = self.convert_st_points_passed_to_index(
            idx_list=self.starting_points, vector_list=self.im_st_points
        )
        if len(idx_slt) == self.num_of_points:
            st_p_df = self.points.loc[idx_slt]
            self.st_p_df_list = [st_p_df for _ in range(self.n_init)]
            return
        elif len(idx_slt) > self.num_of_clusters:
            cps_df = self.reverse_pam_build_from_many_center_points(idx_slt)
            self.st_p_df_list = [cps_df for _ in range(self.n_init)]
            return

        if self.st_p_method == "pam_build":
            self.st_p_df_list = [self.pam_build_st_p_method(st_p_idx=idx_slt) for _ in range(self.n_init)]
        elif self.st_p_method == "pam_build_iter":
            self.st_p_df_list = [self.pam_build_iter_st_p_method(st_p_idx=idx_slt) for _ in range(self.n_init)]
        elif self.st_p_method == "kmeans++":
            self.st_p_df_list = [self.k_means_plus_plus_st_p_method(idx_slt, i) for i in range(self.n_init)]
            """
            st_p_df_list = []
            for i in range(self.n_init):
                int_idx = self.plus_plus(self.points_no_dupl.values, self.num_of_clusters)
                st_p_df_list.append(self.points_no_dupl.iloc[int_idx])
            self.st_p_df_list = st_p_df_list
            """
        elif self.st_p_method == "rand_min_cost":
            self.st_p_df_list = [self.random_min_cost_st_p_method(idx_slt=idx_slt) for _ in range(self.n_init)]
        elif self.st_p_method == "convex_hull":
            self.candidate_st_p = self.select_st_p_from_convex_hull(dim=3)
            self.st_p_df_list = self.select_n_st_p_sets_from_candidates(self.candidate_st_p, idx_slt)
        else:
            self.st_p_df_list = self.select_n_st_p_sets_from_candidates(self.points_no_dupl, idx_slt)

        if print_init_cost:
            self.print_init_config_cost()

    def find_closest_points_to_im_centers(self, vector_list):
        return pd.Index(self.DistFunc.dist_func(self.points, vector_list).idxmin(axis=0).squeeze().unique())

    def convert_st_points_passed_to_index(self, idx_list: pd.Index | list=None, vector_list: list | np.ndarray=None):
        if vector_list is not None:
            self.idx_min_dist_to_im_st_p = self.find_closest_points_to_im_centers(vector_list)

        if idx_list is None or (isinstance(idx_list, list) and not idx_list):
            return self.idx_min_dist_to_im_st_p
        elif isinstance(idx_list[0], int) and not isinstance(self.points.index[0], int):
            return self.points.iloc[idx_list].index.union(self.idx_min_dist_to_im_st_p)
        else:
            print("pam_line_611")
            print(idx_list)
            return self.points.index.intersection(idx_list).union(self.idx_min_dist_to_im_st_p)

    @classmethod
    def choose_n_priority_no_replace(cls, first: pd.DataFrame, second: pd.DataFrame, third: pd.DataFrame, num):
        # This function return dataframe with "n" rows from priority dfs
        # if "num" is bigger from priority df rows the whole df is included,
        # else random samples of each rows are chosen

        remaining = num - len(first.index)
        if remaining < 0:
            """
            return first.loc[list(
                np.random.choice(list(first.index), size=num, replace=False)
            )]
            """
            return mf.select_n_random_rows_from_df(first, n_row=num)

        if remaining == 0:
            return first
        else:
            _2_not_in_1_ = second.loc[~second.index.isin(first.index)]
            _3_not_in_1_ = third.loc[~third.index.isin(first.index)]
            return pd.concat([first, cls.choose_n_priority_no_replace(
                first=_2_not_in_1_, second=_3_not_in_1_, third=pd.DataFrame(), num=remaining)])

    def select_starting_points(self, candidate_st_p=None, idx_slt=None):
        cand_st_p = mf.def_var_value_if_none(value_passed=candidate_st_p, default=self.candidate_st_p)

        first_priority = None
        if idx_slt is not None:
            try:
                first_priority = self.points_no_dupl.loc[idx_slt]
            except KeyError:
                print("Selected points have duplicates, they are discarded.")

        if first_priority is None:
            return self.choose_n_priority_no_replace(
                first=cand_st_p, second=self.points_no_dupl, third=pd.DataFrame(), num=self.num_of_clusters
            )

        return self.choose_n_priority_no_replace(
            first=first_priority, second=cand_st_p, third=self.points_no_dupl, num=self.num_of_clusters
        )

    def select_n_st_p_sets_from_candidates(self, candidate_st_p, idx_slt=None):
        cand_df = def_var_value_if_none(candidate_st_p, default=self.candidate_st_p)

        if self.n_init == 1:
            return [self.select_starting_points(cand_df, idx_slt)]

        n_cl = self.num_of_clusters
        n_sets = self.n_init

        unique_cand = n_sets * n_cl
        if unique_cand > len(cand_df.index):
            np_rand_int = np.random.choice(len(cand_df.index), size=unique_cand, replace=False)
            int_idx_list = [
                np_rand_int[i*n_cl:(i+1)*n_cl] for i in range(n_sets)
            ]
        else:
            int_idx_list = [
                np.random.choice(len(cand_df.index), size=self.num_of_clusters, replace=False)
                for i in range(n_sets)
            ]

        return [
            self.select_starting_points(cand_df.iloc[i_idx], idx_slt)
            for i_idx in int_idx_list
        ]


class PamCore:

    im_mid_funcs_dict = {
        "g_median": gmf.geometric_median,
        "median": lambda x: np.median(x, axis=0),
        "mean": lambda x: np.mean(x, axis=0)
    }

    def __init__(
            self,
            points: pd.DataFrame,
            st_medoids: pd.DataFrame,
            dist_func: Callable,
            points_dupl: pd.DataFrame | None = None,
            cand_options: dict[str, (int, str, bool)] = None,
            dists_p_norm=1,
            max_iter=200,
            iter_comb=1000,
            mid_point_approx: bool = True,
            break_comb_loop: bool = True,
            n_cl_search_strategy: Literal["best_n_cl", "break_desc", "break_desc_ex"] = "break_desc_ex"
    ):

        self.points = points
        self.num_of_points = len(points.index)
        self.points_dupl = self.points.drop_duplicates if points_dupl is None else points_dupl

        self.st_medoids = st_medoids
        self.dist_func = dist_func

        self.dists_p_norm = dists_p_norm

        self.max_iter = max_iter
        self.iter_comb = iter_comb
        self.mid_point_approx = mid_point_approx

        self.break_comb_loop = break_comb_loop
        self.n_cl_search_strategy = n_cl_search_strategy

        self.im_mid_type = None

        if not isinstance(cand_options, dict):
            self.n_cp_cand = None
            self.im_mid_type = None
            self.p_mean_ord = None
        else:
            self.n_cp_cand = mf.def_var_value_if_none(cand_options.get("n_cp_cand"), default=5)
            self.im_mid_type = mf.def_var_value_if_none(cand_options.get("im_mid_type"), default="g_median")
            self.p_mean_ord = mf.def_var_value_if_none(cand_options.get("p_mean_ord"), default=1)

        self.n_cl = len(self.st_medoids.index)
        self.comb_cl_depth = 4
        self.n_cl_change = min(self.n_cl, 5)

        if self.n_cl_search_strategy == "best_n_cl":
            self.fix_n_cl_order = True
            self.break_n_cl_loop = False
        elif self.n_cl_search_strategy == "break_desc":
            self.fix_n_cl_order = True
            self.break_n_cl_loop = True
        elif self.n_cl_search_strategy == "break_desc_ex":
            self.fix_n_cl_order = False
            self.break_n_cl_loop = True

        self.prl_cp_ch_order = list(range(self.n_cl, 0, -1))  # parallel cp change order
        # self.prl_cp_ch_order = list(range(1, self.n_cl+1))  # parallel cp change order
        self.n_cp_change_iter = []  # n_cl comb change history

        self.cp_names_iter = []
        self.labels_iter = []
        self.cps_dists_matrix_iter = []
        self.config_cost_iter = []

        self.imaginary_mid_points_iter = []

        self.iter_data = []
        self.iter_run_count = 0

        self.res_cps_df = pd.DataFrame()
        self.labels = pd.Series()
        self.cps_dists_matrix = pd.DataFrame()
        self.final_config_cost = None
        self.res_clusters_df = pd.DataFrame()
        self.res_clusters_df_dict = pd.DataFrame()

        self.CDMetrics: ClDistMetrics | None = None

        self.run_loop()
        # self.create_iter_data_dict_list()

    def print_all_iter_per_table(self):
        mf.print_list(self.labels_iter)
        # mf.print_list(self.clusters_dict_iter)
        mf.print_list(self.imaginary_mid_points_iter)
        mf.print_list(self.cp_names_iter)

    def print_all_tables_per_iter(self):
        for i, iter_tables in enumerate(self.iter_data):
            print("Iteration {}".format(i))
            for key, table in iter_tables.items():
                print("Table: {}".format(key))
                if isinstance(table, dict):
                    mf.print_dictionary(table)
                else:
                    print(table)

    def __repr__(self):
        return "PamCore obj"

    @classmethod
    def from_k_medoids_obj(
            cls, k_medoids_obj: Kmedoids,
            starting_medoids: pd.DataFrame, points=None,
            dist_func=None
    ):

        return cls(
            points=mf.def_var_value_if_none(value_passed=points, default=k_medoids_obj.points),
            st_medoids=starting_medoids,
            dist_func=mf.def_var_value_if_none(value_passed=dist_func, default=k_medoids_obj.DistFunc),
            points_dupl=k_medoids_obj.points_no_dupl,
            cand_options=k_medoids_obj.cand_options,
            dists_p_norm=k_medoids_obj.dists_norm_ord,
            max_iter=k_medoids_obj.max_num_of_iter,
            iter_comb=k_medoids_obj.iter_cp_comb,
            mid_point_approx=k_medoids_obj.mid_point_approx,
            break_comb_loop=k_medoids_obj.break_comb_loop,
            n_cl_search_strategy=k_medoids_obj.n_cl_search_strategy
        )

    def create_iter_data_dict_list(self):
        self.iter_data = [
            {
                "centers": self.cp_names_iter[i],
                "labels": self.labels_iter[i],
                "cps_dists": self.cps_dists_matrix_iter[i],
                "config_cost": self.config_cost_iter[i]
            }
            for i in range(len(self.labels_iter))
        ]

    @mf.print_execution_time
    def run_loop(self):
        cps_dists_matrix, total_config_cost = self.calc_dists_and_total_cost_for_cps(
            points=self.points,
            center_points=self.st_medoids,
            dist_func=self.dist_func,
            dists_p_norm=self.dists_p_norm
        )
        self.set_up_configuration(cps_dists_matrix, total_config_cost)

        for i in range(self.max_iter):
            break_loop = self.iteration_config_search_set_up(mid_point_approx=self.mid_point_approx)
            if break_loop:
                break
            print(f"\n\t---::finished {i+1} iteration")

        self.iter_run_count = len(self.labels_iter)
        self.labels = self.labels_iter[-1]
        self.final_config_cost = self.config_cost_iter[-1]
        self.cps_dists_matrix = self.cps_dists_matrix_iter[-1]
        self.res_cps_df = self.points.loc[self.cp_names_iter[-1]]

        self.res_clusters_df = gpd.add_1st_lvl_index_to_df(
            self.points, index_list=self.labels, index_name="centers"
        )
        self.CDMetrics = ClDistMetrics.from_pam_core_obj(self)

        print(f"PAM Line 463\n Iter run count: {self.iter_run_count}")
        print(f"Parallel number of cps to change priority: {self.prl_cp_ch_order}")
        print(f"Parallel number of cps to change history: \n{self.n_cp_change_iter}")
        print(f"configuration cost: {self.final_config_cost}\n\n")

    @mf.print_execution_time
    def set_up_configuration(self, cps_dists_df, total_config_cost):
        self.cps_dists_matrix_iter += [cps_dists_df]
        self.config_cost_iter += [total_config_cost]

        cp_labels_sr = cps_dists_df.idxmin(axis=1)
        self.labels_iter += [cp_labels_sr]
        self.cp_names_iter += [list(cps_dists_df.columns)]

        # if not len(cp_labels_sr.loc[cps_dists_df.columns].unique()) == self.n_cl:
        if not len(cps_dists_df.columns) == self.n_cl:
            print(cps_dists_df.columns)
            print(self.cp_names_iter[-1])
            print(self.cp_names_iter[-2])
            print(len(cp_labels_sr.unique()))
            print(self.n_cl)
            raise Exception("There are less centers than specified")

    @mf.print_execution_time
    def calc_cl_cost_of_cand_from_all_cl_points_new(self):
        lb = self.labels_iter[-1]
        cps = self.cp_names_iter[-1]
        """
        return {
            cp: self.calc_norm_pow_of_dists_of_points_from_cps(
                cl_points_df=self.points.loc[lb == cp],
                cand_cps=self.points_dupl.loc[lb == cp].drop(cp),
                dist_func=self.dist_func,
                dists_p_norm=self.dists_p_norm
            ).sort_values(ascending=True).index.to_list()
            for cp in cps
        }
        """
        cps_cost = []
        for cp in cps:
            cl_bl = lb == cp
            idx_cps = self.calc_norm_pow_of_dists_of_points_from_cps(
                    cl_points_df=self.points.loc[cl_bl],
                    cand_cps=self.points_dupl.loc[cl_bl].drop(cp),
                    dist_func=self.dist_func,
                    dists_p_norm=self.dists_p_norm
                ).sort_values(ascending=True).index.to_list()
            int_idx_cps = list(self.points.index.get_indexer(idx_cps))
            cps_cost += [(idx_cps, int_idx_cps)]

        return dict(zip(cps, cps_cost))

    @mf.print_execution_time
    def find_n_cp_cand_cl_cost_from_im_cps(self):
        # find_n_cand_cl_cost_cps_from_im_cps
        cl_cps_cost = {}
        lb = self.labels_iter[-1]
        cps = self.cp_names_iter[-1]
        n_cand = self.n_cp_cand

        imaginary_mid_points: pd.DataFrame = self.find_imaginary_mid_of_clusters()
        self.imaginary_mid_points_iter += [imaginary_mid_points]

        for (im_cp, row), cp in zip(imaginary_mid_points.iterrows(), cps):
            cl_points_df: pd.DataFrame = self.points.loc[lb == cp]
            cl_points_dupl_df: pd.DataFrame = self.points_dupl.loc[lb == cp]

            dists_from_im: pd.Series = self.dist_func(cl_points_dupl_df, row)

            if n_cand == 1:
                d_idx_min = dists_from_im.idxmin()
                if d_idx_min == cp:
                    idx_cps = []    # To recognize the case of no new candidates for this cluster
                else:
                    idx_cps = [d_idx_min]
            elif n_cand > 1:
                cp_cand_idx = dists_from_im.sort_values(ascending=True).head(n_cand).index
                idx_cps: list = self.calc_norm_pow_of_dists_of_points_from_cps(
                    cl_points_df=cl_points_df,
                    cand_cps=cl_points_df.loc[cp_cand_idx].drop(cp),
                    dist_func=self.dist_func,
                    dists_p_norm=self.dists_p_norm
                ).sort_values(ascending=True).index.to_list()
            else:
                # direct use of distance from imaginary point
                idx_cps = dists_from_im.drop(cp).sort_values(ascending=True).index.to_list()

            cl_cps_cost[cp] = (idx_cps, list(self.points.index.get_indexer(idx_cps)))

        return cl_cps_cost

    def find_imaginary_mid_of_clusters(self):
        np_mid_function = self.im_mid_funcs_dict[self.im_mid_type]

        lb = self.labels_iter[-1]
        cps = self.cp_names_iter[-1]

        return pd.DataFrame(
            [np_mid_function(self.points.loc[lb == cp].values) for cp in cps],
            index=cps,
            columns=self.points.columns
        ) # .rename_axis("Im/ry center of cluster:")

    # NOT USED
    @staticmethod
    def calc_norm_of_dists_of_points_from_cps(cl_points_df, cand_cps, dist_func, dists_p_norm=1):
        return dist_func(cl_points_df, cand_cps).apply(lambda x: np.linalg.norm(x, ord=dists_p_norm))

    @staticmethod
    def calc_norm_pow_of_dists_of_points_from_cps(cl_points_df, cand_cps, dist_func, dists_p_norm=1):
        return dist_func(cl_points_df, cand_cps).pow(dists_p_norm).sum()

    def iteration_config_without_cost_aprox(self):
        cps_dists_matrix, min_config_cost = self.check_new_config_cost_1_cp_no_cand(
            break_loop=self.break_comb_loop,
        )
        return cps_dists_matrix, min_config_cost

    def iteration_config_with_cost_aprox(self):
        if isinstance(self.n_cp_cand, int):
            print("Imaginary center point method enabled")
            cl_cps_cost = self.find_n_cp_cand_cl_cost_from_im_cps()
        else:
            cl_cps_cost = self.calc_cl_cost_of_cand_from_all_cl_points_new()

        cps_dists_matrix, min_config_cost = self.check_new_config_costs_by_group_of_cp_combs(
            cl_cps_cost=cl_cps_cost, n_comb=self.iter_comb
        )
        return cps_dists_matrix, min_config_cost

    @mf.print_execution_time
    def iteration_config_search_set_up(self, mid_point_approx=False):
        if mid_point_approx is True:
            cps_dists_matrix, min_config_cost = self.iteration_config_with_cost_aprox()
        else:
            cps_dists_matrix, min_config_cost = self.iteration_config_without_cost_aprox()

        if cps_dists_matrix is None:
            return True     # Break iterations loop

        self.set_up_configuration(cps_dists_matrix, min_config_cost)

    @mf.print_execution_time
    def check_new_config_costs_by_group_of_cp_combs(self, cl_cps_cost, n_comb=500, random_factor=2):
        min_config_cost = None
        cps_dists_matrix = None

        n_cl_cps = self.prl_cp_ch_order
        break_n_cl_loop = self.break_n_cl_loop
        replace_n_cl = self.fix_n_cl_order

        # n_cl_comb = math.ceil(n_comb / num_of_cl)
        n_cl_comb = math.ceil(n_comb / len(n_cl_cps))

        # comb_sample_size_fix_depth = int(n_cl_comb * random_factor / (3 ** i))
        cps_comb = []
        min_n_cp = -1
        best_config_cost = self.config_cost_iter[-1]
        best_cps_dists_matrix = None
        for i in n_cl_cps:
            if i == 1:
                cps_dists_matrix, min_config_cost = self.check_new_config_cost_for_single_cp_change(
                    cl_cps_cost,
                    break_loop=self.break_comb_loop,
                    all_points=True
                )
                """
                print("pam_line_1037")
                print(cps_dists_matrix)
                print(min_config_cost)
                print(cps_dists_matrix.columns) if cps_dists_matrix is not None else None
                cps_dists_matrix, min_config_cost = self.check_new_config_cost_1_cp_no_cand(
                    break_loop=False,
                )
                print(cps_dists_matrix.columns) if cps_dists_matrix is not None else None
                print(min_config_cost)
                print(cps_dists_matrix)
                """
                """
                print("pam_line_1037")
                print(cps_dists_matrix)
                print(min_config_cost)
                print(cps_dists_matrix.columns)
                cps_dists_matrix, min_config_cost = self.check_new_config_cost_for_single_cp_change_old(
                    cl_cps_cost,
                    sample_size=n_cl_comb,
                    exhaustive_depth=True
                )
                print(cps_dists_matrix.columns)
                print(min_config_cost)
                print(cps_dists_matrix)
                """

            else:
                cps_comb = self.create_n_cl_random_comb(
                    cl_cps_cost=cl_cps_cost,
                    n_cl=i,
                    sample_size=n_cl_comb * random_factor,  # 1 + (num_of_cl * (num_of_cl - i))
                    # comb_sample_size = math.ceil(num_of_cl / i),  # num_of_cl
                )
                cps_comb = mf.random_choice_2d(cps_comb, size=n_cl_comb)

                cps_dists_matrix, min_config_cost = self.check_new_config_cost_of_cps_comb(
                    cps_comb=cps_comb, break_loop=self.break_comb_loop
                )
                """
                print("ckm_line_871")
                print(cps_dists_matrix)
                print(cps_dists_matrix.columns) if cps_dists_matrix is not None else None
                print(min_config_cost)
                cps_dists_matrix, min_config_cost = self.check_new_config_cost_of_cps_comb_old(
                    cps_comb=cps_comb, break_loop=self.break_loop
                )
                print(min_config_cost)
                print(cps_dists_matrix.columns) if cps_dists_matrix is not None else None
                print(cps_dists_matrix)
                """

            if cps_dists_matrix is not None:
                # new_cps = cps_dists_matrix.columns.difference(self.cp_names_iter[-1])
                print(f"Configuration min cost: {min_config_cost}")
                print(f"n_cl_cps == {i}")
                print(f"Iteration: {len(self.labels_iter)}")
                if cps_comb:
                    print(f"Search 'n' CP combinations: {len(cps_comb)}")
                print(f"Previous centers: {self.cp_names_iter[-1]}")
                print(f"New centers found: {cps_dists_matrix.columns}")
                # print(f"new_cps: {list(new_cps)}")
                # print(f"n_cl({self.n_cl}) == new_cps({len(new_cps)}) + fix_cps({self.n_cl - len(new_cps)})")
                if not break_n_cl_loop:
                    if best_config_cost > min_config_cost:
                        best_config_cost = min_config_cost
                        best_cps_dists_matrix = cps_dists_matrix
                        min_n_cp = i
                else:
                    best_config_cost = min_config_cost
                    best_cps_dists_matrix = cps_dists_matrix
                    min_n_cp = i
                    break

        if best_cps_dists_matrix is None:
            return None, min_config_cost

        self.n_cp_change_iter += [min_n_cp]

        if break_n_cl_loop:
            # n_cl_cps.remove(min_n_cp)
            # self.prl_cp_ch_order = [min_n_cp] + n_cl_cps
            poss = n_cl_cps.index(min_n_cp)
            # self.prl_cp_ch_order = n_cl_cps[poss:] + n_cl_cps[:poss] if replace_n_cl else n_cl_cps[poss:]
            self.prl_cp_ch_order = n_cl_cps if replace_n_cl else n_cl_cps[poss:]
            print("pam_line_1043")
            print(self.prl_cp_ch_order)

        print(f"Iter {len(self.labels_iter)} configuration cost: {best_config_cost}")
        return best_cps_dists_matrix, best_config_cost

    # NOT USED
    @mf.print_execution_time
    def create_single_cp_change_comb_min_depth_first(self, cl_cps_cost, cp_names, depth, depth_min=0):
        if depth is True:
            depth = self.num_of_points

        mix_cps = []
        fixed_cps = [cp_names[:cp] + cp_names[cp + 1:] for cp in range(len(cp_names))]
        rem_cps_int_poss = list(range(self.n_cl))
        for d in range(depth_min, depth):
            cp_order = np.random.choice(rem_cps_int_poss, size=len(rem_cps_int_poss), replace=False)
            for p in cp_order:
                cand_cps = cl_cps_cost[cp_names[p]][0]
                if len(cand_cps) <= d:
                    rem_cps_int_poss.remove(p)
                    continue

                mix_cps += [[cand_cps[d]] + [fixed_cps[p]]]

            if not rem_cps_int_poss:
                break

        return mix_cps

    # NOT USED
    @staticmethod
    def create_single_cp_change_comb(cl_cps_cost, depth):
        # creates len(cl_cps_cost) X depth combinations
        cp_names = list(cl_cps_cost.keys())
        """
        all_comb_one = []
        for cp in cl_cps_cost.keys():
            all_comb_one += cls.create_cps_combinations(
                cps_to_change=[cp],
                cl_cps_cost=cl_cps_cost,
                depth=mf.def_var_value_if_none(depth, def_func=lambda: math.ceil(5 + 20 / len(cl_cps_cost.keys())))
            )
        return all_comb_one
        """
        """
        mix_cps = []
        # cl_change_poss = [0]
        for cp in range(len(cp_names)):
            # fixed_cps = list(filter(lambda x: x not in [cp], cp_names))
            fixed_cps = cp_names[:cp] + cp_names[cp + 1:]
            new_cps = cl_cps_cost[cp_names[cp]][:depth]
            mix_cps += mf.all_lists_elements_combinations(new_cps, [[fixed_cps]])
            # cl_change_poss += [cl_change_poss[-1] + len(new_cps)]
        """
        mix_cps = []
        for cp in range(len(cp_names)):
            fixed_cps = cp_names[:cp] + cp_names[cp + 1:]
            new_cps = cl_cps_cost[cp_names[cp]][0][:depth]
            mix_cps += [[nc] + [fixed_cps] for nc in new_cps]
        return mix_cps

    # NOT USED
    @mf.print_execution_time
    def check_new_config_cost_for_single_cp_change_old(self, cl_cps_cost, sample_size, exhaustive_depth=None):
        # The sample size is the max number of samples checked in random order
        # The exhaustive depth is the max number of samples per cluster

        depth = math.ceil(sample_size / len(cl_cps_cost.keys()))  # n_comb = 2000, sample_size = 200, depth = 10

        mix_cps = mf.random_choice_2d(self.create_single_cp_change_comb(cl_cps_cost, depth=depth), size=sample_size)

        if exhaustive_depth:
            mix_cps += self.create_single_cp_change_comb_min_depth_first(
                cl_cps_cost, list(cl_cps_cost.keys()), depth=exhaustive_depth, depth_min=depth
            )

        min_config_cost = self.config_cost_iter[-1]
        cps_dists_matrix = None
        print(f"...{len(mix_cps)} comb checks started")
        for i, idx_list in enumerate(mix_cps):
            if isinstance(idx_list[-1], list):
                idx_list = idx_list[:-1] + idx_list[-1]

            cps_dists, total_cost = self.calc_dists_and_total_cost_for_cps(
                dist_func=self.dist_func,
                points=self.points,
                center_points=self.points.loc[idx_list],
                dists_p_norm=self.dists_p_norm
            )
            if total_cost < min_config_cost:
                # print(f"Configuration min cost: {total_cost}")
                min_config_cost = total_cost
                cps_dists_matrix = cps_dists
                print(f"Comb checks completed: {i}")
                break

        return cps_dists_matrix, min_config_cost

    @staticmethod
    def calc_2_nearest_dists_from(cps_dists_matrix, np_arrays=True):
        cdm = cps_dists_matrix
        n_cols = len(cdm[0])
        n_rows = len(cdm)
        d0_near = np.empty(shape=n_rows, dtype="float32")
        d1_near = np.empty(shape=n_rows, dtype="float32")
        for i in range(n_rows):
            v1 = cdm[i, 0]
            v2 = cdm[i, 1]
            if v1 < v2:
                min0 = v1
                min1 = v2
            else:
                min0 = v2
                min1 = v1
            for j in range(2, n_cols):
                val = cdm[i, j]
                if val < min0:
                    min1 = min0
                    min0 = val
                elif cdm[i, j] < min1:
                    min1 = val
            d0_near[i] = min0
            d1_near[i] = min1

        return (d0_near, d1_near) if np_arrays else (list(d0_near), list(d1_near))

    def rtn_all_fixed_cps_single_case(self, cp_names: list, int_poss=True):
        cp_int_idx = list(self.points.index.get_indexer(cp_names)) if int_poss else cp_names
        return [cp_int_idx[:cp] + cp_int_idx[cp + 1:] for cp in range(len(cp_int_idx))]

    @staticmethod
    @njit
    def calc_delta_of_cost(new_cps_min_cost, not_in_cl, d0_near, d1_near):
        delta_cost = 0
        for i in range(len(new_cps_min_cost)):
            if not_in_cl[i]:
                if new_cps_min_cost[i] < d0_near[i]:
                    delta_cost += new_cps_min_cost[i] - d0_near[i]
            else:
                if new_cps_min_cost[i] < d1_near[i]:
                    delta_cost += new_cps_min_cost[i] - d0_near[i]
                else:
                    delta_cost += d1_near[i] - d0_near[i]

        return delta_cost

    def rtn_config_res_for(self, min_swap, min_delta_cost, fixed_cps, counter):
        min_cps = fixed_cps[min_swap[0]] + [min_swap[1]]
        min_config_cost = self.config_cost_iter[-1] + min_delta_cost
        print(f"Single cp swaps checked: {counter}")
        print(f"Min configuration cost: {min_config_cost}")
        return self.dist_func().iloc[:, min_cps].abs(), min_config_cost

    @mf.print_execution_time
    def check_new_config_cost_for_single_cp_change(
            self, cl_cps_cost, sample_size=None, break_loop=True, all_points=True
    ):
        dists_matrix = self.dist_func().values
        labels: np.array = self.labels_iter[-1].values
        n_cl = self.n_cl
        cp_names = list(cl_cps_cost.keys())

        # sample_size >= depth * number_of_clusters
        depth = self.num_of_points if sample_size is None else sample_size // len(cp_names)

        d0_near, d1_near = self.calc_2_nearest_dists_from(self.cps_dists_matrix_iter[-1].values)

        fixed_cps = self.rtn_all_fixed_cps_single_case(cp_names, int_poss=True)

        cand_cps_list = [list(cl_cps_cost[cp_names[p]][1]) for p in range(len(cp_names))]

        if all_points is True:
            for p in range(n_cl):
                other_points = []
                for i in list(range(p))+list(range(p+1, n_cl)):
                    other_points += list(cl_cps_cost[cp_names[i]][1])
                cand_cps_list[p] += other_points

        not_in_cl_list = [labels != cp_names[p] for p in range(n_cl)]

        d_tol = -0.000001
        counter = 0
        min_delta_cost = d_tol
        min_swap = None
        rem_cps_int_poss = list(range(n_cl))
        for d in range(depth):
            # cps_to_drop = []
            # for cp in range(len(cp_names)):
            n = len(rem_cps_int_poss)
            cp_order = np.random.choice(rem_cps_int_poss, size=n, replace=False) if break_loop else rem_cps_int_poss
            for p in cp_order:
                cand_cps = cand_cps_list[p]
                if not len(cand_cps) > d:
                    rem_cps_int_poss.remove(p)
                    continue

                new_cp = cand_cps[d]
                new_cps_min_cost = dists_matrix[new_cp]

                not_in_cl = not_in_cl_list[p]
                delta_cost = self.calc_delta_of_cost(new_cps_min_cost, not_in_cl, d0_near, d1_near)

                counter += 1

                if delta_cost < min_delta_cost:
                    """
                    cps_dists_matrix = self.dist_func()[fixed_cps[cp]+[new_cp]]
                    total_config_cost = np.sum(cps_dists_matrix.abs().min(axis=1) ** self.dists_p_norm)
                    print("pam_line_1263")
                    print(new_cp)
                    print(fixed_cps)
                    print(fixed_cps[cp]+[new_cp])
                    print(total_config_cost)
                    print(min_config_cost + delta_cost)
                    """
                    # min_cps = fixed_cps[cp] + [new_cp]  # int indexes
                    min_swap = (p, new_cp)     # The 1st is poss to fixed_cps, the 2nd is the real new poss
                    min_delta_cost = delta_cost
                    if break_loop:
                        """
                        min_cps = fixed_cps[min_swap[0]] + [min_swap[1]]
                        min_config_cost = self.config_cost_iter[-1] + min_delta_cost
                        print(f"Single cp swaps checked: {counter}")
                        print(f"Min configuration cost: {min_config_cost}")
                        return self.dist_func().iloc[:, min_cps].abs(), self.config_cost_iter[-1] + min_delta_cost
                        """
                        return self.rtn_config_res_for(min_swap, min_delta_cost, fixed_cps, counter)

            """
            if cps_to_drop:
                for i, cp in enumerate(cps_to_drop):
                    cp_names.pop(cp - i)
                    fixed_cps.pop(cp - i)
            
            if not cp_names:
                break
            """
            if not rem_cps_int_poss:
                break

        if min_swap is None:
            print(f"Single cp swaps checked: {counter}")
            print(f"No configuration delta greater than: {d_tol}")
            return None, self.config_cost_iter[-1]

        return self.rtn_config_res_for(min_swap, min_delta_cost, fixed_cps, counter)

    @mf.print_execution_time
    def check_new_config_cost_1_cp_no_cand(self, break_loop=True):
        dists_matrix = self.dist_func().values
        labels: np.array = self.labels_iter[-1].values
        cp_names = list(self.cps_dists_matrix_iter[-1].columns)

        d0_near, d1_near = self.calc_2_nearest_dists_from(self.cps_dists_matrix_iter[-1].values)

        print("pam_line_1255")
        print(d0_near)
        print(d1_near)
        print(np.min(self.cps_dists_matrix_iter[-1].values, axis=1))

        fixed_cps = self.rtn_all_fixed_cps_single_case(cp_names, int_poss=True)
        # not_in_cl_list = [labels != cp_names[p] for p in range(len(cp_names))]

        def order_if_break_loop(n):
            return np.random.choice(n, size=n, replace=False) if break_loop else range(n)

        counter = 0
        d_tol = -0.000001
        min_delta_cost = d_tol
        min_swap = None
        for p in order_if_break_loop(len(cp_names)):
            for new_cp in order_if_break_loop(self.num_of_points):
                new_cps_min_cost = dists_matrix[new_cp]

                not_in_cl = labels != cp_names[p]
                delta_cost = self.calc_delta_of_cost(new_cps_min_cost, not_in_cl, d0_near, d1_near)

                counter += 1

                if delta_cost < min_delta_cost:
                    """
                    cps_dists_matrix = self.dist_func()[fixed_cps[cp]+[new_cp]]
                    total_config_cost = np.sum(cps_dists_matrix.abs().min(axis=1) ** self.dists_p_norm)
                    print("pam_line_1263")
                    print(new_cp)
                    print(fixed_cps)
                    print(fixed_cps[cp]+[new_cp])
                    print(total_config_cost)
                    print(min_config_cost + delta_cost)
                    """
                    # min_cps = fixed_cps[cp] + [new_cp]  # int indexes
                    min_swap = (p, new_cp)  # The 1st is poss to fixed_cps, the 2nd is the real new poss
                    min_delta_cost = delta_cost + d_tol
                    if break_loop:
                        """
                        min_cps = fixed_cps[min_swap[0]] + [min_swap[1]]
                        min_config_cost = self.config_cost_iter[-1] + min_delta_cost
                        print(f"Single cp swaps checked: {counter}")
                        print(f"Min configuration cost: {min_config_cost}")
                        return self.dist_func().iloc[:, min_cps].abs(), self.config_cost_iter[-1] + min_delta_cost
                        """
                        return self.rtn_config_res_for(min_swap, min_delta_cost, fixed_cps, counter)

        if min_swap is None:
            return None, self.config_cost_iter[-1]
        """
        min_cps = fixed_cps[min_swap[0]] + [min_swap[1]]
        min_config_cost = self.config_cost_iter[-1] + min_delta_cost
        print(f"Single cp swaps checked: {counter}")
        print(f"Min configuration cost: {min_config_cost}")
        return self.dist_func().iloc[:, min_cps].abs(), self.config_cost_iter[-1] + min_delta_cost
        """
        return self.rtn_config_res_for(min_swap, min_delta_cost, fixed_cps, counter)

    @staticmethod
    def create_cps_combinations(
            cps_to_change: list[str | int],
            cl_cps_cost: dict[int | str | tuple[str], pd.DataFrame],
            depth: int
    ) -> list[list[int | str | tuple[str]]]:

        if len(cps_to_change) == 0:
            return [list(cl_cps_cost.keys())]
        elif len(cps_to_change) > len(cl_cps_cost):
            raise Exception("cps_to_change can not be more than the number of clusters")

        def new_cps_creation():
            return mf.all_lists_elements_combinations(*[
                # list(cl_cps_cost[cp].index)[:depth] for cp in cps_to_change
                cl_cps_cost[cp][0][:depth] if cl_cps_cost[cp] else [cp] for cp in cps_to_change
            ])

        if len(cps_to_change) == len(cl_cps_cost):
            return new_cps_creation()

        fixed_cps = list(filter(lambda x: x not in cps_to_change, cl_cps_cost.keys()))
        if len(cps_to_change) == 1:
            cp = cps_to_change[0]
            new_cps = cl_cps_cost[cp][0][:depth] if cl_cps_cost[cp] else cps_to_change
            return mf.all_lists_elements_combinations(new_cps, [[fixed_cps]])

        new_cps = new_cps_creation()
        mix_cps = mf.all_lists_elements_combinations(new_cps, [[fixed_cps]])
        """
        print("pam_line_1052")
        print(cl_cps_cost)
        print(cps_to_change)
        print(new_cps)
        print(fixed_cps)
        print(mix_cps)
        """
        return mix_cps

    @classmethod
    @mf.print_execution_time
    def create_n_cl_random_comb(cls, cl_cps_cost, n_cl: int, sample_size: int):
        # creates n_cl X len(cl_cps_cost) X depth combinations

        cp_names = cl_cps_cost.keys()

        # The bigger the comb_sample_size the smaller the depth
        # sample_size = comb_sample_size * depth ^ n_cl
        # comb_sample_size = math.ceil(sample_size / samples_per_comb)
        # samples_per_comb = depth ^ n_cl
        depth = -1 + mf.return_min_num_of_pow_greater_than_min_val(power=n_cl, min_val=sample_size)

        if len(cp_names) == n_cl:
            return cls.create_cps_combinations(
                cps_to_change=list(cp_names),
                cl_cps_cost=cl_cps_cost,
                depth=depth
            )
        elif n_cl == 1:
            mix_cps = []
            cp_names = list(cl_cps_cost.keys())
            depth = math.ceil(sample_size / len(cp_names))
            for cp in range(len(cp_names)):
                # fixed_cps = list(filter(lambda x: x not in [cp], cp_names))
                fixed_cps = cp_names[:cp] + cp_names[cp+1:]
                new_cps = cl_cps_cost[cp_names[cp]][0][:depth]
                mix_cps += mf.all_lists_elements_combinations(new_cps, [[fixed_cps]])
            """
            return [
                mf.all_lists_elements_combinations(
                    cl_cps_cost[cp_names[cp]][:depth], 
                    [cp_names[:cp] + cp_names[cp+1:]]
                )
                for cp in range(len(cp_names))
            ]"""
            return mix_cps[:sample_size]
        elif n_cl == 0:
            raise Exception("'n_cl' must be greater than 0")

        cps_comb = mf.random_choice_2d(
            list(combinations(cp_names, n_cl)),
            size=sample_size
        )
        if depth == 1:
            all_comb_depth = []
            for cps in cps_comb[:sample_size]:
                new_cps = [cl_cps_cost[cp][0][0] if cl_cps_cost[cp][0] else cp for cp in cps]
                fixed_cps = list(filter(lambda x: x not in cps, cl_cps_cost.keys()))
                # all_comb_depth += mf.all_lists_elements_combinations([new_cps], [[fixed_cps]])
                all_comb_depth += [new_cps + [fixed_cps]]
            """
            all_comb_depth = []
            for cps in cps_comb[:sample_size]:
                all_comb_depth += [
                    cls.create_cps_combinations(
                        cps_to_change=cps,
                        cl_cps_cost=cl_cps_cost,
                        depth=1
                    )
                ]
            """
            print("ckm_line_1345")
            print(f"n_cl: {n_cl} _")
            print(f"depth: {depth} _")
            print(f"sample_size({sample_size})")
            print(f"Number of combination generated (all_comb_depth): {len(all_comb_depth)}")
            return mf.random_choice_2d(all_comb_depth, size=sample_size)

        all_comb_n = []
        poss = 0
        for i in range(depth, 0, -1):
            comb_sample_size = math.ceil(sample_size / (i ** n_cl))
            comb_i = []
            for cps in cps_comb[poss:poss+comb_sample_size]:
                comb_i += cls.create_cps_combinations(
                    cps_to_change=cps,
                    cl_cps_cost=cl_cps_cost,
                    depth=i
                )
            all_comb_n += comb_i
            print("ckm_line_1363")
            print(f"n_cl: {n_cl} _")
            print(f"depth: {depth} _")
            print(f"sample_size({sample_size}) <=~ comb_sample_size*depth^n_cl({comb_sample_size * i ** n_cl})")
            print(f"comb_sample_size({comb_sample_size}) >=~ sample_size / depth^n_cl({sample_size / (i ** n_cl)})")
            print(f"len(comb_i): {len(comb_i)}")
            print(f"current depth (i): {i}, samples_per_comb (i^n_cl): {i ** n_cl}")
        print(f"len(all_comb_n) == {len(all_comb_n)}")
        return mf.random_choice_2d(all_comb_n, size=sample_size)

    @staticmethod
    def calc_dists_and_total_cost_for_cps(
            dist_func, points: pd.DataFrame, center_points: pd.DataFrame, dists_p_norm
    ) -> tuple[pd.DataFrame, float]:

        dists_from_cps_matrix = dist_func(points, center_points).abs()
        total_config_cost = np.sum(dists_from_cps_matrix.min(axis=1) ** dists_p_norm)

        return dists_from_cps_matrix, total_config_cost

    @mf.print_execution_time
    def check_new_config_cost_of_cps_comb_old(self, cps_comb: list):
        min_config_cost = self.config_cost_iter[-1]
        cps_dists_matrix = None
        print(f"...{len(cps_comb)} comb checks started")
        for idx_list in cps_comb:
            if isinstance(idx_list[-1], list):
                idx_list = idx_list[:-1] + idx_list[-1]

            cps_dists, total_cost = self.calc_dists_and_total_cost_for_cps(
                dist_func=self.dist_func,
                points=self.points,
                center_points=self.points.loc[idx_list],
                dists_p_norm=self.dists_p_norm
            )
            if total_cost < min_config_cost:
                # print(f"Configuration min cost: {total_cost}")
                min_config_cost = total_cost
                cps_dists_matrix = cps_dists
                break

        return cps_dists_matrix, min_config_cost

    @mf.print_execution_time
    def check_new_config_cost_of_cps_comb(self, cps_comb: list, break_loop=True):
        min_config_cost = self.config_cost_iter[-1]
        cps_dists_matrix_t = self.cps_dists_matrix_iter[-1].values.T
        cps_idx_list = list(self.cps_dists_matrix_iter[-1].columns)
        dists_matrix = self.dist_func().values
        print(f"...{len(cps_comb)} comb checks started")

        min_poss = -1
        for i in range(len(cps_comb)):
            idx_list = cps_comb[i]
            if isinstance(idx_list[-1], list):
                new_cps = idx_list[:-1]
                fixed_cps = idx_list[-1]
            else:
                new_cps = idx_list
                fixed_cps = []

            if isinstance(new_cps[0], int):
                new_cps_int = new_cps
                fixed_cps_int = fixed_cps
            else:
                # new_cps_int = [mf.return_int_number_of_string(i) for i in new_cps]
                new_cps_int = self.points.index.get_indexer(new_cps)
                # fixed_cps_int = [mf.return_int_number_of_string(i) for i in fixed_cps] if fixed_cps else []
                fixed_cps_int = self.points.index.get_indexer(fixed_cps) if fixed_cps else []

            new_cps_min_cost = dists_matrix[new_cps_int]

            """
            print("pam_line_1627")
            print(fixed_cps_int)
            print(cps_idx_list)
            print(self.cps_dists_matrix_iter[-1])
            print([i for i in range(len(cps_idx_list)) if cps_idx_list[i] in fixed_cps_int])
            """

            if fixed_cps_int:
                mix_cps_dist_matrix = np.vstack((
                    cps_dists_matrix_t[[i for i in range(len(cps_idx_list)) if cps_idx_list[i] in fixed_cps_int]],
                    new_cps_min_cost
                ))
            else:
                mix_cps_dist_matrix = new_cps_min_cost

            total_cost = np.sum(np.min(np.abs(mix_cps_dist_matrix), axis=0) ** self.dists_p_norm)

            if min_config_cost > total_cost:
                min_poss = i
                # print(f"Configuration min cost: {total_cost}")
                # min_config_cost = min_config_cost + cost_dif
                # min_config_cps_dist = min_cp_dists_sr
                if break_loop:
                    break

        if min_poss == -1:
            return None, min_config_cost

        idx_list = cps_comb[min_poss]
        if isinstance(idx_list[-1], list):
            idx_list = idx_list[:-1] + idx_list[-1]

        return self.calc_dists_and_total_cost_for_cps(
            dist_func=self.dist_func,
            points=self.points,
            center_points=self.points.loc[idx_list],
            dists_p_norm=self.dists_p_norm
        )

    # NOT USED
    @staticmethod
    @njit
    def create_not_in_ch_cl(labels, fixed_cps):
        not_in_ch_cl = np.array([False] * len(labels))
        for j in range(len(labels)):
            if labels[j] in fixed_cps:
                not_in_ch_cl[j] = True
        return not_in_ch_cl

    # NOT USED
    @staticmethod
    @njit
    def calc_d1_near(labels, dists_matrix, fixed_cps, fixed_cps_int):
        d1_near = np.array([np.inf] * len(labels))
        """
        # fixed_cps_dists_matrix = [dists_matrix[i] for i in fixed_cps_int] 
        for j in range(len(labels)):
            if labels[j] not in fixed_cps:
                d1_near[j] = np.min(fixed_cps_dists_matrix[:, j])
        """
        for i in range(len(labels)):
            if labels[i] not in fixed_cps:
                min_val = dists_matrix[fixed_cps_int[0]][i]
                for j in range(1, len(fixed_cps_int)):
                    val = dists_matrix[fixed_cps_int[j]][i]
                    if min_val > val:
                        min_val = val
                d1_near[i] = min_val
        return d1_near

    # NOT USED
    @mf.print_execution_time
    def check_new_config_cost_of_cps_comb_new_fail(self, cps_comb: list, break_loop=True):
        min_config_cost = self.config_cost_iter[-1]
        # cps_idx_list = list(self.cps_dists_matrix_iter[-1].columns)
        dists_matrix = self.dist_func().values
        labels = self.labels_iter[-1].values
        print(f"...{len(cps_comb)} comb checks started")

        # d0_near, d1_near = self.calc_2_nearest_dists_from(self.cps_dists_matrix_iter[-1].values)
        d0_near = np.min(self.cps_dists_matrix_iter[-1].values, axis=1)

        min_poss = -1
        d_tol = -0.000001
        counter = 0
        min_delta_cost = d_tol
        for i in range(len(cps_comb)):
            counter += 1
            idx_list = cps_comb[i]
            if isinstance(idx_list[-1], list):
                new_cps = idx_list[:-1]
                fixed_cps = idx_list[-1]
            else:
                new_cps = idx_list
                fixed_cps = []

            if isinstance(new_cps[0], int):
                new_cps_int = new_cps
                fixed_cps_int = fixed_cps
            else:
                # new_cps_int = [mf.return_int_number_of_string(i) for i in new_cps]
                new_cps_int = self.points.index.get_indexer(new_cps)
                # fixed_cps_int = [mf.return_int_number_of_string(i) for i in fixed_cps] if fixed_cps else []
                fixed_cps_int = self.points.index.get_indexer(fixed_cps) if fixed_cps else []

            new_cps_dists_matrix = dists_matrix[new_cps_int]

            """
            print("pam_line_1627")
            print(fixed_cps_int)
            print(cps_idx_list)
            print(self.cps_dists_matrix_iter[-1])
            print([i for i in range(len(cps_idx_list)) if cps_idx_list[i] in fixed_cps_int])
            """

            new_cps_min_cost = np.min(new_cps_dists_matrix, axis=0)

            if fixed_cps:
                not_in_ch_cl = self.create_not_in_ch_cl(labels, fixed_cps)
                d1_near = dists_matrix[fixed_cps_int][0] if len(fixed_cps_int) == 1 else self.calc_d1_near(
                    labels, dists_matrix, fixed_cps, fixed_cps_int
                )
            else:
                not_in_ch_cl = np.array([False] * len(labels))
                d1_near = np.array([np.inf] * len(labels))

            delta_cost = self.calc_delta_of_cost(new_cps_min_cost, not_in_ch_cl, d0_near, d1_near)

            if delta_cost < min_delta_cost:
                min_delta_cost = delta_cost
                min_poss = i
                # min_config_cost = min_config_cost + cost_dif
                # min_config_cps_dist = min_cp_dists_sr
                if break_loop:
                    break

        print(f"ckm_line_1546\nNumber of combination checked: {counter}")
        print(f"min delta cost: {min_delta_cost}")
        print(f"Conf min cost: {min_config_cost + min_delta_cost}")

        if min_poss == -1:
            return None, min_config_cost

        idx_list = cps_comb[min_poss]
        if isinstance(idx_list[-1], list):
            idx_list = idx_list[:-1] + idx_list[-1]

        return self.calc_dists_and_total_cost_for_cps(
            dist_func=self.dist_func,
            points=self.points,
            center_points=self.points.loc[idx_list],
            dists_p_norm=self.dists_p_norm
        )


@mf.print_execution_time
def k_medoids_range(set_up_k_medoids: Callable, min_n_cl: int, max_n_cl: int, ascending=True):
    if ascending:
        c_kmedoids_dict = {min_n_cl: set_up_k_medoids(min_n_cl)}
        for n_cl in range(min_n_cl + 1, max_n_cl + 1):
            st_p = c_kmedoids_dict[n_cl - 1].medoids.index
            c_kmedoids_dict[n_cl] = set_up_k_medoids(n_cl, starting_points=st_p)
    else:
        # c_kmedoids_dict = {max_n_cl: set_up_k_medoids(max_n_cl)}
        res_list = [0] * (max_n_cl - min_n_cl + 1)
        res_list[max_n_cl - min_n_cl] = set_up_k_medoids(max_n_cl)
        for n_cl in range(max_n_cl - 1, min_n_cl - 1, -1):
            st_p = res_list[n_cl - min_n_cl + 1].medoids.index
            res_list[n_cl - min_n_cl] = set_up_k_medoids(n_cl, starting_points=st_p)
        c_kmedoids_dict = dict(zip(list(range(min_n_cl, max_n_cl + 1)), res_list))
    return c_kmedoids_dict


"""
    @classmethod
    @mf.print_execution_time
    def create_n_cl_random_comb_old(cls, cl_cps_cost, n_cl: int, sample_size: int, comb_sample_size=None):
        # creates n_cl X len(cl_cps_cost) X depth combinations

        if comb_sample_size is None:
            comb_sample_size = len(cl_cps_cost)
        elif comb_sample_size <= 0:
            comb_sample_size = 1

        cp_names = cl_cps_cost.keys()

        # The bigger the comb_sample_size the smaller the depth
        # sample_size = comb_sample_size * depth ^ n_cl
        # comb_sample_size = math.ceil(sample_size / samples_per_comb)
        # samples_per_comb = depth ^ n_cl
        samples_per_comb = sample_size if len(cp_names) == n_cl else math.ceil(sample_size / comb_sample_size)
        depth = -1 + mf.return_min_num_of_pow_greater_than_min_val(power=n_cl, min_val=samples_per_comb)

        if len(cp_names) == n_cl:
            return cls.create_cps_combinations(
                cps_to_change=list(cp_names),
                cl_cps_cost=cl_cps_cost,
                depth=depth
            )
        elif n_cl == 1:
            return cls.create_all_one_cp_change_comb(
                cl_cps_cost, depth=math.ceil(sample_size / len(cl_cps_cost))
            )[:sample_size]
        # elif len(cl_cps_cost.keys()) == n_cl - 1:
            # cps_comb = list(combinations(cl_cps_cost.keys(), n_cl))

        cps_comb = mf.random_choice_2d(
            list(combinations(cp_names, n_cl)),
            size=math.ceil(sample_size / (depth ** n_cl))   # comb_sample_size
        )
        """"""
        print("pam_line_1716")
        print(f"clusters({len(cl_cps_cost)}) - n_cl({n_cl})")
        print(f"cps_comb({math.ceil(sample_size / (depth ** n_cl))}) >= comb_sample_size({comb_sample_size})")
        print(f"comb_sample_size({comb_sample_size}) * depth({depth}) ^ n_cl({n_cl}) == sample_size({sample_size})")
        print(f"cps_comp({math.ceil(sample_size / (depth ** n_cl))}) >= real cps_comb({len(cps_comb)}) >= ")
        print(f"real cps_comb({len(cps_comb)}) >= combinations({len(list(combinations(cl_cps_cost.keys(), n_cl)))})")
        print(f"{math.ceil(sample_size / (depth ** n_cl)) * (depth ** n_cl)} >= {sample_size}")
        """"""
        print("pam_line_1190")
        print(f"cps combinations that will change: {len(cps_comb)}")
        print(f"number of cps in combinations: {len(cps_comb[0])} == {n_cl}")
        print(f"Depth of new cps comb in changed clusters: {depth}")
        print(f"Max number of comb: {(depth ** n_cl) * len(cps_comb)} >~= {sample_size}")

        all_comb_n = [
            cls.create_cps_combinations(
                cps_to_change=cps,
                cl_cps_cost=cl_cps_cost,
                depth=depth
            )[0]
            for cps in cps_comb
        ]

        return mf.random_choice_2d(all_comb_n, size=sample_size)

    @staticmethod
    @njit(fastmath=True)
    def check_config_cost_of_cp_combs(
            cps_comb: np.ndarray,
            dist_matrix: np.ndarray,
            min_config_cost: float,
            dists_p_norm: int
    ):
        min_config_cost = min_config_cost
        poss = -1
        for k in np.arange(len(cps_comb)):
            # center_points = points.loc[idx_list]
            # centers_idx = np.array([int(cp[3:])-1 for cp in cps_comb[i]])
            centers_idx = cps_comb[k]
            comb = cps_comb[k]
            if not isinstance(comb[0], int):
                for cp in range(len(comb)):
                    cp_name = comb[cp]
                    cp_int = ""
                    str_poss = 0
                    for i in cp_name:
                        if isinstance(i, int):
                            break
                        str_poss += 1
                    for i in range(len(cp_name)-str_poss):
                        cp_int += cp_name[i]
                    centers_idx[cp] = int(cp_int) - 1

            # dist_from_cps_point_df = dist_func(points, center_points).abs()
            # dist_from_cps_point_df = abs(dist_matrix[centers_idx])
            # total_config_cost = np.sum(dist_from_cps_point_df.min(axis=1) ** dists_p_norm)
            total_config_cost = 0
            for i in prange(len(dist_matrix)):
                dist_array = dist_matrix[i]
                # row = [dist_array[centers_idx[j]] for j in range(len(centers_idx))]
                row = dist_array[centers_idx]

                row_min = row[0]
                for j in np.arange(1, len(row)):
                    if row_min > row[j]:
                        row_min = row[j]
                total_config_cost += row_min ** dists_p_norm

            if total_config_cost < min_config_cost:
                min_config_cost = total_config_cost
                poss = k
                return poss, min_config_cost

        if poss == -1:
            return -1, min_config_cost
        else:
            return poss, min_config_cost

    @mf.print_execution_time
    def check_new_config_cost_of_cps_comb_njit(self, cps_comb: list):
        print(f"...{len(cps_comb)} comb checks started")
        # comb: int | np.ndarray
        poss, min_config_cost = self.check_config_cost_of_cp_combs(
            cps_comb=np.array(cps_comb),
            dist_matrix=self.dist_func().values,
            min_config_cost=self.config_cost_iter[-1],
            dists_p_norm=self.dists_p_norm
        )
        if poss == -1:
            return None, min_config_cost

        min_config_cps_dist = self.dist_func()[cps_comb[poss]].abs()
        return min_config_cps_dist, min_config_cost
"""
