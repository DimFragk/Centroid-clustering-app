import math
from time import time, process_time

import pandas as pd
import numpy as np
from itertools import combinations

from scipy.spatial.distance import cdist
from scipy.spatial.distance import euclidean
from scipy.spatial import ConvexHull

from utils import general_functions as mf
import utils.math_functions as gmf
import utils.pandas_functions as gpd
import utils.data_processing as gdp
# import utils.visualization_functions as gvp

if __name__ == "__main__":
    from centroid_clustering.clustering_metrics import DistFunction, ClDistMetrics
else:
    from .clustering_metrics import DistFunction, ClDistMetrics


class Kmedoids:
    def __init__(
            self, data, n_clusters: int,
            n_init: int = 5,
            dim_redux: int = None,
            st_p_method: str = None,
            starting_points: list[int, str, tuple[int, str]] = None,
            im_st_points: np.ndarray | pd.DataFrame | pd.Series | list = None,
            dist_func: DistFunction = None,
            dists_p_norm: int = 1,
            cand_options: dict[str, int | bool] = None,
            max_iter: int = None,
            iter_cp_comb: int = None):

        self.data = data
        self.num_of_clusters = n_clusters
        self.dim_redux = dim_redux

        self.n_init = n_init
        self.starting_points = starting_points
        self.im_st_points = im_st_points
        self.st_p_method = mf.def_var_value_if_none(st_p_method, default="convex_hull")

        self.dists_norm_ord = dists_p_norm
        self.cand_options = cand_options

        if self.cand_options is not None:
            self.cand_options["n_cp_cand"] = mf.def_var_value_if_none(cand_options["n_cp_cand"], default=5)
            self.cand_options["g_median"] = mf.def_var_value_if_none(cand_options["g_median"], default=False)
            self.cand_options["p_mean_ord"] = mf.def_var_value_if_none(cand_options["p_mean_ord"], default=1)

        self.num_of_dimensions = len(self.data.columns)
        self.num_of_points = len(self.data.index)

        self.max_num_of_iter = mf.def_var_value_if_none(value_passed=max_iter, default=50)
        self.iter_cp_comb = mf.def_var_value_if_none(iter_cp_comb, default=1000)

        self.m_idx_merge = False
        self.merge_sep_str = " |-| "
        self.idx_name = self.data.index.name
        self.points = self.prepare_points_df_from_data(dim_redux=self.dim_redux)
        self.points_no_dupl = self.points.drop_duplicates()

        if dist_func is None:
            self.DistFunc = DistFunction(
                dist_metric="euclidean",
                norm_order=2,
                vector_dist_func=None,
                cache_points=self.points
            )
        elif isinstance(dist_func, DistFunction):
            dist_func.check_if_all_dists_are_cached()

            empty_cache_check = dist_func.num_of_nan_in_cache() >= len(self.points.index)*len(self.points.columns)

            if empty_cache_check and not dist_func.check_cache_compatibility(self.points):
                print("pam_line_723")
                print("Creating new cache")
                dist_func.df_to_cache(df=self.points)

            self.DistFunc = dist_func

        self.dist_metric = self.DistFunc.dist_metric
        self.norm_order = self.DistFunc.norm_order
        self.dist_func_1d = self.DistFunc.dist_func_0d
        self.dist_func = self.DistFunc.dist_func
        self.dist_func_cache = self.DistFunc.dist_func_cache

        # self.cp_cand_num = self.choose_max_num_of_medoid_candidates(n_cp_cand=cp_cand_num)

        self.idx_min_dist_to_im_st_p = pd.Index([])
        self.candidate_st_p = self.select_candidate_starting_points(method=self.st_p_method)
        self.st_p_df_list = self.select_n_sets_of_unique_starting_points(n_sets=self.n_init)
        self.starting_points_df = pd.DataFrame()

        self.PamCore: PamCore | None = None
        self.cost_metrics = pd.DataFrame()

        self.select_best_pam_of_st_points()
        # self.test_dists_cache_method()

        self.labels = pd.Series()
        self.medoids = pd.DataFrame()
        self.data_df = pd.DataFrame()
        self.clusters_df = pd.DataFrame()
        self.clusters_dict = {}
        self.inertia: float | None = None

        self.set_up_res_from_pam_core(pam_core=self.PamCore)

    def print_clusters_point_count(self):
        print(f"Number of n_clusters created: {len(self.clusters_dict.keys())}")
        for key, cl_df in self.clusters_dict.items():
            print(f"Cluster with center '{key}' has {cl_df.shape[0]} data_pd")

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
        print(f"Multi-index data: {self.m_idx_merge}")
        print(f"Max number of iterations (loops): {self.max_num_of_iter}")
        print("---")
        print(f"1d distance function passed (bool): {self.dist_func_1d is None}")
        print(f"Distance metric: {self.dist_metric}")
        print(f"Norm order: {self.norm_order}")
        print("---")
        print("Candidate center points options")
        print(f"Max number of candidate cluster medoids: {self.cand_options['n_cp_cand']}")
        print(f"Imaginary center point method is the geometric median aprox: {self.cand_options['g_median']}")
        print(f"Imaginary center point method is power mean of order: {self.cand_options['p_mean_ord']}")
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
        self.print_clusters_point_count()
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

    def __repr__(self):
        # self.print_all_iter_per_table()
        # self.print_all_tables_per_iter()
        self.print_selected()
        return "Kmedoids object"

    @classmethod
    def from_dist_vars(
            cls, data, n_clusters: int, dim_redux: int = None,
            n_init: int = 10, starting_points: list = None, im_st_points=None, st_p_method=None,
            dists_p_norm=1, cand_options=None, max_iter: int = None,
            dist_func_1d: callable = None, dist_metric=None, norm_order=None):

        return cls(
            data=data,
            n_clusters=n_clusters,
            dim_redux=dim_redux,
            n_init=n_init,
            starting_points=starting_points,
            im_st_points=im_st_points,
            st_p_method=st_p_method,
            dist_func=DistFunction(
                vector_dist_func=dist_func_1d,
                dist_metric=dist_metric,
                norm_order=norm_order,
                cache_points=None,  # points inserted inside object
            ),
            dists_p_norm=dists_p_norm,
            cand_options=cand_options,
            max_iter=max_iter)

    def set_up_res_from_pam_core(self, pam_core=None):
        pam_core = mf.def_var_value_if_none(pam_core, default=self.PamCore)

        self.labels = self.set_mult_idx_labels_from(pam_core.res_labels)
        self.medoids = self.set_mult_idx_labels_from(pam_core.res_cps_df)
        self.data_df = pd.DataFrame(self.data, index=self.labels.index)
        self.clusters_df = gpd.add_1st_lvl_index_to_df(self.data_df, self.labels, index_name="medoid").sort_index()
        self.clusters_dict = {idx: df for idx, df in self.clusters_df.groupby(level=0)}
        # self.clusters_dict = dict(pd.concat([self.data_df, self.labels.rename("medoid")], axis=1).groupby(["medoid"]))
        # self.clusters_df = pd.concat([df for df in self.clusters_dict], keys=self.clusters_dict.keys())1
        self.inertia = pam_core.final_config_cost

    def select_best_pam_of_st_points(self):
        # self.print_test_random_st_p_selection()
        pam_core_list = []
        inertia_list = []
        distortion_list = []
        config_cost_list = []
        for st_p_df in self.st_p_df_list:
            print("pam_line_270")
            print(f"starting points \n{st_p_df.index}")
            print(self.points.loc[st_p_df.index].drop_duplicates())
            print(f"num of np.nan: ", self.DistFunc.num_of_nan_in_cache())

            pam_core_res = PamCore.from_k_medoids_obj(
                k_medoids_obj=self,
                starting_medoids=st_p_df,
                dist_func=self.DistFunc.distance_func_cache_all
            )
            print(f"num of np.nan: ", self.DistFunc.num_of_nan_in_cache())
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

    def test_dists_cache_method(self):
        dist_cache_methods = {
            # "no_caching": lambda a, b, bl=False: self.DistFunc.distance_func_cache(a, b, bl),
            # "medoids": lambda a, b, bl=True: self.DistFunc.distance_func_cache_medoids(a, b, bl),
            # "recursion": lambda a, b, bl=True: self.DistFunc.distance_func_cache_na_plus(a, b, bl),
            # "drop_na": lambda a, b, bl=True: self.DistFunc.distance_func_cache_drop_na(a, b, bl),
            "calc_all": lambda a=None, b=None, bl=True: self.DistFunc.distance_func_cache_all(a, b, bl),
            # "elem_wise": lambda a, b, bl=True: self.DistFunc.distance_func_cache_element_wise(a, b, bl),
        }

        res_dict = {}
        for key in dist_cache_methods:
            self.dist_func_cache = dist_cache_methods[key]
            self.DistFunc.delete_cache()

            st_time = time()
            st_p_time = process_time()

            self.select_best_pam_of_st_points()

            end_time = time()
            end_p_time = process_time()

            res_dict[key] = [end_time - st_time, end_p_time - st_p_time]

        mf.print_dictionary(res_dict)
        print(pd.DataFrame(res_dict, index=["time", "processing_time"]).T)

    @staticmethod
    def merge_points_idx_if_m_index_data(data, sep_str):
        multi_index = False
        if isinstance(data.index, pd.MultiIndex):
            multi_index = True

        data_df = gpd.merge_data_pd_multi_index(data, sep_str=sep_str, index_name=None)

        return data_df, multi_index

    def prepare_points_df_from_data(self, dim_redux: int = None):
        # data_df, self.m_idx_merge = self.merge_points_idx_if_m_index_data(self.data, sep_str=self.merge_sep_str)

        data_df = pd.DataFrame(self.data) if not isinstance(self.data, pd.DataFrame | pd.Series) else self.data

        data_df = data_df.astype("float32")

        if dim_redux is None:
            return data_df

        # TODO PCoA method for all dist functions
        data_df = gdp.apply_pca(dataframe=data_df, remaining_dim=dim_redux)

        return data_df

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

    @staticmethod
    def plus_plus(ds, k, random_state=42):
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

        np.random.seed(random_state)
        centroids = [ds[0]]

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

        return np.array(centroids)

    def select_candidate_starting_points(self, method: str = None):
        if method is None:
            return self.points_no_dupl

        if method == "convex_hull":
            return self.select_st_p_from_convex_hull(dim=3)
        elif method == "kmeans++":
            return self.find_closest_points_to_im_centers(
                self.plus_plus(self.points_no_dupl, self.num_of_clusters)
            )

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

    def find_closest_points_to_im_centers(self, vector_list):
        return pd.Index(self.dist_func(self.points, vector_list).idxmin(axis=0).squeeze().unique())

    def convert_st_points_passed_to_index(self, idx_list, vector_list):
        if vector_list is not None:
            self.idx_min_dist_to_im_st_p = self.find_closest_points_to_im_centers(vector_list)

        if idx_list is None:
            idx_slt = self.idx_min_dist_to_im_st_p

        elif all(isinstance(st, int) for st in idx_list):
            idx_slt = self.points.iloc[mf.unique_list_vals_filter(
                flt_func=lambda x: x < self.num_of_points,
                list_of_vals=idx_list
            )].index.union(self.idx_min_dist_to_im_st_p)

        elif all(isinstance(st, (str, tuple, pd.Index, pd.MultiIndex)) for st in idx_list):
            idx_slt = self.points.loc[self.points.index.isin(idx_list)].index.union(self.idx_min_dist_to_im_st_p)

        else:
            idx_slt = self.points.loc[self.points.index.isin(idx_list)].index.union(self.idx_min_dist_to_im_st_p)

        """
        print("pam line 381 ")
        print(idxs)
        if self.im_st_points is not None:
            print(self.dist_func(self.points, self.im_st_points))
            print(self.dist_func(self.points, self.im_st_points).sort_values(by=[0]))
            print(self.dist_func(self.points, self.im_st_points).sort_values(by=[1]))
            print(self.dist_func(self.points, self.im_st_points).idxmin(axis=0))
        print(idx_slt)
        """
        return idx_slt

    def select_starting_points(self, candidate_st_p=None):
        cand_st_p = mf.def_var_value_if_none(value_passed=candidate_st_p, default=self.candidate_st_p)

        idx_slt = self.convert_st_points_passed_to_index(idx_list=self.starting_points, vector_list=self.im_st_points)
        """
        if st_list is None:
            return self.choose_n_priority_no_replace(
                first=self.candidate_st_p, second=self.points, third=pd.DataFrame(), num=self.num_of_clusters)

        if all(isinstance(st, str) for st in st_list):
            return self.choose_n_priority_no_replace(
                first=self.points.loc[self.points.index.isin(st_list)],
                second=self.candidate_st_p, third=self.points, num=self.num_of_clusters)

        if all(isinstance(st, int) for st in st_list):
            flt_st_list = list(filter(lambda x: x < self.num_of_points, np.unique(st_list)))
            return self.choose_n_priority_no_replace(
                first=self.points.iloc[flt_st_list],
                second=self.candidate_st_p, third=self.points, num=self.num_of_clusters)
        """
        if idx_slt.empty:
            return self.choose_n_priority_no_replace(
                first=cand_st_p, second=self.points_no_dupl, third=pd.DataFrame(), num=self.num_of_clusters)

        return self.choose_n_priority_no_replace(
            first=self.points_no_dupl.loc[idx_slt],
            second=cand_st_p, third=self.points_no_dupl, num=self.num_of_clusters)

    def select_n_sets_of_unique_starting_points(self, n_sets, n_unique=None):
        n_unique = min(
            mf.def_var_value_if_none(value_passed=n_unique, def_func=lambda: math.ceil(self.num_of_clusters*2/3)),
            self.num_of_clusters
        )

        n_set_df_list = []
        remaining_cand = self.candidate_st_p.copy()
        for i in range(n_sets):
            n_set_df_list += [self.select_starting_points(remaining_cand)]
            excluded_cand_idx = mf.select_n_random_rows_from_df(n_set_df_list[-1], n_row=n_unique).index
            remaining_cand = remaining_cand.loc[~remaining_cand.index.isin(excluded_cand_idx)]
            """
            print("pam line 459")
            print(excluded_cand_idx)
            print(remaining_cand.index)
            print(n_set_df_list[-1].index)
            """
        return n_set_df_list

    @staticmethod
    def create_m_idx_df_from_merged_idx(dataframe, separation_str, names_list, drop=True):
        # print("line 484 pam\n\n", dataframe.index)
        idx_tuple = dataframe.index.map(lambda x: tuple(x.split(separation_str))).set_names(names_list)
        return dataframe.reset_index(drop=drop).set_axis(idx_tuple)

    def set_mult_idx_labels_from(self, merged_idx_pd):
        if isinstance(merged_idx_pd.index, pd.MultiIndex) or not self.m_idx_merge:
            return merged_idx_pd
        else:
            return self.create_m_idx_df_from_merged_idx(
                dataframe=merged_idx_pd,
                separation_str=self.merge_sep_str,
                names_list=["DM", "post-opt"],
                drop=True
            )


class PamCore:
    def __init__(
            self, points: pd.DataFrame, st_medoids: pd.DataFrame, dist_func: callable,
            cand_options=None, dists_p_norm=1, max_iter=50, iter_comb=1000):

        self.points = points
        self.points_dupl = self.points.loc[~self.points.duplicated()]
        self.dupl_dict = {
            idx: self.points.loc[self.points.eq(self.points.loc[idx], axis=1).all(axis=1)].index
            for idx in self.points_dupl.index
        }

        self.st_medoids = st_medoids
        self.dist_func = dist_func

        self.dists_p_norm = dists_p_norm

        self.max_iter = max_iter
        self.iter_comb = iter_comb

        if cand_options is None:
            self.n_cp_cand = None
            self.g_median = None
            self.p_mean_ord = None
        else:
            self.n_cp_cand = mf.def_var_value_if_none(cand_options["n_cp_cand"], default=5)
            self.g_median = mf.def_var_value_if_none(cand_options["g_median"], default=False)
            self.p_mean_ord = mf.def_var_value_if_none(cand_options["p_mean_ord"], default=1)

        self.n_cl = len(self.st_medoids.index)
        self.comb_cl_depth = 4
        self.n_cl_change = min(self.n_cl, 5)

        self.cps_df_iter = []
        self.cp_labels_sr_iter = []
        self.cps_dists_matrix_iter = []
        self.config_cost_iter = []
        self.clusters_df_iter = []
        self.clusters_dict_iter = []

        self.imaginary_mid_points_iter = []

        self.iter_data = []
        self.iter_run_count = 0

        self.res_cps_df = pd.DataFrame()
        self.res_labels = pd.Series()
        self.cps_dists_matrix = pd.DataFrame()
        self.final_config_cost = None
        self.res_clusters_df = pd.DataFrame()
        self.res_clusters_df_dict = pd.DataFrame()

        self.CDMetrics: ClDistMetrics | None = None

        self.run_loop(starting_medoids=self.st_medoids)
        if self.check_for_single_res_label():
            return

        self.create_iter_data_dict_list()
        self.create_labels_iter_dataframe()

    def print_all_iter_per_table(self):
        mf.print_list(self.cp_labels_sr_iter)
        mf.print_list(self.clusters_dict_iter)
        mf.print_list(self.imaginary_mid_points_iter)
        mf.print_list(self.cps_df_iter)

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
            dist_func=mf.def_var_value_if_none(value_passed=dist_func, default=k_medoids_obj.dist_func),
            cand_options=k_medoids_obj.cand_options,
            dists_p_norm=k_medoids_obj.dists_norm_ord,
            max_iter=k_medoids_obj.max_num_of_iter,
            iter_comb=k_medoids_obj.iter_cp_comb
        )

    def create_labels_iter_dataframe(self):
        return pd.concat([val.rename(f"cps_iter({i + 1})") for i, val in enumerate(self.cp_labels_sr_iter)])

    def create_iter_data_dict_list(self):
        iter_data = []
        for i in range(len(self.clusters_df_iter)):
            iter_data_dict = {
                "centers": self.cps_df_iter[i],
                "labels": self.cp_labels_sr_iter[i],
                "cps_dists": self.cps_dists_matrix_iter[i],
                "config_cost": self.config_cost_iter[i],
                "cl_df": self.clusters_df_iter[i],
                "cl_dict": self.clusters_dict_iter[i],
                # "im_mids": self.imaginary_mid_points_iter[i]
            }
            iter_data += [iter_data_dict]
        self.iter_data = iter_data

    """
    def create_final_loop_result(self, new_labels_check=False):
        print(f"Line 463\n Iter run count: {self.iter_run_count}\n len(Iter_data): {len(self.iter_data)}")

        self.res_cps_df = self.cps_df_iter[-1]
        self.res_clusters_df_dict = self.clusters_dict_iter[-1]
        self.res_labels = self.cp_labels_sr_iter[-1]
        self.CDMetrics = ClDistMetrics.from_pam_core_obj(self)

        if not new_labels_check:
            return

        cps_dists_matrix, total_config_cost = self.calc_dists_and_total_cost_for_cps(
            points=self.points, center_points=self.res_cps_df, dist_func=self.dist_func, dists_p_norm=self.dists_p_norm
        )
        res_labels_new = self.create_labels_from_cps_dist_matrix(cps_dists_matrix)

        new_cld_metrics = ClDistMetrics.from_pam_core_obj(self, res_labels_new)

        print("pam line 1236")
        print(f"distortion: {self.CDMetrics.distortion}")
        print(f"new labels distortion: {new_cld_metrics.distortion}")
        print(f"Change in labels:")
        print(pd.concat([self.res_labels, res_labels_new], axis=1, keys=["Iter", "New"]).apply(
            lambda x: None if x.iloc[1] == x.iloc[0] else f"{x.iloc[0]}->{x.iloc[1]}", axis=1).dropna()
        )

        if new_cld_metrics.distortion < self.CDMetrics.distortion:
            self.res_labels = res_labels_new
            self.CDMetrics = new_cld_metrics
    """

    def run_loop(self, starting_medoids):
        cps_dists_matrix, total_config_cost = self.calc_dists_and_total_cost_for_cps(
            points=self.points,
            center_points=starting_medoids,
            dist_func=self.dist_func,
            dists_p_norm=self.dists_p_norm
        )
        self.set_up_configuration(cps_dists_matrix, total_config_cost)

        condition = True
        iter_count = 0

        while condition:
            cps_dists_matrix, total_config_cost = self.iteration_config_search_set_up(self.clusters_df_iter[-1])

            if cps_dists_matrix is not None:
                self.set_up_configuration(cps_dists_matrix, total_config_cost)

                if iter_count == self.max_iter:
                    condition = False

            else:
                condition = False

            iter_count += 1
            print("\n\t---::finished {} iterations".format(iter_count))

        self.iter_run_count = iter_count
        self.res_cps_df = self.cps_df_iter[-1]
        self.res_clusters_df = self.clusters_df_iter[-1]
        self.res_clusters_df_dict = self.clusters_dict_iter[-1]
        self.res_labels = self.cp_labels_sr_iter[-1]
        self.final_config_cost = self.config_cost_iter[-1]
        self.cps_dists_matrix = self.cps_dists_matrix_iter[-1]
        self.CDMetrics = ClDistMetrics.from_pam_core_obj(self)
        print(f"PAM Line 463\n Iter run count: {self.iter_run_count}")
        print(f"len(cps_df_iter): {len(self.cps_df_iter)}")
        print(f"len(cp_labels_sr_iter): {len(self.cp_labels_sr_iter)}")
        print(f"configuration cost: {self.final_config_cost}\n\n")

    def check_for_single_res_label(self):
        if len(self.cp_labels_sr_iter[-1].unique()) == 1:
            print("\n\tOnly one label found!\nSelecting random data_pd and running again")
            # self.select_starting_points()
            # self.run_loop(num_of_iter=self.max_num_of_iter)
            return True
        else:
            return False

    def set_up_configuration(self, cps_dists_matrix, total_config_cost):
        self.cps_dists_matrix_iter += [cps_dists_matrix]
        self.config_cost_iter += [total_config_cost]

        cp_labels_sr = self.create_labels_from_cps_dist_matrix(cps_dists_matrix)
        self.cp_labels_sr_iter += [cp_labels_sr]
        self.cps_df_iter += [self.points.loc[cps_dists_matrix.columns]]

        clusters_df = gpd.add_1st_lvl_index_to_df(self.points, index_list=cp_labels_sr, index_name="centers")
        self.clusters_df_iter += [clusters_df]
        self.clusters_dict_iter += [self.create_clusters_dict_from_clusters_df(clusters_df)]
        """
        print("pam line 1311")
        print("cps_dists_matrix\n", cps_dists_matrix)
        print(cps_dists_matrix.loc[cps_dists_matrix.columns])
        print(cps_dists_matrix.min(axis=1))
        print(cp_labels_sr.loc[cps_dists_matrix.columns])
        """
        if not len(cp_labels_sr.loc[cps_dists_matrix.columns].unique()) == self.n_cl:
            print(cps_dists_matrix.columns)
            print(self.cps_df_iter[-1])
            print(self.cps_df_iter[-2])
            print(len(cp_labels_sr.unique()))
            print(self.n_cl)
            raise Exception("There are duplicate points in the data")
        """
        print("cps_dists_matrix index:\n", cps_dists_matrix.index)
        print(list(self.cps_df_iter[-1].index.sort_values()))
        print(np.sort(cp_labels_sr.unique()))
        print(list(self.clusters_df_iter[-1].index.get_level_values(0).unique().sort_values()))
        """

    def calc_cl_cost_of_cand_from_imaginary_centers(self, clusters_df, p_mean_ord=None, g_median=None, n_cand=None):
        p_mean_ord = mf.def_var_value_if_none(p_mean_ord, default=self.p_mean_ord)
        g_median = mf.def_var_value_if_none(g_median, default=self.g_median)
        n_cand = mf.def_var_value_if_none(n_cand, default=self.n_cp_cand)

        imaginary_mid_points = self.find_imaginary_mid_of_clusters(
            clusters_df, p_mean_ord=p_mean_ord, geometric_median=g_median
        )
        self.imaginary_mid_points_iter += [imaginary_mid_points]

        print("pam_line_1644")
        print(clusters_df)
        print(imaginary_mid_points)

        dist_from_im, cl_cps_cost = self.find_n_cand_cl_cost_cps_from_im_cps(clusters_df, imaginary_mid_points, n_cand)
        return cl_cps_cost

    def calc_cl_cost_of_cand_from_all_cl_points(self, clusters_df):
        return {
            cp: self.calc_norm_of_dists_of_points_from_cps(
                cl_points_df=clusters_df.loc[cp],
                cand_cps=clusters_df.loc[cp].loc[
                    clusters_df.loc[cp].drop(cp).index.intersection(self.points_dupl.index)
                ],
                dist_func=self.dist_func,
                dists_p_norm=self.dists_p_norm
            ).sort_values(ascending=True)
            for cp in self.cps_df_iter[-1].index
        }

    def iteration_config_search_set_up(self, clusters_df):
        print("ckm_line_808")
        print(self.n_cp_cand)
        if isinstance(self.n_cp_cand, int):
            cl_cps_cost = self.calc_cl_cost_of_cand_from_imaginary_centers(clusters_df)
        else:
            cl_cps_cost = self.calc_cl_cost_of_cand_from_all_cl_points(clusters_df)

        """
        cps_comb = self.create_all_comb_for_n_cl_cps(
            cl_cps_cost=cl_cps_cost,
            cps_idx=self.cps_df_iter[-1].index,
            comb_cl_depth=self.comb_cl_depth,
            n_cp_change=self.n_cl_change
        )
        """
        """
        cps_comb = self.select_n_random_comb_any_cps(
            cl_cps_cost=cl_cps_cost,
            n_sets=10,
            n_sets_size=15,
            n_sets_pool=50
        )
        """

        min_config_cps_dist, min_config_cost = self.check_new_config_costs_by_group_of_cp_combs(
            cl_cps_cost=cl_cps_cost, n_comb=self.iter_comb
        )
        # cps_comb = self.create_n_comb_of_cps(cl_cps_cost=cl_cps_cost, n_comb=500)
        # min_config_cps_dist, min_config_cost = self.check_new_config_cost_of_cps_comb(cps_comb)

        return min_config_cps_dist, min_config_cost

    def check_new_config_costs_by_group_of_cp_combs(self, cl_cps_cost, n_comb=500, random_factor=2):
        min_config_cost = None
        min_config_cps_dist = None
        num_of_cl = len(cl_cps_cost)

        # n_cl_cps = mf.flatten_list_dimensions([[num_of_cl - k, k + 1] for k in range(math.ceil(num_of_cl / 2))])
        n_cl_cps = list(range(num_of_cl, 0, -1))
        # n_cl_cps = list(range(num_of_cl, num_of_cl-2, -1)) + [1]

        # n_cl_comb = math.ceil(n_comb / num_of_cl)
        n_cl_comb = math.ceil(n_comb / len(n_cl_cps))

        # comb_sample_size_fix_depth = int(n_cl_comb * random_factor / (3 ** i))

        for i in n_cl_cps:
            min_config_cps_dist, min_config_cost = self.check_new_config_cost_of_cps_comb(
                cps_comb=mf.random_choice_2d(
                    self.create_n_cl_random_comb(
                        cl_cps_cost=cl_cps_cost,
                        n_cl=i,
                        comb_sample_size=math.ceil(num_of_cl / i),     # num_of_cl
                        sample_size=n_cl_comb * random_factor   # 1 + (num_of_cl * (num_of_cl - i))
                    ),
                    size=n_cl_comb
                )
            )
            if min_config_cps_dist is not None:
                break

        return min_config_cps_dist, min_config_cost

    @staticmethod
    def create_labels_from_cps_dist_matrix(cps_dists: pd.DataFrame):
        return cps_dists.idxmin(axis=1).squeeze().rename("Center_points")

    @staticmethod
    def calc_dists_and_total_cost_for_cps(
            dist_func, points: pd.DataFrame, center_points: pd.DataFrame, dists_p_norm
    ) -> tuple[pd.DataFrame, float]:

        dist_from_cps_point_df = dist_func(points, center_points).abs()
        total_config_cost = np.sum(dist_from_cps_point_df.min(axis=1)**dists_p_norm)

        return dist_from_cps_point_df, total_config_cost

    @staticmethod
    def create_clusters_dict_from_clusters_df(clusters_df):
        return {cp: clusters_df.loc[cp] for cp in np.unique(clusters_df.index.get_level_values(0).values)}

    @classmethod
    def find_imaginary_mid_of_clusters(cls, clusters_df, p_mean_ord=1, geometric_median=True):
        if geometric_median:
            return pd.DataFrame([
                pd.Series(cls.geometric_median(clusters_df.loc[cp].values), name=cp, index=clusters_df.columns)
                for cp in clusters_df.index.get_level_values(0).unique()
            ])  # .rename_axis("Im/ry center of cluster:")

        return pd.DataFrame([
            clusters_df.loc[cp].apply(
                lambda col: gmf.calc_power_mean(col, p_mean_ord=p_mean_ord)
            ).rename(cp)
            for cp in np.unique(clusters_df.index.get_level_values(0).values)
        ])  # .rename_axis("Im/ry center of cluster:")

    @staticmethod
    def geometric_median(X, eps=1e-5):
        """
        Yehuda Vardi and Cun-Hui Zhang's algorithm for the geometric median,
        described in the paper "The multivariate L1-median and associated data depth"
        https://www.pnas.org/doi/pdf/10.1073/pnas.97.4.1423
        """

        y = np.mean(X, 0)

        while True:
            D = cdist(X, [y])
            nonzeros = (D != 0)[:, 0]

            Dinv = 1 / D[nonzeros]
            Dinvs = np.sum(Dinv)
            W = Dinv / Dinvs
            T = np.sum(W * X[nonzeros], 0)

            num_zeros = len(X) - np.sum(nonzeros)
            if num_zeros == 0:
                y1 = T
            elif num_zeros == len(X):
                return y
            else:
                R = (T - y) * Dinvs
                r = np.linalg.norm(R)
                rinv = 0 if r == 0 else num_zeros / r
                y1 = max(0, 1 - rinv) * T + min(1, rinv) * y

            if euclidean(y, y1) < eps:
                return y1

            y = y1

    @staticmethod
    def calc_norm_of_dists_of_points_from_cps(cl_points_df, cand_cps, dist_func, dists_p_norm=1):
        return dist_func(cl_points_df, cand_cps).apply(lambda x: np.linalg.norm(x, ord=dists_p_norm))

    def find_n_cand_cl_cost_cps_from_im_cps(self, clusters_df, imaginary_mid_points, n_cand=None):

        n_cand = mf.def_var_value_if_none(n_cand, default=self.n_cp_cand)
        cl_cps_cost = {}
        dist_from_im = {}

        for cp in self.cps_df_iter[-1].index:
            # print(f"pam line 1437:\ncp: {cp}\nclusters_df:\n{clusters_df}")
            """
            if self.n_cp_cand <= 5 or cl_len <= 5:
                print("pam line 1385\nFound a cluster with less than 5 points\ngoing back 2 configurations")
                self.set_up_configuration(self.cps_dists_matrix_iter[-2], self.config_cost_iter[-2])
                imaginary_mid_points = self.find_imaginary_mid_of_clusters(
                    self.clusters_df_iter[-1], p_mean_ord=self.p_mean_ord
                )
                self.imaginary_mid_points_iter += [imaginary_mid_points]
                return self.find_n_cand_cl_cost_cps_from_im_cps(
                    self.clusters_df_iter[-1], imaginary_mid_points
                )
            """
            cl_points_df: pd.DataFrame = clusters_df.loc[cp]
            cl_p_df_no_dupl = cl_points_df.loc[cl_points_df.drop(cp).index.intersection(self.points_dupl.index)]
            if cl_p_df_no_dupl.empty:
                raise Exception(
                    f"Cluster ({cp}) is empty:\n\n{cl_points_df}\n\nduplicates: {self.points_dupl.index}\n\n"
                )
            print("pam_line_1822")
            print(f"cl_points_df: {cl_points_df}\n\nduplicates: {self.points_dupl.index}")

            if cl_p_df_no_dupl.shape[0] == 1:
                # NO NEED TO CALCULATE
                dist_from_im[cp] = pd.Series(
                    [1],
                    index=cl_p_df_no_dupl.index,
                    name=f"dist from {cp}"
                )
                cl_cps_cost[cp] = dist_from_im[cp]
                continue

            dist_from_im[cp] = pd.Series(
                data=self.dist_func(
                    cl_p_df_no_dupl,
                    imaginary_mid_points.loc[cp],
                    False
                ),
                index=cl_p_df_no_dupl.index,
                name=f"dist from {cp}"
            ).sort_values(ascending=True)

            if n_cand <= 0:
                cl_cps_cost[cp] = dist_from_im[cp]
                continue

            cl_dist_from_im_cand_idx = dist_from_im[cp].head(min(len(cl_p_df_no_dupl.index), n_cand)).index
            cl_cps_cost[cp] = self.calc_norm_of_dists_of_points_from_cps(
                cl_points_df=cl_points_df,
                cand_cps=cl_points_df.loc[cl_dist_from_im_cand_idx],
                dist_func=self.dist_func,
                dists_p_norm=self.dists_p_norm
            ).sort_values(ascending=True)

        return dist_from_im, cl_cps_cost

    @classmethod
    def create_all_comb_for_n_cl_cps(
            cls, cl_cps_cost: dict, comb_cl_depth, n_cp_change):

        r_cl_k = mf.random_choice_2d(matrix=list(cl_cps_cost.keys()), size=min(n_cp_change, len(cl_cps_cost.keys())))
        # fixed_cps = list(cps_idx.difference(r_cl_k))

        mix_cps = cls.create_cps_combinations(cps_to_change=r_cl_k, cl_cps_cost=cl_cps_cost, depth=comb_cl_depth)

        return mix_cps

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

        if len(cps_to_change) == len(cl_cps_cost):
            return mf.all_lists_elements_combinations(*[list(cl_cps_cost[cl].index)[:depth] for cl in cps_to_change])

        fixed_cps = list(filter(lambda x: x not in cps_to_change, cl_cps_cost))
        if len(cps_to_change) == 1:
            return mf.all_lists_elements_combinations(list(cl_cps_cost[cps_to_change[0]].index)[:depth], [fixed_cps])

        new_cps = mf.all_lists_elements_combinations(*[list(cl_cps_cost[cl].index)[:depth] for cl in cps_to_change])
        mix_cps = mf.all_lists_elements_combinations(new_cps, [fixed_cps])

        return mix_cps

    @classmethod
    def select_n_random_comb_any_cps(
            cls, cl_cps_cost: dict, n_sets=10, n_sets_size=20, n_sets_pool=50, single_cp_change=False):
        """
        depth_n_cp_pairs = [
            (2, 10), (3, 6), (4, 5),
            (5, 4), (6, 4),
            (7, 3), (8, 3), (9, 3), (10, 3), (11, 3), (12, 3), (13, 3), (14, 3),
            (15, 2), (16, 2), (17, 2), (18, 2), (19, 2), (20, 2), (21, 2), (22, 2), (23, 2), (24, 2), (25, 2),
            (1000, 1)
        ]   # 25 TOTAL
        """
        if single_cp_change:
            return cls.create_all_comb_for_n_cl_cps(cl_cps_cost, comb_cl_depth=1000, n_cp_change=1)

        depth_n_cp_pairs_slt_depth = mf.return_min_pow_of_n_num_combs(n_sets=n_sets-2, min_comb_set_size=n_sets_pool)

        depth_n_cp_pairs_slt_n_cps = mf.return_min_num_of_n_power_combs(int(n_sets/2), min_comb_set_size=n_sets_pool)

        depth_n_cp_pairs = pd.Series([(100, 1)]*3 + depth_n_cp_pairs_slt_depth + depth_n_cp_pairs_slt_n_cps).unique()

        rand_mix_cps = []
        for pair in depth_n_cp_pairs:
            # print(f"pam_line_1588 : depth({pair[0]}) - centers({pair[1]})")
            rand_mix_cps += mf.random_choice_2d(
                matrix=cls.create_all_comb_for_n_cl_cps(
                    cl_cps_cost,
                    comb_cl_depth=pair[0],
                    n_cp_change=pair[1]
                ),
                size=n_sets_size
            )
        """
        # The correction of the fantom error 
        cps_comb = []
        for mix_cps in rand_mix_cps:
            if len(mix_cps) == len(cps_idx):
                cps_comb += [mix_cps]
            else:
                print("pam_line_1665: Found length missmatch in rand_mix_cps")
                pass
        """
        return rand_mix_cps

    @classmethod
    def create_n_comb_of_cps(
            cls,
            cl_cps_cost: dict[int | str | tuple[str], pd.DataFrame],
            n_comb: int = 500
    ) -> list[list[int | str | tuple[str]]]:

        num_of_cl = len(cl_cps_cost)

        n_cl_comb = math.ceil(n_comb / num_of_cl)

        all_comb_one = cls.create_all_one_cp_change_comb(
            cl_cps_cost=cl_cps_cost,
            depth=math.ceil(n_cl_comb / num_of_cl)
        )

        all_comb_two = cls.create_n_cl_random_comb(
            cl_cps_cost=cl_cps_cost,
            n_cl=2,
            comb_sample_size=num_of_cl,
            sample_size=2*n_cl_comb
        )

        if len(cl_cps_cost) < 3:
            return all_comb_two + all_comb_one

        all_comb_n = []
        for i in range(len(cl_cps_cost), 2, -1):
            all_comb_n += cls.create_n_cl_random_comb(
                cl_cps_cost=cl_cps_cost,
                n_cl=i,
                comb_sample_size=num_of_cl,
                sample_size=n_cl_comb
            )

        print("pam_line_1613")
        print(f"num_of_cl: {num_of_cl}")
        print(f"combinations total: {n_comb}")
        print(f"combinations pre cluster: {n_cl_comb}")
        print(f"{len(all_comb_one)} ~<=~ {num_of_cl * math.ceil(n_cl_comb / num_of_cl)}")
        print(f"{len(all_comb_two)} ~<=~ {2*n_cl_comb}")
        print(f"{len(all_comb_n)} ~<=~ {n_cl_comb*(num_of_cl - 2)}")
        print(f"Total num of combinations {len(all_comb_one)+len(all_comb_two)+len(all_comb_n)}")

        return all_comb_n + all_comb_two + all_comb_one

    @classmethod
    def create_all_one_cp_change_comb(cls, cl_cps_cost, depth=None):
        # creates len(cl_cps_cost) X depth combinations

        all_comb_one = []
        for cp in cl_cps_cost.keys():
            all_comb_one += cls.create_cps_combinations(
                cps_to_change=[cp],
                cl_cps_cost=cl_cps_cost,
                depth=mf.def_var_value_if_none(depth, def_func=lambda: math.ceil(5 + 20 / len(cl_cps_cost.keys())))
            )
        return all_comb_one

    @classmethod
    def create_n_cl_random_comb(cls, cl_cps_cost, n_cl: int, comb_sample_size=None, sample_size=None):
        # creates n_cl X len(cl_cps_cost) X depth combinations
        if comb_sample_size is None:
            comb_sample_size = len(cl_cps_cost)
        elif comb_sample_size <= 0:
            comb_sample_size = 1

        # sample_size = comb_sample_size * depth ^ n_cl
        samples_per_comb = math.ceil(sample_size / comb_sample_size)

        if len(cl_cps_cost.keys()) == n_cl:
            return cls.create_cps_combinations(
                cps_to_change=list(cl_cps_cost.keys()),
                cl_cps_cost=cl_cps_cost,
                depth=mf.return_min_num_of_pow_greater_than_min_val(
                    power=n_cl,
                    min_val=sample_size
                ) - 1
            )
        elif n_cl == 1:
            return cls.create_all_one_cp_change_comb(
                cl_cps_cost, depth=math.ceil(sample_size / len(cl_cps_cost))
            )[:sample_size]
        # elif len(cl_cps_cost.keys()) == n_cl - 1:
            # cps_comb = list(combinations(cl_cps_cost.keys(), n_cl))

        depth = -1 + mf.return_min_num_of_pow_greater_than_min_val(power=n_cl, min_val=samples_per_comb)

        cps_comb = mf.random_choice_2d(
            list(combinations(cl_cps_cost.keys(), n_cl)),
            size=math.ceil(sample_size / (depth ** n_cl))
        )
        """
        print("pam_line_1716")
        print(f"clusters({len(cl_cps_cost)}) - n_cl({n_cl})")
        print(f"cps_comb({math.ceil(sample_size / (depth ** n_cl))}) >= comb_sample_size({comb_sample_size})")
        print(f"comb_sample_size({comb_sample_size}) * depth({depth}) ^ n_cl({n_cl}) == sample_size({sample_size})")
        print(f"cps_comp({math.ceil(sample_size / (depth ** n_cl))}) >= real cps_comb({len(cps_comb)}) >= ")
        print(f"real cps_comb({len(cps_comb)}) >= combinations({len(list(combinations(cl_cps_cost.keys(), n_cl)))})")
        print(f"{math.ceil(sample_size / (depth ** n_cl)) * (depth ** n_cl)} >= {sample_size}")"""
        # samples_per_comb = depth ^ n_cl
        # comb_sample_size = math.ceil(sample_size / samples_per_comb)

        all_comb_n = []
        for cps in cps_comb:
            all_comb_n += cls.create_cps_combinations(
                cps_to_change=cps,
                cl_cps_cost=cl_cps_cost,
                depth=depth
            )

        return mf.random_choice_2d(all_comb_n, size=sample_size)

    def check_new_config_cost_of_cps_comb(self, cps_comb: list):
        min_config_cost = self.config_cost_iter[-1]
        min_config_cps_dist = None
        print(f"...{len(cps_comb)} comb checks started")
        for idx_list in cps_comb:
            cps_dists, total_cost = self.calc_dists_and_total_cost_for_cps(
                dist_func=self.dist_func,
                points=self.points,
                center_points=self.points.loc[idx_list],
                dists_p_norm=self.dists_p_norm
            )
            if total_cost < min_config_cost:
                # print(f"Configuration min cost: {total_cost}")
                min_config_cost = total_cost
                min_config_cps_dist = cps_dists
                break

        if min_config_cps_dist is not None:
            print(f"Configuration min cost: {min_config_cost}")
            print(f"CP combinations of this search: {len(cps_comb)}")
            new_cps = min_config_cps_dist.columns.difference(self.cps_df_iter[-1].index)
            # print(f"{new_cps}")
            print(f"n_cl({self.n_cl}) - new_cps({len(new_cps)})")

        return min_config_cps_dist, min_config_cost
