from functools import partial

import pandas as pd

from centroid_clustering import clustering_selection as cl_slt
from centroid_clustering.clustering_metrics import ClMetrics, DistFunction
import utils.visualization_functions as gvp


def main():
    # Random data generation for demo
    data, target_labels, cps = cl_slt.create_data_fit_for_clustering()
    # The class "ClSelect" is the main class that executes the
    # selected clustering algorithm for k=2 to k=20 clusters by default.
    # As an input argument needs a function that follows the protocol:
    #   def func(data: pd.DataFrame, n_clusters: int, max_iter: int) -> ClMetrics:

    # There are 3 default functions to use:
    #     - fasterPAM: "cl_metrics_set_up_for_faster_pam"
    #     - custom k-medoids: "cl_metrics_set_up_for_k_medoids"
    #     - k-means: "cl_metrics_set_up_for_kms_obj"

    # Use functools.partial to define any setting parameters of the above functions,
    # or define a new function that follows the above protocol.
    # For example, in order to compute the distance matrix between all data samples only once,
    # an instance of "DistFunction" class is created outside the 3 default functions above
    # and is given to them as input argument with functools.partial.

    # "DistFunction" initialization creates the samples distances matrix
    dist_func_obj = DistFunction(dist_metric="euclidean", cache_points=data)
    # Create the function that follows the protocol of "ClSelect"
    cl_m_fn = partial(
        cl_slt.cl_metrics_set_up_for_faster_pam,
        dist_func_obj=dist_func_obj,
        max_iter=100,
        dist_metric="euclidean"
    )
    # Creating an instance of "ClSelect"
    n_cl_obj = cl_slt.ClSelect(data=data, cl_metrics_obj_func=cl_m_fn, min_n_cl=2, max_n_cl=20, n_iter=100)

    # Get the default best clustering
    best_cl_metrics_obj: ClMetrics = n_cl_obj.cl_m_slt
    # To get a quick overview of the data clusters
    gvp.set_up_3d_graph_data(data, best_cl_metrics_obj.labels, "PCA").show()
    gvp.set_up_3d_graph_data(data, target_labels, "PCA").show()
    gvp.set_up_3d_graph_data(data, best_cl_metrics_obj.labels, "LDA").show()
    gvp.set_up_3d_graph_data(data, target_labels, "LDA").show()

    # Result data for best clustering ("ClMetrics" class instance)
    best_number_of_clusters: int = best_cl_metrics_obj.n_cl
    best_predicted_labels: pd.Series = best_cl_metrics_obj.labels
    cluster_centers: pd.DataFrame = best_cl_metrics_obj.cps_df
    metrics_per_cluster: pd.DataFrame = best_cl_metrics_obj.cluster_metrics_df
    metrics_per_sample: pd.DataFrame = best_cl_metrics_obj.samples_metrics_df
    clustering_metrics: pd.Series = best_cl_metrics_obj.metrics_sr

    # Result data for finding the best clustering ("ClSelect" class instance)
    all_labels: pd.DataFrame = n_cl_obj.labels_df
    all_clustering_metrics_score: pd.DataFrame = n_cl_obj.n_cl_metrics
    all_clustering_metric_fitting: pd.DataFrame = n_cl_obj.n_cl_m_fit
    clustering_unified_metric_score: pd.Series = n_cl_obj.n_cl_score

if __name__ == "__main__":
    main()