# Welcome to the centroid clustering app
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

The goal of the centroid based clustering app, is to support the selection 
of the "best" among many different clustering versions/attempts, 
on the same input data. 
        
There are 2 main decision parameters: 
- The clustering method (including settings parameters)
- The number of clusters (k)

### Implemented clustering methods
- k-medoids
- k-means

#### The k-medoids centroid_clustering algorithms used are:  
1. 'fasterpam': One of the PAM algorithms from the python library "kmedoids"
([pypi.org](https://pypi.org/project/kmedoids/)) 
([github](https://github.com/kno10/python-kmedoids)) 
2. 'Custom kmedoids': My attempt at creating a k-medoids centroid_clustering algorithm, it is included in this project 

#### The k-means centroid_clustering algorithm used is:
1. 'KMeans': from the python library scikit-learn
([Documentation](https://scikit-learn.org/1.5/modules/generated/sklearn.cluster.KMeans.html))

### General Methodology

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

### Installation and documentation

- This project is not uploaded to PyPI. 
- Python 3.11.0 is used.
- The apps web GUI is created with the [streamlit](https://streamlit.io/) 
python library ([streamlit documentation](https://docs.streamlit.io/)). 

>After activating the desirable virtual environment in the project folder:
>
>`pip install -r requirements.txt`
>
>or to install the latest versions of the imported libraries (may not always work):
>
>`pip install -r requirements_without_versions.txt`
> 
> Then, to start the streamlit app locally:
> 
>`streamlit run Home_page.py`
> 

If you want to use the code directly into your project, 
place the folders "utils" and "centroid_clustering" in the root of your project.
Then, a simple demo.py file looks like this:

- Imports and data generation
> ```python
> from functools import partial
> from centroid_clustering import clustering_selection as cl_slt
> from centroid_clustering.clustering_metrics import ClMetrics, DistFunction
> import utils.visualization_functions as gvp
> # Random data generation for demo
> data, target_labels, cps = cl_slt.create_data_fit_for_clustering()
> ```

- Create an instance of class "ClSelect" 
> The class "ClSelect" is the main class that executes the 
> selected clustering algorithm for k=2 to k=20 clusters by default.
> As an input argument needs a function that follows the protocol: 
> ```python
> def func(data: pd.DataFrame, n_clusters: int, max_iter: int) -> ClMetrics:
> ```
> There are 3 default functions to use: 
> - fasterPAM: "cl_metrics_set_up_for_faster_pam"
> - custom k-medoids: "cl_metrics_set_up_for_k_medoids"
> - k-means: "cl_metrics_set_up_for_kms_obj"
>
> Use functools.partial to define any setting parameters of the above functions, 
> or define a new function that follows the above protocol.
> For example, in order to compute the distance matrix between all data samples only once, 
> an instance of "DistFunction" class is created outside the 3 default functions above 
> and is given to them as input argument with functools.partial.
> 
> ```python
> # "DistFunction" initialization computes all samples distances matrix
> dist_func_obj = DistFunction(dist_metric="euclidean", cache_points=data)
> # Create the function that follows the protocol of "ClSelect"
> cl_m_fn = partial(
>     cl_slt.cl_metrics_set_up_for_faster_pam, 
>     dist_func_obj=dist_func_obj, 
>     max_iter=100, 
>     dist_metric="euclidean"
> )
> # Creating an instance of "ClSelect"
> n_cl_obj = cl_slt.ClSelect(data=data, cl_metrics_obj_func=cl_m_fn, min_n_cl=2, max_n_cl=20, n_iter=100)
> ```

- Results overview
> Quick overview with a plotly 3d scatter plot:
>```python
> # Get the default best clustering instance of "ClMetrics"
> best_cl_metrics_obj: ClMetrics = n_cl_obj.cl_m_slt
> # Get a quick overview of the data clusters with dimension reduction of the features (samples columns)
> gvp.set_up_3d_graph_data(data, best_cl_metrics_obj.labels, "PCA").show()
> gvp.set_up_3d_graph_data(data, target_labels, "PCA").show()
> gvp.set_up_3d_graph_data(data, best_cl_metrics_obj.labels, "LDA").show()
> gvp.set_up_3d_graph_data(data, target_labels, "LDA").show()
> ```
> To extract the main result data:
> - Best clustering ("ClMetrics" class instance) 
> ```python 
> best_number_of_clusters: int = best_cl_metrics_obj.n_cl
> best_predicted_labels: pd.Series = best_cl_metrics_obj.labels
> cluster_centers: pd.DataFrame = best_cl_metrics_obj.cps_df
> metrics_per_cluster: pd.DataFrame = best_cl_metrics_obj.cluster_metrics_df
> metrics_per_sample: pd.DataFrame = best_cl_metrics_obj.samples_metrics_df
> clustering_metrics: pd.Series = best_cl_metrics_obj.metrics_sr
> ```
>  - Searching for the best clustering ("ClSelect" class instance)
> ```python
> all_labels: pd.DataFrame = n_cl_obj.labels_df
> all_clustering_metrics_score: pd.DataFrame = n_cl_obj.n_cl_metrics
> all_clustering_metric_fitting: pd.DataFrame = n_cl_obj.n_cl_m_fit
> clustering_unified_metric_score: pd.Series = n_cl_obj.n_cl_score
> ```

There are many more things to find in the code! 
Search the centroid_clustering\streamlit_app\result_pages for the results shown on the streamlit app. 
The custom_k_medoids works a lot slower than fasterPAM but the benefit is that it uses some experimental settings.

### Reporting issues and support requests

Please report errors, support requests and questions as 
[issue](https://github.com/DimFragk/Centroid-clustering-app/issues) 
within the repository's issue tracker and i will do my best to address them.

### License: GPL-3 or later
