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

### Installation

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
> Then to start the streamlit app:
>`streamlit run Home_page.py`

### Reporting issues and support requests

Please report errors, support requests and questions as 
[issue](https://github.com/kno10/python-kmedoids/issues) 
within the repository's issue tracker and i will do my best to address them.

### License: GPL-3 or later
