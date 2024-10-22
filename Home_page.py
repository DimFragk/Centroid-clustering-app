import streamlit as st

from centroid_clustering.streamlit_app.clustering_app import main as cl_app_main

if __name__ == "__main__":
    st.set_page_config(
        page_title="Clustering app",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items={
            "Get help": "https://github.com/DimFragk/Centroid-clustering-app/issues",
            "Report a bug": "https://github.com/DimFragk/Centroid-clustering-app/issues",
            "About": "Selection of the best centroid based clustering version with k-medoids and k-means"
        }
    )

    cl_app_main()
