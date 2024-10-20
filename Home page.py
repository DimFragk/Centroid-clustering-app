import streamlit as st

from centroid_clustering.streamlit_app.clustering_app import main as cl_app_main

if __name__ == "__main__":
    st.set_page_config(
        page_title="Clustering app",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="auto",
    )
    """    
        menu_items={
            'Get Help': "",
            'Report a bug': "",
            'About': "Welcome to the Large Group Decision Support System (LGDSS) app!"
        }
    )
    """
    cl_app_main()
