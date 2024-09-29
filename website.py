

import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import os
from PIL import Image
from website_static_clusters import static_clusters_page
from website_static_visualizations import static_visualization_page
from website_dynamic_visualizations import dynamic_visualization_page
from website_static_strength import static_strength_page
from website_dynamic_strength import dynamic_strength_page
from website_introduction import website_introduction


def main():
    st.set_page_config(layout="wide")  
    st.title('How Secure is Your Password?')
    page = 'Password Strength Analysis'
    # Sidebar with page options
    page = st.sidebar.radio(
        'Choose a Page', ['Introduction', 'General Password Analysis', 
                          'Check Your Password Strength',
                          'Password Strength Statistics', 'Password Strength Analysis'])
    
    if page == 'Introduction':
        website_introduction()

    if page == 'General Password Analysis':
        st.header('General Password Analysis')
        static_visualization_page()

    elif page == 'Check Your Password Strength':
        st.header('Check Your Password Strength')
        dynamic_visualization_page()
       
    elif page == 'Password Strength Statistics':
        st.header('Password Strength Statistics')
        static_strength_page()

    elif page == 'Password Strength Analysis':
        st.header('Password Strength Analysis')
        dynamic_strength_page()


if __name__ == "__main__":
    main()

