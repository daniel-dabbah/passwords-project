import streamlit as st
import pickle
from dynamic_visualization import (

    plot_password_length_histogram,
    plot_ascii_usage_histogram,
    plot_categories_bar_plot,
    special_character_position_violin_plot_for_specific_characters,
    plot_year_histogram,
    number_position_violin_plot,
    special_character_position_violin_plot,
    plot_categories_pie_plot_via_matplotlib,
    plot_entropy__percentages_histogram,
    plot_entropy_by_length_percentages_histogram,
    plot_categories_pie_plot_via_plotly
)

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    

def dynamic_visualization_page():


    st.header('Password Length Distribution')
    st.write("The password length histogram provides insights into the distribution of password lengths in the dataset. It helps us understand common password length preferences and potential vulnerabilities related to short passwords.")

    """ Entropy Clustering """
    dataset_name = 'rockyou_mini.txt'
    dataset_name = 'rockyou2024-100K.txt'

    loaded_data = load_data(f'{dataset_name}_data_passwords_statistics.pkl')
    loaded_passwords = loaded_data['passwords']
    loaded_statistics = loaded_data['statistics']

    
    # Length statistics analysis
    # Get password input from user for length comparison
    password = st.text_input("Enter a password for analysis", type="password", help="Type your password here to see how it compares to common length patterns")
    st.caption("Your password will be analyzed locally and not stored or transmitted.")

    if password:
        # Calculate the length of the entered password
        password_length = len(password)


        fig = plot_password_length_histogram(loaded_statistics['length_percentages'], dataset_name, password_length)
        st.pyplot(fig)

     # ASCII character usage analysis
    # Get password input from user for ASCII character comparison
    password = st.text_input("Enter a password for ASCII analysis", type="password", help="Type your password here to see how its characters compare to common usage patterns")
    st.caption("Your password will be analyzed locally and not stored or transmitted.")

    if password:
        # Use the password for ASCII character comparison
        fig = plot_ascii_usage_histogram(loaded_statistics['ascii_counts'], dataset_name, password)
        st.pyplot(fig)

    


   

    