import streamlit as st
import pickle
from visualizations import (

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
    

def static_visualization_page():


    st.header('Password Length Distribution')
    st.write("The password length histogram provides insights into the distribution of password lengths in the dataset. It helps us understand common password length preferences and potential vulnerabilities related to short passwords.")

    """ Entropy Clustering """
    dataset_name = 'rockyou_mini.txt'
    dataset_name = 'rockyou2024-100K.txt'

    loaded_data = load_data(f'{dataset_name}_data_passwords_statistics.pkl')
    loaded_passwords = loaded_data['passwords']
    loaded_statistics = loaded_data['statistics']

    
    fig = plot_password_length_histogram(loaded_statistics['length_percentages'], dataset_name)
    st.pyplot(fig)

    fig = plot_ascii_usage_histogram(loaded_statistics['ascii_counts'], dataset_name)
    st.pyplot(fig)
  
    fig = plot_categories_bar_plot(loaded_statistics, dataset_name)
    st.pyplot(fig)

    fig = plot_categories_pie_plot_via_matplotlib(loaded_statistics, dataset_name)
    st.pyplot(fig)

    fig =  number_position_violin_plot(loaded_statistics['number_positions'], dataset_name)
    st.pyplot(fig)

    fig = special_character_position_violin_plot(loaded_statistics['special_char_positions'], dataset_name)
    st.pyplot(fig)
    fig=special_character_position_violin_plot_for_specific_characters(loaded_statistics['special_char_positions_per_char']['.'], '.', dataset_name)
    st.pyplot(fig)
    fig=special_character_position_violin_plot_for_specific_characters(loaded_statistics['special_char_positions_per_char']['!'], '!', dataset_name)
    st.pyplot(fig)
    fig=special_character_position_violin_plot_for_specific_characters(loaded_statistics['special_char_positions_per_char']['@'], '@', dataset_name)
    st.pyplot(fig)
    fig=special_character_position_violin_plot_for_specific_characters(loaded_statistics['special_char_positions_per_char']['#'], '#', dataset_name)
    st.pyplot(fig)
    fig=special_character_position_violin_plot_for_specific_characters(loaded_statistics['special_char_positions_per_char']['$'], '$', dataset_name)
    st.pyplot(fig)
    fig=special_character_position_violin_plot_for_specific_characters(loaded_statistics['special_char_positions_per_char']['%'], '%', dataset_name)
    st.pyplot(fig)

    fig = plot_year_histogram(loaded_statistics['year_counts'], dataset_name)
    st.pyplot(fig)

    entropy = loaded_statistics['entropies']
    entropy_by_length = loaded_statistics['entropy_by_length']
    fig = plot_entropy__percentages_histogram(entropy, dataset_name)
    st.pyplot(fig)
    fig = plot_entropy_by_length_percentages_histogram(entropy_by_length[8], 8, dataset_name)
    st.pyplot(fig)

    