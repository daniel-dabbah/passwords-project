import streamlit as st
import json
import plotly.express as px
import pandas as pd
import numpy as np
import re

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)

def plot_password_length_distribution(loaded_statistics):
    """Plot the password length distribution."""
    st.header('Password Length Distribution')
    st.write("Understand the distribution of password lengths.")

    length_percentages = loaded_statistics['length_percentages']
    lengths = list(map(int, length_percentages.keys()))
    percentages = list(length_percentages.values())

    df_length = pd.DataFrame({
        'Password Length': lengths,
        'Percentage': percentages
    })

    # Sort by length
    df_length = df_length.sort_values('Password Length')

    # Create bar chart
    fig_length = px.bar(
        df_length,
        x='Password Length',
        y='Percentage',
        labels={'Password Length': 'Password Length', 'Percentage': 'Percentage'},
        title='Password Length Distribution',
        color='Percentage',
        color_continuous_scale='Blues',
        hover_data={'Percentage': ':.2f'}
    )

    # Update layout for better visualization
    fig_length.update_layout(
        xaxis=dict(
            tickmode='linear',
            dtick=1
        )
    )

    st.plotly_chart(fig_length, use_container_width=True)

def plot_ascii_character_usage(loaded_statistics):
    """Plot ASCII character usage in passwords."""
    st.header('ASCII Character Usage')
    st.write("Discover which ASCII characters are most commonly used in passwords.")

    ascii_counts = loaded_statistics['ascii_counts']

    ascii_order = [chr(i) for i in range(32, 127)]  # ASCII characters from space to '~'
    ascii_percentages = [ascii_counts.get(char, 0) for char in ascii_order]

    df_ascii = pd.DataFrame({
        'ASCII Character': ascii_order,
        'Percentage': ascii_percentages
    })

    # Ensure 'ASCII Character' is treated as a categorical variable with the specified order
    df_ascii['ASCII Character'] = pd.Categorical(df_ascii['ASCII Character'], categories=ascii_order, ordered=True)

    # Create bar chart
    fig_ascii = px.bar(
        df_ascii,
        x='ASCII Character',
        y='Percentage',
        labels={'ASCII Character': 'ASCII Character', 'Percentage': 'Percentage'},
        title='ASCII Character Usage in Passwords',
        color='Percentage',
        color_continuous_scale='Viridis'
    )

    # Update layout for better visualization
    fig_ascii.update_layout(
        xaxis_tickangle=-90,
        xaxis=dict(dtick=1)
    )

    st.plotly_chart(fig_ascii, use_container_width=True)

def plot_password_categories_distribution(loaded_statistics):
    """Plot password categories distribution using both bar chart and treemap."""
    st.header('Password Categories Distribution')
    st.write("Visualize the distribution of different password characteristics using bar charts and treemaps.")
    
    # Define the categories and their corresponding percentages
    categories = {
        "Lower Case Only": loaded_statistics.get('lower_case_only_percentage', 0),
        "Upper Case Only": loaded_statistics.get('upper_case_only_percentage', 0),
        "Numbers Only": loaded_statistics.get('numbers_only_percentage', 0),
        "Lower and Numbers": loaded_statistics.get('lower_case_and_numbers_percentage', 0),
        "Lower and Special": loaded_statistics.get('lower_case_and_special_characters_percentage', 0),
        "Upper and Numbers": loaded_statistics.get('upper_case_and_numbers_percentage', 0),
        "Upper and Special": loaded_statistics.get('upper_case_and_special_characters_percentage', 0),
        "Special and Numbers": loaded_statistics.get('special_characters_and_numbers_percentage', 0),
        "Lower Upper and Numbers": loaded_statistics.get('lower_upper_case_and_numbers_percentage', 0),
        "Lower Upper and Special": loaded_statistics.get('lower_upper_case_and_special_characters_percentage', 0),
        "Lower Special and Numbers": loaded_statistics.get('lower_special_characters_and_numbers_percentage', 0),
        "Upper Special and Numbers": loaded_statistics.get('upper_special_characters_and_numbers_percentage', 0),
        "All Character Types": loaded_statistics.get('all_character_types_percentage', 0),
        "Lower and Upper": loaded_statistics.get('lower_case_and_upper_case_percentage', 0)
    }

    # Create a DataFrame for the bar chart
    df_categories = pd.DataFrame({
        'Category': list(categories.keys()),
        'Percentage': list(categories.values())
    })

    # **Treemap Visualization**
    st.subheader('Treemap')
    fig_categories_treemap = px.treemap(
        df_categories,
        path=['Category'],
        values='Percentage',
        color='Percentage',
        color_continuous_scale='Inferno',
        title='Password Categories Distribution Treemap',
        height=800,  # Increased height
        width=1000    # Increased width
    )

    fig_categories_treemap.update_traces(textinfo='label+percent parent')

    st.plotly_chart(fig_categories_treemap, use_container_width=True)

def plot_number_position_violin(loaded_statistics):
    """Plot the position of numbers in passwords."""
    st.header('Position of Numbers in Passwords')
    st.write("Visualize where numbers are commonly placed within passwords.")

    number_positions = loaded_statistics['number_positions']
    if number_positions:
        df_number_positions = pd.DataFrame(number_positions, columns=['Password Length', 'Normalized Position'])
        df_number_positions['Password Length'] = df_number_positions['Password Length'].astype(str)

        fig_violin_numbers = px.violin(
            df_number_positions,
            x='Password Length',
            y='Normalized Position',
            color='Password Length',  # Use Password Length as categorical color
            title='Position of Numbers in Passwords',
            box=True,
            points=False,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        fig_violin_numbers.update_layout(
            xaxis=dict(
                tickmode='linear',
                dtick=1
            ),
            showlegend=False
        )

        st.plotly_chart(fig_violin_numbers, use_container_width=True)
    else:
        st.write("No number position data available.")

def plot_special_character_position_violin(loaded_statistics):
    """Plot the position of special characters in passwords."""
    st.header('Position of Special Characters in Passwords')
    st.write("Visualize where special characters are commonly placed within passwords.")

    special_char_positions = loaded_statistics['special_char_positions']
    if special_char_positions:
        df_special_positions = pd.DataFrame(special_char_positions, columns=['Password Length', 'Normalized Position'])
        df_special_positions['Password Length'] = df_special_positions['Password Length'].astype(str)

        fig_violin_special = px.violin(
            df_special_positions,
            x='Password Length',
            y='Normalized Position',
            color='Password Length',
            title='Position of Special Characters in Passwords',
            box=True,
            points=False,
            color_discrete_sequence=px.colors.qualitative.Plotly
        )

        fig_violin_special.update_layout(
            xaxis=dict(
                tickmode='linear',
                dtick=1
            ),
            showlegend=False
        )

        st.plotly_chart(fig_violin_special, use_container_width=True)
    else:
        st.write("No special character position data available.")

def plot_position_of_specific_special_characters(loaded_statistics):
    """Plot positions of a specific special character entered by the user."""
    st.header('Position of Specific Special Characters in Passwords')
    st.write("Visualize where a specific special character is commonly placed within passwords.")

    special_char_positions_per_char = loaded_statistics['special_char_positions_per_char']

    # Allow user to input special character
    special_char = st.text_input('Enter a special character to analyze its position:', value='!')

    if special_char:
        if special_char in special_char_positions_per_char:
            positions = special_char_positions_per_char[special_char]
            if positions:
                df_char_positions = pd.DataFrame(positions, columns=['Password Length', 'Normalized Position'])
                df_char_positions['Password Length'] = df_char_positions['Password Length'].astype(str)

                fig_char_violin = px.violin(
                    df_char_positions,
                    x='Password Length',
                    y='Normalized Position',
                    color='Password Length',  # Use Password Length as categorical color
                    title=f'Position of "{special_char}" in Passwords',
                    box=True,
                    points=False,
                    color_discrete_sequence=px.colors.qualitative.Dark24  # Optional: specify color palette
                )

                fig_char_violin.update_layout(
                    xaxis=dict(
                        tickmode='linear',
                        dtick=1
                    ),
                    showlegend=False  # Hide legend if it's too cluttered
                )

                st.plotly_chart(fig_char_violin, use_container_width=True)
            else:
                st.write(f"No position data available for special character '{special_char}'.")
        else:
            st.write(f"No data available for special character '{special_char}'.")
    else:
        st.write("Please enter a special character to analyze.")

def plot_year_usage(loaded_statistics):
    """Plot year usage in passwords."""
    st.header('Year Usage in Passwords')
    st.write("Discover how often years appear in passwords, indicating potential use of dates or birth years.")

    year_counts = loaded_statistics['year_counts']
    years = sorted(map(int, year_counts.keys()))
    counts = [year_counts[str(year)] for year in years]

    df_years = pd.DataFrame({
        'Year': years,
        'Count': counts
    })

    fig_years = px.bar(
        df_years,
        x='Year',
        y='Count',
        labels={'Year': 'Year', 'Count': 'Count'},
        title='Years Found in Passwords',
        color='Count',
        color_continuous_scale='Turbo'
    )

    st.plotly_chart(fig_years, use_container_width=True)

def plot_entropy_distribution(loaded_statistics):
    """Plot entropy distribution of passwords."""
    st.header('Entropy Distribution')
    st.write("Entropy measures the unpredictability of passwords. Higher entropy indicates stronger passwords.")

    entropies = loaded_statistics['entropies']

    df_entropy = pd.DataFrame({
        'Entropy': entropies
    })

    fig_entropy = px.histogram(
        df_entropy,
        x='Entropy',
        nbins=50,
        labels={'Entropy': 'Entropy (bits)', 'count': 'Count'},
        title='Entropy Distribution of Passwords',
        color_discrete_sequence=['indianred']
    )

    st.plotly_chart(fig_entropy, use_container_width=True)

def plot_entropy_distribution_user_length(loaded_statistics):
    """Plot entropy distribution for a user-specified password length."""
    st.header('Entropy Distribution for Specific Password Length')
    st.write("Enter a password length to see the entropy distribution.")
    
    # User input for password length
    length = st.number_input('Select Password Length:', min_value=1, max_value=30, value=8, step=1)
    
    entropies_by_length = loaded_statistics['entropy_by_length_percentages']
    entropies_length = entropies_by_length.get(str(length), [])
    
    if entropies_length:
        df_entropy_length = pd.DataFrame({
            'Entropy': entropies_length
        })
    
        fig_entropy_length = px.histogram(
            df_entropy_length,
            x='Entropy',
            nbins=50,
            labels={'Entropy': 'Entropy (bits)', 'percentage': 'Percenatge'},
            title=f'Entropy Distribution for Passwords of Length {length}',
            color_discrete_sequence=['teal'],
            height=600,  # Increased height
            width=1000    # Increased width
        )
    
        st.plotly_chart(fig_entropy_length, use_container_width=True)
    else:
        st.write(f"No entropy data available for passwords of length {length}.")

def plot_average_numbers_by_length(loaded_statistics, character_type):
    """Plot the average number of specified character type by password length."""
    if character_type == 'Numbers':
        percentages_data = loaded_statistics.get('count_of_numbers_per_length_per_count_percentages', {})
        title = 'Average Number of Numbers by Password Length'
        y_label = 'Average Number of Numbers'
        color_scale = 'Blues'
    elif character_type == 'Upper Case Letters':
        percentages_data = loaded_statistics.get('count_of_upper_case_per_length_per_count_percentages', {})
        title = 'Average Number of Upper Case Letters by Password Length'
        y_label = 'Average Number of Upper Case Letters'
        color_scale = 'Oranges'
    elif character_type == 'Special Characters':
        percentages_data = loaded_statistics.get('count_of_special_characters_per_length_per_count_percentages', {})
        title = 'Average Number of Special Characters by Password Length'
        y_label = 'Average Number of Special Characters'
        color_scale = 'Magma'
    else:
        st.write(f"No data available for character type '{character_type}'.")
        return

    if percentages_data:
        data = []
        for length_str, counts_dict in percentages_data.items():
            length = int(length_str)
            expected_value = 0
            for count_str, percentage in counts_dict.items():
                count = int(count_str)
                expected_value += count * (percentage / 100)
            data.append({
                'Password Length': length,
                y_label: expected_value
            })
        
        # Create a DataFrame from the list of dictionaries
        df_data = pd.DataFrame(data)
        
        # Sort the DataFrame by Password Length
        df_data = df_data.sort_values('Password Length')
        
        # Create a bar plot
        fig = px.bar(
            df_data,
            x='Password Length',
            y=y_label,
            labels={'Password Length': 'Password Length', y_label: y_label},
            title=title,
            color=y_label,
            color_continuous_scale=color_scale
        )
        
        # Update layout for better visualization
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                dtick=1  # Tick every 1 unit
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"No data available for {character_type} by password length.")

def plot_average_numbers_for_length(loaded_statistics, length):
    """Plot average numbers of upper letters, special characters, and numbers for a specific password length."""
    st.header(f'Average Character Counts for Passwords of Length {length}')
    length_str = str(length)
    data = {}

    # For Numbers
    percentages_data = loaded_statistics.get('count_of_numbers_per_length_per_count_percentages', {})
    counts_dict = percentages_data.get(length_str, {})
    expected_numbers = 0
    for count_str, percentage in counts_dict.items():
        count = int(count_str)
        expected_numbers += count * (percentage / 100)
    data['Average Number of Numbers'] = expected_numbers

    # For Upper Case Letters
    percentages_data = loaded_statistics.get('count_of_upper_case_per_length_per_count_percentages', {})
    counts_dict = percentages_data.get(length_str, {})
    expected_upper = 0
    for count_str, percentage in counts_dict.items():
        count = int(count_str)
        expected_upper += count * (percentage / 100)
    data['Average Number of Upper Case Letters'] = expected_upper

    # For Special Characters
    percentages_data = loaded_statistics.get('count_of_special_characters_per_length_per_count_percentages', {})
    counts_dict = percentages_data.get(length_str, {})
    expected_special = 0
    for count_str, percentage in counts_dict.items():
        count = int(count_str)
        expected_special += count * (percentage / 100)
    data['Average Number of Special Characters'] = expected_special

    # Create a DataFrame
    df_data = pd.DataFrame({
        'Character Type': list(data.keys()),
        'Average Count': list(data.values())
    })

    # Create a bar plot
    fig = px.bar(
        df_data,
        x='Character Type',
        y='Average Count',
        labels={'Character Type': 'Character Type', 'Average Count': 'Average Count'},
        title=f'Average Character Counts for Passwords of Length {length}',
        color='Character Type',
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    st.plotly_chart(fig, use_container_width=True)

def plot_distribution_for_length(loaded_statistics, length, character_type):
    """Plot the distribution of counts for a specific character type at a given password length."""
    length_str = str(length)
    if character_type == 'Numbers':
        percentages_data = loaded_statistics.get('count_of_numbers_per_length_per_count_percentages', {})
        title = f'Distribution of Number Counts for Passwords of Length {length}'
        x_label = 'Number of Numbers'
        color = 'Blues'
    elif character_type == 'Upper Case Letters':
        percentages_data = loaded_statistics.get('count_of_upper_case_per_length_per_count_percentages', {})
        title = f'Distribution of Upper Case Letter Counts for Passwords of Length {length}'
        x_label = 'Number of Upper Case Letters'
        color = 'Oranges'
    elif character_type == 'Special Characters':
        percentages_data = loaded_statistics.get('count_of_special_characters_per_length_per_count_percentages', {})
        title = f'Distribution of Special Character Counts for Passwords of Length {length}'
        x_label = 'Number of Special Characters'
        color = 'Magma'
    else:
        st.write(f"No data available for character type '{character_type}'.")
        return

    counts_dict = percentages_data.get(length_str, {})
    if counts_dict:
        counts = []
        percentages = []
        for count_str, percentage in counts_dict.items():
            counts.append(int(count_str))
            percentages.append(percentage)
        df_counts = pd.DataFrame({
            x_label: counts,
            'Percentage': percentages
        })

        # Sort the DataFrame by count
        df_counts = df_counts.sort_values(x_label)

        # Create a bar plot
        fig = px.bar(
            df_counts,
            x=x_label,
            y='Percentage',
            labels={x_label: x_label, 'Percentage': 'Percentage'},
            title=title,
            color='Percentage',
            color_continuous_scale=color
        )

        # Update layout for better visualization
        fig.update_layout(
            xaxis=dict(
                tickmode='linear',
                dtick=1  # Tick every 1 unit
            )
        )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.write(f"No data available for passwords of length {length}.")

def plot_average_numbers_for_length(loaded_statistics, length):
    """Plot average numbers and distributions for upper letters, special characters, and numbers for a specific password length."""
    st.header(f'Character Counts for Passwords of Length {length}')
    length_str = str(length)
    data = {}

    # For Numbers
    percentages_data_numbers = loaded_statistics.get('count_of_numbers_per_length_per_count_percentages', {})
    counts_dict_numbers = percentages_data_numbers.get(length_str, {})
    expected_numbers = 0
    for count_str, percentage in counts_dict_numbers.items():
        count = int(count_str)
        expected_numbers += count * (percentage / 100)
    data['Average Number of Numbers'] = expected_numbers

    # For Upper Case Letters
    percentages_data_upper = loaded_statistics.get('count_of_upper_case_per_length_per_count_percentages', {})
    counts_dict_upper = percentages_data_upper.get(length_str, {})
    expected_upper = 0
    for count_str, percentage in counts_dict_upper.items():
        count = int(count_str)
        expected_upper += count * (percentage / 100)
    data['Average Number of Upper Case Letters'] = expected_upper

    # For Special Characters
    percentages_data_special = loaded_statistics.get('count_of_special_characters_per_length_per_count_percentages', {})
    counts_dict_special = percentages_data_special.get(length_str, {})
    expected_special = 0
    for count_str, percentage in counts_dict_special.items():
        count = int(count_str)
        expected_special += count * (percentage / 100)
    data['Average Number of Special Characters'] = expected_special

    # Create a DataFrame for the average counts
    df_data = pd.DataFrame({
        'Character Type': list(data.keys()),
        'Average Count': list(data.values())
    })

    # Create an aggregated bar plot for average counts
    fig = px.bar(
        df_data,
        x='Character Type',
        y='Average Count',
        labels={'Character Type': 'Character Type', 'Average Count': 'Average Count'},
        title=f'Average Character Counts for Passwords of Length {length}',
        color='Character Type',
        color_discrete_sequence=px.colors.qualitative.Set3
    )

    st.plotly_chart(fig, use_container_width=True)

    # Plot distributions for each character type
    plot_distribution_for_length(loaded_statistics, length, 'Numbers')
    plot_distribution_for_length(loaded_statistics, length, 'Upper Case Letters')
    plot_distribution_for_length(loaded_statistics, length, 'Special Characters')

def static_visualization_page():
    st.title('Password Dataset Visualization')
    st.write("Explore various statistics of passwords in the dataset with interactive visualizations.")

    # Load the dataset
    dataset_name = 'rockyou2024-100K.txt'  # Update with your dataset name

    loaded_data = load_data(f'{dataset_name}_data_passwords_statistics.json')
    loaded_passwords = loaded_data['passwords']
    loaded_statistics = loaded_data['statistics']

    # Plotting functions
    plot_password_length_distribution(loaded_statistics)
    plot_ascii_character_usage(loaded_statistics)
    plot_password_categories_distribution(loaded_statistics)
    plot_number_position_violin(loaded_statistics)
    plot_special_character_position_violin(loaded_statistics)
    plot_position_of_specific_special_characters(loaded_statistics)
    plot_year_usage(loaded_statistics)
    plot_entropy_distribution(loaded_statistics)
    # plot_entropy_distribution_user_length(loaded_statistics)


    # New plots
    plot_average_numbers_by_length(loaded_statistics, 'Numbers')
    plot_average_numbers_by_length(loaded_statistics, 'Upper Case Letters')
    plot_average_numbers_by_length(loaded_statistics, 'Special Characters')

    # Input for specific password length
    st.header('Character Counts for Specific Password Length')
    st.write("Enter a password length to see the average counts and distributions of upper letters, special characters, and numbers.")

    length = st.number_input('Password Length:', min_value=1, max_value=30, value=8, step=1)

    plot_average_numbers_for_length(loaded_statistics, length)

if __name__ == '__main__':
    static_visualization_page()
