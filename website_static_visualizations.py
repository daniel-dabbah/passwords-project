import streamlit as st
import json
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import re

def load_data(filename):
    """Load data from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

def plot_password_length_distribution(loaded_statistics):
    """Plot the password length distribution."""
    st.header('Password Length Distribution')
    st.write("""
    Here we show the distribution of different password lengths. The graph displays the percentage of passwords for each length.
    
    We can observe that the distribution forms a curve-like shape. This visualization helps us understand which password lengths are most common among users.
    
    Key observations:
    - The peak of the curve represents the most frequently used password length.
    - Shorter and longer passwords are less common, forming the tails of the distribution.
    - This curve shape is typical in password datasets, reflecting users' tendencies to choose passwords of certain lengths.

    Analyzing this distribution can provide insights into user behavior and potential vulnerabilities in password selection.
    """)

    length_percentages = loaded_statistics.get('length_percentages', {})
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
    st.header('ASCII character usage in passwords')
    st.write("""
    This graph shows the distribution of each ASCII character in the dataset. We can observe several interesting patterns:

    1. Lowercase characters are very popular in passwords, indicating a strong preference among users.
    2. Characters like '?' or '^' are less common. This suggests that incorporating these less frequently used characters could potentially improve password security.
    3. The varying usage of different characters provides insights into user behavior and potential areas for enhancing password strength.

    Analyzing this distribution can help in understanding common practices in password creation and identifying opportunities for improving password security.
    """)

    ascii_counts = loaded_statistics.get('ascii_counts', {})

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
    """Plot password categories distribution using treemap."""
    st.header('Password Categories Distribution')
    st.write("""
    This treemap visualization shows the distribution of different password categories based on their character composition. Here are some key observations:

    1. The 'Lower and Numbers' category is very common, indicating that many users prefer passwords combining lowercase letters and numbers.
    2. 'Lowercase Only' passwords are also prevalent, suggesting a significant portion of users prefer simple, easily memorable passwords.
    3. The 'Numbers Only' category is less common, which is a positive sign as such passwords are generally considered weak.
    4. The 'Upper Special and Numbers' category is one of the least common, indicating that more complex password combinations are less frequently used.

    The size and color intensity of each box in the treemap represent the percentage of passwords falling into that category. Larger, brighter boxes indicate more common categories, while smaller, darker boxes represent less common ones.

    This visualization helps us understand user behavior in password creation and identify areas where password policies might be improved to enhance overall security.
    """)
    
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

    # Create a DataFrame for the treemap
    df_categories = pd.DataFrame({
        'Category': list(categories.keys()),
        'Percentage': list(categories.values())
    })

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
    st.header('Distribution of Number Positions in Passwords')
    st.write("""
    This visualization shows the distribution of number positions in passwords, categorized by password length.

    Key observations:
    1. Numbers frequently appear at the start and end of passwords.
    2. The middle sections of passwords often lack numbers.
    3. The distribution patterns remain relatively consistent across different password lengths.

    Insights:
    - Users tend to place numbers at password extremities, potentially reducing security.
    - Incorporating numbers in the middle of passwords could enhance security.
    - These patterns reveal common user habits in password creation.

    This analysis helps identify areas for improving password strength through better number placement.
    """)

    number_positions = loaded_statistics.get('number_positions', [])
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
    st.header('Where Special Characters Appear in Passwords')
    st.write("""
    This chart shows where special characters are used in passwords of different lengths. Special characters are characters from this set: !@#$%^&*()-_=+[]{|;:'\,.<>?/`~ }

    When users choose to include special characters in their passwords, we can observe that:

    1. People often put special characters in the middle of their passwords.
    2. There aren't many special characters at the end of passwords.
    3. This pattern stays mostly the same, even for different password lengths.

    What this means:
    - Putting special characters in the middle is a common choice.
    - Not using special characters at the end might make passwords less secure.
    - People tend to use special characters in similar ways, regardless of password length.

    Try it yourself:
    Below, you can type in a symbol from this group: !@#$%^&*()-_=+[]{|;:'\,.<>?/`~
    You'll see where these special characters usually appear in passwords of different lengths.

    This visualization helps us understand how people use special characters when creating passwords, and how the distribution of these characters varies based on password length.
    """)

    special_char_positions = loaded_statistics.get('special_char_positions', [])
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
    st.header('Analyze Special Character Placement in Passwords')
    st.write("""
    Discover where specific special characters are commonly placed within passwords.
    
    Try it yourself:
    Enter a symbol from this set: !@#$%^&*()-_=+[]{|;:'\,.<>?/`~
    
    You'll see a visualization showing how the chosen special character is distributed 
    across different password lengths. This can reveal interesting patterns in how people 
    incorporate special characters into their passwords.
    """)

    special_char_positions_per_char = loaded_statistics.get('special_char_positions_per_char', {})

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

    year_counts = loaded_statistics.get('year_counts', {})
    if not year_counts:
        st.write("No year usage data available.")
        return

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
    """Plot entropy distribution of passwords"""
    st.header('Entropy Distribution')
    st.write("Entropy measures the unpredictability of passwords. Higher entropy indicates stronger passwords.")

    entropies = loaded_statistics.get('entropies', [])

    # Filter the entropies to include only values between 0 and 200
    filtered_entropies = [entropy for entropy in entropies if 0 <= entropy <= 180]

    df_entropy = pd.DataFrame({
        'Entropy': filtered_entropies
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
    elif character_type == 'Lower Case Letters':
        percentages_data = loaded_statistics.get('count_of_lower_case_per_length_per_count_percentages', {})
        title = 'Average Number of Lower Case Letters by Password Length'
        y_label = 'Average Number of Lower Case Letters'
        color_scale = 'Greens'
    elif character_type == 'Special Characters':
        percentages_data = loaded_statistics.get('count_of_special_characters_per_length_per_count_percentages', {})
        title = 'Average Number of Special Characters by Password Length'
        y_label = 'Average Number of Special Characters'
        color_scale = 'Purples'
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
    """Plot average numbers and distributions for upper letters, lower letters, special characters, and numbers for a specific password length."""
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

    # For Lower Case Letters
    percentages_data_lower = loaded_statistics.get('count_of_lower_case_per_length_per_count_percentages', {})
    counts_dict_lower = percentages_data_lower.get(length_str, {})
    expected_lower = 0
    for count_str, percentage in counts_dict_lower.items():
        count = int(count_str)
        expected_lower += count * (percentage / 100)
    data['Average Number of Lower Case Letters'] = expected_lower

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
    plot_distribution_for_length(loaded_statistics, length, 'Lower Case Letters')
    plot_distribution_for_length(loaded_statistics, length, 'Special Characters')

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
    elif character_type == 'Lower Case Letters':
        percentages_data = loaded_statistics.get('count_of_lower_case_per_length_per_count_percentages', {})
        title = f'Distribution of Lower Case Letter Counts for Passwords of Length {length}'
        x_label = 'Number of Lower Case Letters'
        color = 'Greens'
    elif character_type == 'Special Characters':
        percentages_data = loaded_statistics.get('count_of_special_characters_per_length_per_count_percentages', {})
        title = f'Distribution of Special Character Counts for Passwords of Length {length}'
        x_label = 'Number of Special Characters'
        color = 'Purples'
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
        st.write(f"No data available for passwords of length {length} and character type '{character_type}'.")

def static_visualization_page():
    
    st.write("Welcome to our comprehensive Password Dataset Visualization!")
    st.write("This interactive page presents an in-depth analysis of 1 million passwords from the rockyou dataset.")
    st.write("Explore a variety of insightful visualizations that shed light on password creation patterns and potential security implications:")
    st.markdown("""
    - Password length distribution
    - Character type usage (uppercase, lowercase, numbers, special characters)
    - Positional analysis of specific characters within passwords
    - Entropy and strength assessment
    - Common patterns and trends in password creation
    """)
    st.write("Dive into the visualizations below to uncover fascinating trends in password habits and gain valuable insights into cybersecurity practices.")

    # Load the dataset
    dataset_name = 'rockyou2024-1M.txt'  # Update with your dataset name

    loaded_data = load_data(f'{dataset_name}_data_passwords_statistics.json')
    loaded_statistics = loaded_data.get('statistics', {})

    # Plotting functions
    
    plot_password_length_distribution(loaded_statistics)
    plot_ascii_character_usage(loaded_statistics)
    plot_password_categories_distribution(loaded_statistics)
    plot_number_position_violin(loaded_statistics)
    
    plot_special_character_position_violin(loaded_statistics)
    plot_position_of_specific_special_characters(loaded_statistics)
    plot_year_usage(loaded_statistics)
    plot_entropy_distribution(loaded_statistics)

    # New plots
    plot_average_numbers_by_length(loaded_statistics, 'Numbers')
    plot_average_numbers_by_length(loaded_statistics, 'Upper Case Letters')
    plot_average_numbers_by_length(loaded_statistics, 'Lower Case Letters')
    plot_average_numbers_by_length(loaded_statistics, 'Special Characters')

    # Input for specific password length
    st.header('Character Counts for Specific Password Length')
    st.write("Enter a password length to see the average counts and distributions of upper letters, lower letters, special characters, and numbers.")

    length = st.number_input('Password Length:', min_value=1, max_value=30, value=8, step=1)

    plot_average_numbers_for_length(loaded_statistics, length)

if __name__ == '__main__':
    static_visualization_page()
