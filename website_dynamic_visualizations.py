import streamlit as st
import pickle
import plotly.express as px
import pandas as pd
import os
import re

def load_data(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def analyze_password(password):
    """Analyze the user's password and extract relevant information."""
    if password:
        from password_statistics import calculate_entropy
        entropy_value = calculate_entropy(password)
        password_length = len(password)

        # Analyze character types
        has_lower = any(c.islower() for c in password)
        has_upper = any(c.isupper() for c in password)
        has_digit = any(c.isdigit() for c in password)
        has_special = any(not c.isalnum() for c in password)

        # Extract characters
        user_chars = set(password)
        user_special_chars = set(c for c in password if not c.isalnum())
        user_years = set(re.findall(r'(19\d{2}|20\d{2})', password))

        # Extract positions of numbers and special characters
        user_number_positions = []
        user_special_positions = []
        for i, c in enumerate(password):
            if c.isdigit():
                normalized_position = i / len(password)
                user_number_positions.append({
                    'Password Length': len(password),
                    'Normalized Position': normalized_position,
                    'Type': 'User Password'
                })
            if not c.isalnum():
                normalized_position = i / len(password)
                user_special_positions.append({
                    'Password Length': len(password),
                    'Normalized Position': normalized_position,
                    'Type': 'User Password'
                })

        return {
            'entropy_value': entropy_value,
            'password_length': password_length,
            'has_lower': has_lower,
            'has_upper': has_upper,
            'has_digit': has_digit,
            'has_special': has_special,
            'user_chars': user_chars,
            'user_special_chars': user_special_chars,
            'user_years': user_years,
            'user_number_positions': user_number_positions,
            'user_special_positions': user_special_positions
        }
    else:
        return None

def display_password_analysis(password_info):
    """Display the analysis of the user's password."""
    if password_info:
        entropy_value = password_info['entropy_value']
        password_length = password_info['password_length']
        has_lower = password_info['has_lower']
        has_upper = password_info['has_upper']
        has_digit = password_info['has_digit']
        has_special = password_info['has_special']

        st.write(f"Your password length: {password_length}")
        st.write(f"Your password entropy: {entropy_value:.2f} bits")

        st.write("Your password contains:")
        st.write(f"- Lowercase letters: {'Yes' if has_lower else 'No'}")
        st.write(f"- Uppercase letters: {'Yes' if has_upper else 'No'}")
        st.write(f"- Numbers: {'Yes' if has_digit else 'No'}")
        st.write(f"- Special characters: {'Yes' if has_special else 'No'}")
    else:
        st.write("Enter a password above to receive analysis.")

def plot_password_length_distribution(loaded_statistics, password_length):
    """Plot the password length distribution and highlight the user's password length."""
    st.header('Password Length Distribution')
    st.write("Understand the distribution of password lengths.")

    length_percentages = loaded_statistics['length_percentages']
    lengths = list(length_percentages.keys())
    percentages = list(length_percentages.values())

    # Create a DataFrame from the lengths and percentages
    df_length = pd.DataFrame({
        'Password Length': lengths,
        'Percentage': percentages
    })

    # Highlight the user's password length in the plot
    if password_length is not None:
        df_length['Color'] = df_length['Password Length'].apply(lambda x: 'red' if x == password_length else 'blue')
    else:
        df_length['Color'] = 'blue'

    # Create an interactive bar chart using Plotly
    fig_length = px.bar(
        df_length,
        x='Password Length',
        y='Percentage',
        labels={'Password Length': 'Password Length', 'Percentage': 'Percentage'},
        title='Password Length Distribution',
        color='Color',
        color_discrete_map={'red': 'red', 'blue': 'blue'},
        hover_data={'Percentage': ':.2f'}
    )

    st.plotly_chart(fig_length, use_container_width=True)

def plot_ascii_character_usage(loaded_statistics, user_chars):
    """Plot ASCII character usage and highlight user's characters."""
    st.header('ASCII Character Usage')
    st.write("Discover which ASCII characters are most commonly used in passwords.")

    ascii_counts = loaded_statistics['ascii_counts']

    # Create a list of ASCII characters in order
    ascii_order = [chr(i) for i in range(32, 127)]  # ASCII characters from space to '~'
    ascii_percentages = [ascii_counts.get(char, 0) for char in ascii_order]

    # Create a DataFrame for ASCII characters
    df_ascii = pd.DataFrame({
        'ASCII Character': ascii_order,
        'Percentage': ascii_percentages
    })

    # Ensure 'ASCII Character' is treated as a categorical variable with the specified order
    df_ascii['ASCII Character'] = pd.Categorical(df_ascii['ASCII Character'], categories=ascii_order, ordered=True)

    # Highlight the user's password characters in the plot
    if user_chars:
        df_ascii['Color'] = df_ascii['ASCII Character'].apply(lambda x: 'red' if x in user_chars else 'blue')
        st.write(f"Characters in your password: {' '.join(sorted(user_chars))}")
    else:
        df_ascii['Color'] = 'blue'

    # Create an interactive bar chart using Plotly
    fig_ascii = px.bar(
        df_ascii,
        x='ASCII Character',
        y='Percentage',
        labels={'ASCII Character': 'ASCII Character', 'Percentage': 'Percentage'},
        title='ASCII Character Usage in Passwords',
        color='Color',
        color_discrete_map={'red': 'red', 'blue': 'blue'},
        category_orders={'ASCII Character': ascii_order}
    )

    # Update layout for better visualization
    fig_ascii.update_layout(
        xaxis_tickangle=-90,
        xaxis=dict(dtick=1)
    )

    st.plotly_chart(fig_ascii, use_container_width=True)

def plot_special_character_usage(loaded_statistics, user_special_chars):
    """Plot special character usage and highlight user's special characters."""
    st.header('Special Character Usage')
    st.write("Examine the usage frequency of special characters in passwords.")

    special_char_counts = loaded_statistics.get('special_char_counts', {})
    if special_char_counts:
        special_chars = list(special_char_counts.keys())
        special_counts = list(special_char_counts.values())

        # Create a DataFrame for special characters
        df_special = pd.DataFrame({
            'Special Character': special_chars,
            'Count': special_counts
        })

        # Highlight the user's special characters in the plot
        if user_special_chars:
            df_special['Color'] = df_special['Special Character'].apply(lambda x: 'red' if x in user_special_chars else 'blue')
            st.write(f"Special characters in your password: {' '.join(sorted(user_special_chars))}")
        else:
            df_special['Color'] = 'blue'
            st.write("No special characters in your password.")

        # Create an interactive bar chart for special characters
        fig_special = px.bar(
            df_special,
            x='Special Character',
            y='Count',
            labels={'Special Character': 'Special Character', 'Count': 'Count'},
            title='Special Character Usage in Passwords',
            color='Color',
            color_discrete_map={'red': 'red', 'blue': 'blue'}
        )

        st.plotly_chart(fig_special, use_container_width=True)

def plot_year_usage(loaded_statistics, user_years):
    """Plot year usage in passwords and highlight years in user's password."""
    st.header('Year Usage in Passwords')
    st.write("Discover how often years appear in passwords, indicating potential use of dates or birth years.")

    year_counts = loaded_statistics['year_counts']
    years = sorted(year_counts.keys())
    counts = [year_counts[year] for year in years]

    # Create a DataFrame for years
    df_year = pd.DataFrame({
        'Year': years,
        'Count': counts
    })

    # Highlight years present in the user's password
    if user_years:
        df_year['Color'] = df_year['Year'].apply(lambda x: 'red' if str(x) in user_years else 'blue')
        st.write(f"Years in your password: {' '.join(sorted(user_years))}")
    else:
        df_year['Color'] = 'blue'
        st.write("No years found in your password.")

    fig_year = px.bar(
        df_year,
        x='Year',
        y='Count',
        labels={'Year': 'Year', 'Count': 'Count'},
        title='Year Usage in Passwords',
        color='Color',
        color_discrete_map={'red': 'red', 'blue': 'blue'}
    )

    st.plotly_chart(fig_year, use_container_width=True)

def plot_entropy_distribution(loaded_statistics, entropy_value):
    """Plot entropy distribution and indicate user's password entropy."""
    st.header('Entropy Distribution')
    st.write("Entropy measures the unpredictability of passwords. Higher entropy indicates stronger passwords.")

    entropies = loaded_statistics['entropies']

    # Create a DataFrame for entropies
    df_entropy = pd.DataFrame({
        'Entropy': entropies
    })

    # Create an interactive histogram using Plotly
    fig_entropy = px.histogram(
        df_entropy,
        x='Entropy',
        nbins=50,
        labels={'Entropy': 'Entropy (bits)', 'count': 'Count'},
        title='Entropy Distribution of Passwords'
    )

    # Add vertical line and annotation for user's password entropy
    if entropy_value is not None:
        fig_entropy.add_vline(x=entropy_value, line_color='red', line_width=3)
        fig_entropy.add_annotation(
            x=entropy_value,
            y=0,
            text=f"Your entropy: {entropy_value:.2f}",
            showarrow=True,
            arrowhead=1,
            arrowcolor='red',
            arrowsize=1,
            arrowwidth=2,
            ax=0,
            ay=-40
        )

    st.plotly_chart(fig_entropy, use_container_width=True)

def plot_number_position_violin(loaded_statistics, user_number_positions):
    """Plot the position of numbers in passwords."""
    st.header('Position of Numbers in Passwords')
    st.write("Visualize where numbers are commonly placed within passwords.")

    number_positions = loaded_statistics['number_positions']
    if number_positions:
        number_plot_df = pd.DataFrame(number_positions, columns=['Password Length', 'Normalized Position'])
        number_plot_df['Type'] = 'Dataset'

        # Combine user's number positions with dataset
        if user_number_positions:
            user_number_df = pd.DataFrame(user_number_positions)
            combined_df = pd.concat([number_plot_df, user_number_df], ignore_index=True)
        else:
            combined_df = number_plot_df

        fig_violin = px.violin(
            combined_df,
            x='Password Length',
            y='Normalized Position',
            color='Type',
            title='Position of Numbers by Password Length',
            box=True,
            points=False,  # Removed individual data points
            color_discrete_map={'User Password': 'red', 'Dataset': 'blue'},
            hover_data=None  # Removed hover data
        )

        # Update x-axis to show all lengths
        fig_violin.update_layout(
            xaxis=dict(
                tickmode='linear',
                dtick=1  # Tick every 1 unit
            )
        )

        # Remove hover information
        fig_violin.update_traces(hovertemplate=None)

        st.plotly_chart(fig_violin, use_container_width=True)
    else:
        st.write("No number position data available.")

def plot_special_character_position_violin(loaded_statistics, user_special_positions):
    """Plot the position of special characters in passwords."""
    st.header('Position of Special Characters in Passwords')
    st.write("Visualize where special characters are commonly placed within passwords.")

    special_char_positions = loaded_statistics['special_char_positions']
    if special_char_positions:
        special_char_plot_df = pd.DataFrame(special_char_positions, columns=['Password Length', 'Normalized Position'])
        special_char_plot_df['Type'] = 'Dataset'

        # Combine user's special character positions with dataset
        if user_special_positions:
            user_special_df = pd.DataFrame(user_special_positions)
            combined_special_df = pd.concat([special_char_plot_df, user_special_df], ignore_index=True)
        else:
            combined_special_df = special_char_plot_df

        fig_violin_special = px.violin(
            combined_special_df,
            x='Password Length',
            y='Normalized Position',
            color='Type',
            title='Position of Special Characters by Password Length',
            box=True,
            points=False,  # Removed individual data points
            color_discrete_map={'User Password': 'red', 'Dataset': 'blue'},
            hover_data=None  # Removed hover data
        )

        # Update x-axis to show all lengths
        fig_violin_special.update_layout(
            xaxis=dict(
                tickmode='linear',
                dtick=1  # Tick every 1 unit
            )
        )

        # Remove hover information
        fig_violin_special.update_traces(hovertemplate=None)

        st.plotly_chart(fig_violin_special, use_container_width=True)
    else:
        st.write("No special character position data available.")

def plot_specific_special_character_positions(loaded_statistics, user_special_chars):
    """Plot positions of specific special characters used in the user's password."""
    if user_special_chars:
        st.header('Position of Specific Special Characters in Passwords')
        st.write("Visualize where specific special characters are commonly placed within passwords.")

        special_char_positions_per_char = loaded_statistics.get('special_char_positions_per_char', {})
        for special_char in user_special_chars:
            if special_char in special_char_positions_per_char:
                char_positions = special_char_positions_per_char[special_char]
                if char_positions:
                    char_plot_df = pd.DataFrame(char_positions, columns=['Password Length', 'Normalized Position'])
                    fig_char_violin = px.violin(
                        char_plot_df,
                        x='Password Length',
                        y='Normalized Position',
                        title=f'Position of "{special_char}" in Passwords by Length',
                        box=True,
                        points=False
                    )
                    # Update x-axis to show all lengths
                    fig_char_violin.update_layout(
                        xaxis=dict(
                            tickmode='linear',
                            dtick=1  # Tick every 1 unit
                        )
                    )
                    st.plotly_chart(fig_char_violin, use_container_width=True)
                else:
                    st.write(f"No position data available for special character '{special_char}'.")
            else:
                st.write(f"No data available for special character '{special_char}'.")

def plot_special_characters_by_length(loaded_statistics):
    """Plot the average number of special characters used by password length."""
    st.header('Average Number of Special Characters by Password Length')
    st.write("Explore how the average number of special characters used varies with password length.")
    
    # Retrieve the nested dictionary from loaded_statistics
    special_chars_percentages = loaded_statistics.get('count_of_special_characters_per_length_per_count_percentages', {})
    
    if special_chars_percentages:
        data = []
        for length_str, counts_dict in special_chars_percentages.items():
            length = int(length_str)
            expected_special_chars = 0
            for count_str, percentage in counts_dict.items():
                count = int(count_str)
                expected_special_chars += count * (percentage / 100)
            data.append({
                'Password Length': length,
                'Average Number of Special Characters': expected_special_chars
            })
        
        # Create a DataFrame from the list of dictionaries
        df_special_chars = pd.DataFrame(data)
        
        # Sort the DataFrame by Password Length
        df_special_chars = df_special_chars.sort_values('Password Length')
        
        # Create a bar plot
        fig = px.bar(
            df_special_chars,
            x='Password Length',
            y='Average Number of Special Characters',
            labels={'Password Length': 'Password Length', 'Average Number of Special Characters': 'Average Number of Special Characters'},
            title='Average Number of Special Characters by Password Length'
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
        st.write("No data available for number of special characters by password length.")



def plot_password_characteristics_pie(loaded_statistics):
    """Plot the password characteristics distribution as a pie chart."""
    st.header('Password Characteristics Distribution')
    st.write("Visualize the distribution of different password characteristics.")

    # Prepare data for pie chart
    labels = [
        "Lower Case Only", "Upper Case Only", "Numbers Only",
        "Lower and Numbers", "Lower and Special",
        "Upper and Numbers", "Upper and Special", "Lower Upper and Numbers",
        "Special and Numbers", "Lower Upper and Special", "Lower Special and Numbers",
        "Upper Special and Numbers", "All Character Types", "Lower and Upper"
    ]
    percentages = [
        loaded_statistics['lower_case_only_percentage'],
        loaded_statistics['upper_case_only_percentage'],
        loaded_statistics['numbers_only_percentage'],
        loaded_statistics['lower_case_and_numbers_percentage'],
        loaded_statistics['lower_case_and_special_characters_percentage'],
        loaded_statistics['upper_case_and_numbers_percentage'],
        loaded_statistics['upper_case_and_special_characters_percentage'],
        loaded_statistics['lower_upper_case_and_numbers_percentage'],
        loaded_statistics['special_characters_and_numbers_percentage'],
        loaded_statistics['lower_upper_case_and_special_characters_percentage'],
        loaded_statistics['lower_special_characters_and_numbers_percentage'],
        loaded_statistics['upper_special_characters_and_numbers_percentage'],
        loaded_statistics['all_character_types_percentage'],
        loaded_statistics['lower_case_and_upper_case_percentage']
    ]

    # Create a DataFrame for the pie chart
    df_pie = pd.DataFrame({
        'Characteristics': labels,
        'Percentage': percentages
    })

    # Since the pie chart represents categories, we won't highlight user data here

    # Create an interactive pie chart using Plotly
    fig_pie = px.pie(
        df_pie,
        names='Characteristics',
        values='Percentage',
        title='Password Characteristics Distribution',
        hole=0.4
    )

    fig_pie.update_traces(textposition='inside', textinfo='percent+label')

    st.plotly_chart(fig_pie, use_container_width=True)

def display_password_strength_feedback(entropy_value):
    """Provide feedback on the user's password strength."""
    st.header('Password Strength Feedback')

    if entropy_value is not None:
        # Provide feedback
        if entropy_value < 50:
            st.warning("Your password has low entropy. Consider making it longer and more complex.")
        elif entropy_value < 80:
            st.info("Your password has moderate entropy. Adding more unique characters can strengthen it.")
        else:
            st.success("Your password has high entropy!")
    else:
        st.write("Enter a password above to receive feedback on its strength.")

def dynamic_visualization_page():
    st.title('Interactive Password Statistics Dashboard')
    st.write("Explore various statistics of passwords in the dataset with interactive visualizations.")

    # Load the dataset
    dataset_name = 'rockyou2024-100K.txt'  # Update with your dataset name

    loaded_data = load_data(f'{dataset_name}_data_passwords_statistics.pkl')
    loaded_statistics = loaded_data['statistics']

    # **Password Input**
    password = st.text_input("Enter your password for analysis", type="password", key='user_password')

    # **Password Analysis**
    password_info = analyze_password(password)
    display_password_analysis(password_info)

    if password_info:
        user_chars = password_info['user_chars']
        user_special_chars = password_info['user_special_chars']
        user_years = password_info['user_years']
        password_length = password_info['password_length']
        entropy_value = password_info['entropy_value']
        user_number_positions = password_info['user_number_positions']
        user_special_positions = password_info['user_special_positions']
    else:
        user_chars = set()
        user_special_chars = set()
        user_years = set()
        password_length = None
        entropy_value = None
        user_number_positions = []
        user_special_positions = []

    # Plotting functions
    plot_password_length_distribution(loaded_statistics, password_length)
    plot_ascii_character_usage(loaded_statistics, user_chars)
    plot_special_character_usage(loaded_statistics, user_special_chars)
    plot_year_usage(loaded_statistics, user_years)
    plot_entropy_distribution(loaded_statistics, entropy_value)
    plot_number_position_violin(loaded_statistics, user_number_positions)
    plot_special_character_position_violin(loaded_statistics, user_special_positions)
    if password_info and user_special_chars:
        plot_specific_special_character_positions(loaded_statistics, user_special_chars)
    plot_special_characters_by_length(loaded_statistics)
    plot_password_characteristics_pie(loaded_statistics)
    display_password_strength_feedback(entropy_value)

