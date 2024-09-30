import json
import streamlit as st
import pickle
import plotly.express as px
import pandas as pd
import os
import re
import plotly.graph_objects as go


def load_data(filename):
    """Load data from a JSON file."""
    with open(filename, 'r') as f:
        return json.load(f)

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
        pass
        # st.write("Enter a password above to receive analysis.")

def plot_password_length_distribution(loaded_statistics, password_length):
    """Plot the password length distribution and highlight the user's password length."""
    st.header('Password Length Distribution')
    st.write("Here we show the distribution of different password lengths. The graph displays the percentage of passwords for each length. It also compares to the length of the password you entered.")

    length_percentages = loaded_statistics['length_percentages']
    lengths = list(map(int, length_percentages.keys()))  # Ensure lengths are integers
    percentages = list(length_percentages.values())

    # Create a DataFrame from the lengths and percentages
    df_length = pd.DataFrame({
        'Password Length': lengths,
        'Percentage': percentages
    })

    # Highlight the user's password length in the plot
    if password_length is not None:
        df_length['Color'] = ['Your Password Length' if length == password_length else 'Other Lengths' for length in df_length['Password Length']]
    else:
        df_length['Color'] = 'Other Lengths'

    # Create an interactive bar chart using Plotly with more eye-pleasant colors
    fig_length = px.bar(
        df_length,
        x='Password Length',
        y='Percentage',
        labels={'Password Length': 'Password Length', 'Percentage': 'Percentage'},
        title='Password Length Distribution',
        color='Color',
        color_discrete_map={'Your Password Length': '#FA8072', 'Other Lengths': '#4682B4'},
        hover_data={'Percentage': ':.2f'}
    )

    # Update layout to improve visualization
    fig_length.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        paper_bgcolor='white'
    )

    st.plotly_chart(fig_length, use_container_width=True)

def plot_ascii_character_usage(loaded_statistics, user_chars):
    """Plot ASCII character usage and highlight user's characters."""
    st.header('ASCII character usage in passwords')
    st.write("This graph shows the distribution of each ASCII character in the dataset. We can see which characters appear in the password you entered.")

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
        df_ascii['Color'] = df_ascii['ASCII Character'].apply(lambda x: 'Characters in your password' if x in user_chars else 'Other Characters')
        st.write(f"Characters in your password: {' '.join(sorted(user_chars))}")
    else:
        df_ascii['Color'] = 'Other Characters'

    # Create an interactive bar chart using Plotly with more eye-pleasant colors
    fig_ascii = px.bar(
        df_ascii,
        x='ASCII Character',
        y='Percentage',
        labels={'ASCII Character': 'ASCII Character', 'Percentage': 'Percentage'},
        title='ASCII Character Usage in Passwords',
        color='Color',
        color_discrete_map={'Characters in your password': '#FA8072', 'Other Characters': '#4682B4'},
        category_orders={'ASCII Character': ascii_order}
    )

    # Update layout for better visualization
    fig_ascii.update_layout(
        xaxis_tickangle=-90,
        xaxis=dict(dtick=1),
        paper_bgcolor='white'
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

def plot_year_usage_histogram(loaded_statistics, user_years):
    """
    Plot year usage in passwords as a histogram.
    Highlights the year present in the user's password in red.
    
    Args:
        loaded_statistics (dict): The statistics dictionary loaded from JSON.
        user_years (set): Set of years detected in the user's password.
    """
    if user_years:
        st.header('Year Usage in Passwords')
        st.write("Discover how often years appear in passwords, indicating potential use of dates or birth years.")
        
        year_counts = loaded_statistics.get('year_counts', {})
        if not year_counts:
            st.write("No year usage data available.")
            return
        
        years = sorted(year_counts.keys())
        counts = [year_counts[year] for year in years]
    
        # Create a DataFrame for years
        df_year = pd.DataFrame({
            'Year': years,
            'Count': counts
        })
    
        # Highlight years present in the user's password
        df_year['Color'] = df_year['Year'].apply(lambda x: 'red' if str(x) in user_years else 'blue')
    
        # Create the histogram using Plotly Graph Objects
        fig_year = go.Figure()
    
        # Add bars for each year
        for _, row in df_year.iterrows():
            fig_year.add_trace(go.Bar(
                x=[row['Year']],
                y=[row['Count']],
                marker_color=row['Color'],
                name=row['Year']
            ))
    
        fig_year.update_layout(
            title='Year Usage in Passwords Histogram',
            xaxis_title='Year',
            yaxis_title='Count',
            showlegend=False,
            height=600,
            width=800
        )
    
        st.plotly_chart(fig_year, use_container_width=True)
    else:
        st.write("No years detected in your password. Year usage plot will not be displayed.")

def plot_entropy_distribution(loaded_statistics, entropy_value):
    """Plot entropy distribution and indicate user's password entropy."""
    st.header('How Strong is Your Password?')
    st.write("Let's see how your password stacks up against others in terms of strength!\n\n"
             "We use a measure called 'entropy' to gauge password strength. Think of it like a strength meter - "
             "the higher the entropy, the stronger and more unpredictable your password is.\n\n"
             "The graph below shows how your password compares to others in our database."
             "Can you spot where your password falls on the strength spectrum?")

    entropies = loaded_statistics['entropies']

    # Filter the entropies to include only values between 0 and 200
    filtered_entropies = [entropy for entropy in entropies if 0 <= entropy <= 180]

    # Create a DataFrame for filtered entropies
    df_entropy = pd.DataFrame({
        'Entropy': filtered_entropies
    })

    # Create an interactive histogram using Plotly
    fig_entropy = px.histogram(
        df_entropy,
        x='Entropy',
        nbins=50,
        labels={'Entropy': 'Entropy (bits)', 'count': 'Count'},
        title='Entropy Distribution of Passwords'
    )

    # Add vertical line and annotation for user's password entropy, if it's in the range of 0-200
    if entropy_value is not None and 0 <= entropy_value <= 180:
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

def categorize_password(password, special_chars):
    """
    Determine the category of the password based on its characteristics.
    Returns the category name as a string.
    """
    has_lower = any(c.islower() for c in password)
    has_upper = any(c.isupper() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in special_chars for c in password)
    
    # Define category based on presence of character types
    if has_lower and not has_upper and not has_digit and not has_special:
        return "Lower Case Only"
    elif has_upper and not has_lower and not has_digit and not has_special:
        return "Upper Case Only"
    elif has_digit and not has_lower and not has_upper and not has_special:
        return "Numbers Only"
    elif has_lower and has_digit and not has_upper and not has_special:
        return "Lower and Numbers"
    elif has_lower and has_special and not has_upper and not has_digit:
        return "Lower and Special"
    elif has_upper and has_digit and not has_lower and not has_special:
        return "Upper and Numbers"
    elif has_upper and has_special and not has_lower and not has_digit:
        return "Upper and Special"
    elif has_special and has_digit and not has_lower and not has_upper:
        return "Special and Numbers"
    elif has_lower and has_upper and has_digit and not has_special:
        return "Lower Upper and Numbers"
    elif has_lower and has_upper and has_special and not has_digit:
        return "Lower Upper and Special"
    elif has_lower and has_special and has_digit and not has_upper:
        return "Lower Special and Numbers"
    elif has_upper and has_special and has_digit and not has_lower:
        return "Upper Special and Numbers"
    elif has_lower and has_upper and has_digit and has_special:
        return "All Character Types"
    elif has_lower and has_upper and not has_digit and not has_special:
        return "Lower and Upper"
    
    return "Uncategorized"


def plot_password_categories_distribution(loaded_statistics, user_password, special_chars):
    """
    Plot password categories distribution as a treemap.
    Highlights the category of the user-inputted password in red.
    
    Args:
        loaded_statistics (dict): The statistics dictionary loaded from JSON.
        user_password (str): The password input by the user.
        special_chars (str): String containing special characters.
    """
    st.header('Password Categories Distribution')
    st.write("This treemap visualization shows the distribution of different password categories based on their character composition.")
    st.write("Your password's category will be highlighted in red, allowing you to see which group it belongs to and what percentage of passwords fall into this category.")
    
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
    
    # Determine the category of the user-inputted password
    if user_password:
        user_category = categorize_password(user_password, special_chars)
        st.write(f"**Your password is categorized as:** {user_category}")
        if user_category not in categories:
            st.warning(f"The input password does not fit into any predefined category. It is categorized as 'Uncategorized'.")
            # Add 'Uncategorized' to your categories with percentage 0
            df_categories = df_categories.append({'Category': 'Uncategorized', 'Percentage': 0}, ignore_index=True)
    else:
        user_category = None

    # Assign colors: red for the user's category, blue scale for others
    def assign_colors_graph_objects(df, highlight_category):
        colors = []
        blues = px.colors.sequential.Blues
        # Get the maximum percentage to normalize the blue scale
        max_percentage = df['Percentage'].max() if df['Percentage'].max() > 0 else 1
        for _, row in df.iterrows():
            if row['Category'] == highlight_category:
                colors.append('red')
            else:
                # Normalize the percentage to get a shade of blue
                normalized = row['Percentage'] / max_percentage
                normalized = max(0, min(normalized, 1))  # Ensure normalized is between 0 and 1
                # Select a color from Blues scale based on normalized value
                color_index = int(normalized * (len(blues) - 1))
                colors.append(blues[color_index])
        return colors

    # Get the color list
    color_list = assign_colors_graph_objects(df_categories, user_category if user_category != "Uncategorized" else None)

    # Assign colors
    df_categories['Custom_Color'] = color_list

    # Create the treemap using Plotly Graph Objects
    fig = go.Figure(go.Treemap(
        labels=df_categories['Category'],
        parents=[""] * len(df_categories),  # All categories have no parent
        values=df_categories['Percentage'],
        marker=dict(colors=df_categories['Custom_Color']),
        textinfo='label+percent parent'
    ))

    fig.update_layout(
        title='Password Categories Distribution Treemap',
        height=800,
        width=1000
    )

    st.plotly_chart(fig, use_container_width=True)


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


import streamlit as st
import json
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
from password_statistics import calculate_entropy
from datasketch import MinHash
import math
import os
import numpy as np
from sklearn.manifold import MDS

# Function to load n-gram probabilities from JSON
def load_ngram_probabilities(ngram_prob_path):
    with open(ngram_prob_path, 'r', encoding='utf-8') as file:
        ngram_probs = json.load(file)
    return ngram_probs

def calculate_ngram_log_likelihood(s, ngram_probs, n=2, smoothing=1e-10):
    log_likelihood_value = 0.0
    len_s = len(s)
    for i in range(len_s - n + 1):
        prev_ngram = s[i:i+n-1]
        next_char = s[i+n-1]
        # Access the nested dictionary directly
        next_char_probs = ngram_probs.get(prev_ngram)
        if next_char_probs is not None:
            prob = next_char_probs.get(next_char, smoothing)
        else:
            prob = smoothing
        log_likelihood_value += math.log(prob)
    return log_likelihood_value

def find_nearest_ngram_cluster(password_ll, ngram_clusters):
    """
    Find the nearest n-gram cluster based on log-likelihood.
    """
    nearest_cluster_key = None
    min_diff = float('inf')
    
    for cluster_key, cluster_info in ngram_clusters.items():
        try:
            cluster_ll = float(cluster_info.get('Average Log Likelihood', 0))
        except (ValueError, TypeError):
            continue
        diff = abs(cluster_ll - password_ll)
        if diff < min_diff:
            min_diff = diff
            nearest_cluster_key = cluster_key
    
    if nearest_cluster_key is None:
        st.error("No n-gram clusters available.")
        return None
    
    return nearest_cluster_key

def find_and_show_closest_ngram_cluster(input_string, clusters, ngram_probs, top_n=10, n=2, smoothing=1e-10):
    input_ll = calculate_ngram_log_likelihood(input_string.lower(), ngram_probs, n, smoothing)
    closest_cluster_key = find_nearest_ngram_cluster(input_ll, clusters)

    if closest_cluster_key is None:
        st.error("No clusters found for the given password.")
        return [], None

    closest_cluster = clusters[closest_cluster_key]

    # Extract passwords from the cluster
    passwords_list = closest_cluster.get('Passwords', [])
    extracted_passwords = []
    for entry in passwords_list:
        if isinstance(entry, dict) and 'Password' in entry:
            extracted_passwords.append(entry['Password'])
        elif isinstance(entry, str):
            extracted_passwords.append(entry)
        else:
            continue

    if not extracted_passwords:
        st.warning("No passwords found in the nearest n-gram cluster.")
        return [], closest_cluster.get('Average Log Likelihood', 0)

    # Find top N closest passwords based on log-likelihood difference
    distances = []
    for password in extracted_passwords:
        password_ll = calculate_ngram_log_likelihood(password.lower(), ngram_probs, n, smoothing)
        ll_diff = abs(input_ll - password_ll)
        distances.append((password, ll_diff))

    # Sort by smallest difference
    distances.sort(key=lambda x: x[1])
    return [password for password, _ in distances[:top_n]], closest_cluster.get('Average Log Likelihood', 0)

def create_ngram_cluster_dataframe(clusters, user_cluster_key, ngram_probs, n=2):
    cluster_data = []
    for cluster_key, cluster_info in clusters.items():
        try:
            cluster_ll = float(cluster_info.get('Average Log Likelihood', 0))
        except (ValueError, TypeError):
            continue
        
        is_user_cluster = (cluster_key == user_cluster_key)

        # Extract sample passwords
        passwords_list = cluster_info.get('Passwords', [])[:3]
        sample_passwords = []
        for entry in passwords_list:
            if isinstance(entry, dict) and 'Password' in entry:
                sample_passwords.append(entry['Password'])
            elif isinstance(entry, str):
                sample_passwords.append(entry)
            else:
                continue

        sample_passwords = sample_passwords if sample_passwords else ['No Passwords']

        cluster_data.append({
            'Log Likelihood': cluster_ll,
            'Cluster Size': len(cluster_info.get('Passwords', [])),
            'User Cluster': is_user_cluster,
            'Sample Passwords': '<br>'.join(sample_passwords)
        })

    df = pd.DataFrame(cluster_data)
    if df.empty:
        st.warning("No clusters available for visualization.")
        return df

    df = df.sort_values(by='Log Likelihood')
    return df

def visualize_ngram_clusters(df, threshold):
    """
    Create a scatter plot for n-gram clusters, highlighting the user's cluster,
    and include the threshold as a vertical line.
    """
    fig = px.scatter(
        df,
        x='Log Likelihood',
        y='Cluster Size',
        title=f"Password Clusters by N-gram Log-Likelihood (Threshold: {threshold:.2f})",
        labels={'Log Likelihood': 'N-gram Log-Likelihood', 'Cluster Size': 'Number of Passwords'},
        size='Cluster Size',
        size_max=20,
        color='User Cluster',
        color_discrete_map={True: 'red', False: 'blue'},
        hover_name='Log Likelihood',
        hover_data={'Cluster Size': True, 'Sample Passwords': True}
    )

    # Update hover template
    fig.update_traces(
        hovertemplate=(
            '<b>N-gram Log-Likelihood:</b> %{x:.2f}<br>'
            '<b>Cluster Size:</b> %{y}<br>'
            '<b>Sample Passwords:</b><br>%{customdata[0]}<br>'
            '<extra></extra>'
        ),
        customdata=df[['Sample Passwords']].values
    )

    # Add a vertical line for the threshold
    fig.add_vline(
        x=threshold,
        line_color='red',
        line_dash="dash",
        annotation_text="Threshold",
        annotation_position="top right"
    )

    return fig

# Function to find the nearest entropy cluster
def find_nearest_entropy_cluster(password_entropy, clusters):
    """Find the nearest cluster by entropy value."""
    nearest_cluster = None
    min_diff = float('inf')

    for entropy in clusters.keys():
        entropy_value = float(entropy)
        diff = abs(entropy_value - password_entropy)
        if diff < min_diff:
            min_diff = diff
            nearest_cluster = entropy_value
    return nearest_cluster

# Show Closest Cluster and Find Top N Closest Passwords
def find_and_show_closest_cluster(input_string, clusters, top_n=10):
    input_entropy = calculate_entropy(input_string)
    closest_cluster_entropy = None
    closest_cluster = None
    min_entropy_diff = float('inf')

    # Step 1: Find the closest cluster by calculating the absolute entropy difference
    for cluster_entropy, cluster_members in clusters.items():
        entropy_diff = abs(input_entropy - float(cluster_entropy))
        if entropy_diff < min_entropy_diff:
            min_entropy_diff = entropy_diff
            closest_cluster_entropy = float(cluster_entropy)
            closest_cluster = cluster_members

    # Step 2: Return the top N closest passwords within the closest cluster
    if closest_cluster is not None:
        distances = []
        for password in closest_cluster:
            password_entropy = calculate_entropy(password)
            entropy_diff = abs(input_entropy - password_entropy)
            distances.append((password, entropy_diff))

        # Sort passwords by entropy difference and return the top N closest ones
        distances.sort(key=lambda x: x[1])
        return [password for password, _ in distances[:top_n]], closest_cluster_entropy

    return [], None  # If no cluster found

# Function to create a MinHash for a given string
def get_minhash(string, num_perm=128):
    """Generate a MinHash signature for the input string."""
    minhash = MinHash(num_perm=num_perm)
    for char in set(string):  # Treat the string as a set of unique characters
        minhash.update(char.encode('utf8'))
    return minhash

# Function to find the closest cluster by MinHash similarity
def find_and_show_closest_minhash_cluster(input_string, clusters, top_n=10, num_perm=128):
    """
    Find the closest cluster by MinHash similarity and return the top N closest passwords.
    """
    input_minhash = get_minhash(input_string, num_perm=num_perm)
    closest_cluster = None
    max_similarity = -1  # Similarity starts from 0 and goes up to 1

    # Step 1: Compare input MinHash with each cluster's MinHash to find the closest one
    for cluster_label, cluster_members in clusters.items():
        if not cluster_members:
            continue  # Skip empty clusters
        # Use the first password to represent the cluster
        cluster_minhash = get_minhash(cluster_members[0], num_perm=num_perm)
        similarity = input_minhash.jaccard(cluster_minhash)
        
        if similarity > max_similarity:
            max_similarity = similarity
            closest_cluster = cluster_members

    # Step 2: Sort passwords within the closest cluster based on MinHash similarity
    distances = []
    if closest_cluster:
        for password in closest_cluster:
            password_minhash = get_minhash(password, num_perm=num_perm)
            similarity = input_minhash.jaccard(password_minhash)
            distances.append((password, similarity))
    
        # Sort passwords by similarity (highest to lowest) and return the top N closest ones
        distances.sort(key=lambda x: x[1], reverse=True)
        return [password for password, _ in distances[:top_n]], max_similarity

    return [], None

def find_closest_minhash_cluster_label(input_string, clusters, num_perm=128):
    """
    Find the label of the closest cluster by MinHash similarity.
    """
    input_minhash = get_minhash(input_string, num_perm=num_perm)
    closest_cluster_label = None
    max_similarity = -1  # Similarity starts from 0 and goes up to 1

    # Step 1: Compare input MinHash with each cluster's MinHash to find the closest one
    for cluster_label, cluster_members in clusters.items():
        if not cluster_members:
            continue  # Skip empty clusters
        # Use the first password to represent the cluster
        cluster_minhash = get_minhash(cluster_members[0], num_perm=num_perm)
        similarity = input_minhash.jaccard(cluster_minhash)
        
        if similarity > max_similarity:
            max_similarity = similarity
            closest_cluster_label = cluster_label

    return closest_cluster_label, max_similarity

def plot_entropy(password, entropy_clusters):
    # Calculate the entropy of the input password
    password_entropy = calculate_entropy(password)

    # Find the nearest entropy cluster and the closest passwords
    nearest_entropy_cluster = find_nearest_entropy_cluster(password_entropy, entropy_clusters)
    closest_passwords, closest_entropy = find_and_show_closest_cluster(password, entropy_clusters)

    # Prepare data for Entropy visualization
    entropy_cluster_data = []
    for entropy, passwords in entropy_clusters.items():
        is_user_cluster = (float(entropy) == nearest_entropy_cluster)
        sample_passwords = passwords[:3] if len(passwords) > 0 else ['No Passwords']
        entropy_cluster_data.append({
            'Entropy': float(entropy),
            'Cluster Size': len(passwords),
            'User Cluster': is_user_cluster,
            'Sample Passwords': '<br>'.join(sample_passwords)
        })

    # Create a DataFrame for Entropy Clusters
    df_entropy = pd.DataFrame(entropy_cluster_data)
    df_entropy = df_entropy.sort_values(by='Entropy')

    # Create an interactive scatter plot for Entropy Clusters
    fig_entropy = px.scatter(
        df_entropy,
        x='Entropy',
        y='Cluster Size',
        title="Password Clusters by Entropy",
        labels={'Entropy': 'Entropy Value', 'Cluster Size': 'Number of Passwords'},
        size='Cluster Size',
        size_max=20,
        color='User Cluster',
        color_discrete_map={True: 'red', False: 'blue'}
    )

    # Update hover template
    fig_entropy.update_traces(
        hovertemplate=('<b>Entropy:</b> %{x:.2f}<br>'
                       '<b>Cluster Size:</b> %{y}<br>'
                       '<b>Sample Passwords:</b><br>%{customdata[0]}<br>'
                       '<extra></extra>'),
        customdata=df_entropy[['Sample Passwords']].values
    )

    # Show the entropy cluster plot
    st.plotly_chart(fig_entropy, use_container_width=True)

    # Display the entropy of the input password
    st.subheader(f"Your password's entropy: **{password_entropy:.2f}**")
    st.write(f"Assigned to entropy cluster: {nearest_entropy_cluster:.2f}")

    # Display the top 10 closest passwords with their entropy values
    st.subheader("Top 10 Closest Passwords by Entropy")
    if closest_passwords:
        for i, p in enumerate(closest_passwords, 1):
            if p == password:
                st.markdown(f"**{i}. {p} (Entropy: {calculate_entropy(p):.2f})** - **Your password**")
            else:
                st.write(f"{i}. {p} (Entropy: {calculate_entropy(p):.2f})")
    else:
        st.write("No similar passwords found in the nearest entropy cluster.")


def plot_minhash(password, minhash_clusters):
    closest_passwords_minhash, max_similarity_minhash = find_and_show_closest_minhash_cluster(
        password, minhash_clusters
    )

    # Display MinHash similarity for the user's password
    st.subheader(f"Max MinHash similarity: **{max_similarity_minhash:.2f}**")

    # Display the top 10 closest passwords by MinHash similarity
    st.subheader("Top 10 Closest Passwords by MinHash Similarity")
    if closest_passwords_minhash:
        for i, p in enumerate(closest_passwords_minhash, 1):
            minhash_sim = get_minhash(password).jaccard(get_minhash(p))
            if p == password:
                st.markdown(f"**{i}. {p} (MinHash Similarity: {minhash_sim:.2f})** - **Your password**")
            else:
                st.write(f"{i}. {p} (MinHash Similarity: {minhash_sim:.2f})")
    else:
        st.write("No similar passwords found in the nearest MinHash cluster.")


def plot_ngram(password, ngram_clusters_json, ngram_probs):
    threshold = ngram_clusters_json.get('Threshold', None)
    ngram_clusters = ngram_clusters_json.get('Clusters', {})
    n = 2  # Define n for n-grams

    # Calculate n-gram log-likelihood and find the nearest n-gram cluster
    password_ngram_ll = calculate_ngram_log_likelihood(password, ngram_probs, n, smoothing=1e-10)
    nearest_ngram_cluster_key = find_nearest_ngram_cluster(password_ngram_ll, ngram_clusters)
    closest_ngram_passwords, closest_ngram_ll = find_and_show_closest_ngram_cluster(
        password, ngram_clusters, ngram_probs, top_n=10, n=2, smoothing=1e-10
    )

    # Prepare data for visualization
    ngram_df = create_ngram_cluster_dataframe(ngram_clusters, nearest_ngram_cluster_key, ngram_probs, n=2)

    # Visualize n-gram clusters if data is available
    if not ngram_df.empty:
        ngram_fig = visualize_ngram_clusters(ngram_df, threshold)
        st.plotly_chart(ngram_fig, use_container_width=True)
    else:
        st.warning("No n-gram clusters available for visualization.")

    # Display the log-likelihood of the input password
    st.subheader(f"Your password's n-gram log-likelihood: **{password_ngram_ll:.2f}**")
    st.write(f"Assigned to n-gram cluster with Average Log Likelihood: {closest_ngram_ll:.2f}")

    # Display the top 10 closest passwords by n-gram log-likelihood
    st.subheader("Top 10 Closest Passwords by N-gram Log-Likelihood")
    if closest_ngram_passwords:
        for i, p in enumerate(closest_ngram_passwords, 1):
            ngram_ll = calculate_ngram_log_likelihood(p, ngram_probs, n, smoothing=1e-10)
            if p == password:
                st.markdown(f"**{i}. {p} (Log-Likelihood: {ngram_ll:.2f})** - **Your password**")
            else:
                st.write(f"{i}. {p} (Log-Likelihood: {ngram_ll:.2f})")
    else:
        st.write("No similar passwords found in the nearest n-gram cluster.")


# New functions for loading and visualizing clusters with similarities
def load_clusters_and_similarities(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    clusters = data['Clusters']
    similarities = data['Cluster Similarities']
    return clusters, similarities

def get_top_k_clusters(clusters, similarities, top_k=10, include_clusters=None):
    # Sort clusters by size
    sorted_clusters = sorted(clusters.items(), key=lambda x: len(x[1]), reverse=True)
    # Ensure include_clusters are in the list
    if include_clusters is not None:
        # Move included clusters to the front
        included = [item for item in sorted_clusters if item[0] in include_clusters]
        others = [item for item in sorted_clusters if item[0] not in include_clusters]
        sorted_clusters = included + others

    # Take the top K clusters
    top_clusters = sorted_clusters[:top_k]

    # Now, extract the data
    cluster_names = [cluster[0] for cluster in top_clusters]
    cluster_sizes = np.array([len(cluster[1]) for cluster in top_clusters])
    cluster_examples = [cluster[1][:3] for cluster in top_clusters]

    # Build the similarity matrix
    num_clusters = len(cluster_names)
    similarity_matrix = np.ones((num_clusters, num_clusters))

    for i, cluster_i in enumerate(cluster_names):
        for j, cluster_j in enumerate(cluster_names):
            if i != j:
                pair_key = f"{cluster_i} vs {cluster_j}"
                similarity = similarities.get(pair_key, None)
                if similarity is not None:
                    similarity_matrix[i, j] = 1 - similarity  # Convert similarity to distance
                else:
                    similarity_matrix[i, j] = 1  # Max distance if similarity not found
    similarity_matrix = make_symmetric(similarity_matrix)
    return cluster_names, cluster_sizes, cluster_examples, similarity_matrix

def make_symmetric(matrix):
    """Ensure the matrix is symmetric."""
    sym_matrix = np.copy(matrix)
    for i in range(len(matrix)):
        for j in range(i + 1, len(matrix)):
            sym_matrix[j, i] = sym_matrix[i, j]
    # Ensure diagonal is 0 (as it represents the distance of a cluster with itself)
    np.fill_diagonal(sym_matrix, 0)
    return sym_matrix

def visualize_clusters(cluster_names, cluster_sizes, cluster_examples, similarity_matrix, user_cluster_label=None):
    # Perform multidimensional scaling (MDS) to convert similarity matrix to 2D positions
    mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42)
    positions = mds.fit_transform(similarity_matrix)

    # Calculate marker sizes so that marker areas are proportional to cluster sizes
    scaling_factor = 2  # Adjust this value to scale marker sizes for better visualization
    marker_sizes = np.sqrt(cluster_sizes) * scaling_factor

    # Create a DataFrame for plotting
    plot_df = pd.DataFrame({
        'x': positions[:, 0],
        'y': positions[:, 1],
        'Cluster Name': cluster_names,
        'Cluster Size': cluster_sizes,
        'Examples': cluster_examples,
    })

    # Determine colors
    plot_df['Color'] = plot_df['Cluster Name'].apply(
        lambda x: 'User Cluster' if x == user_cluster_label else 'Other Clusters'
    )

    # Create hover text
    plot_df['Hover Text'] = plot_df.apply(
        lambda row: f"<b>{row['Cluster Name']}</b><br>Size: {row['Cluster Size']}<br><b>Examples:</b><br>{'<br>'.join(row['Examples'])}",
        axis=1
    )

    # Create a scatter plot using Plotly Express
    fig = px.scatter(
        plot_df,
        x='x',
        y='y',
        size='Cluster Size',
        size_max=20,
        color='Color',
        color_discrete_map={'User Cluster': 'red', 'Other Clusters': 'blue'},
        hover_name='Cluster Name',
        hover_data={'Cluster Size': True, 'Examples': True},
    )

    # Update hover template to use the custom hover text
    fig.update_traces(
        hovertemplate='%{customdata}<extra></extra>',
        customdata=plot_df['Hover Text']
    )

    # Update layout for aesthetics
    fig.update_layout(
        title="Visualization of Top Clusters by Size and Similarity",
        xaxis_title="MDS Dimension 1",
        yaxis_title="MDS Dimension 2",
        template="plotly_white",
        width=900,
        height=700,
        showlegend=False  # Hide legend if desired
    )

    # Show the interactive plot using Streamlit
    st.plotly_chart(fig, use_container_width=True)

def dynamic_visualization_page():

   # Load the dataset
    dataset_name = 'rockyou2024-1M.txt'  # Update with your dataset name

    loaded_data = load_data(f'{dataset_name}_data_passwords_statistics.json')
    loaded_statistics = loaded_data['statistics']
    
    st.write("""
    Try our Password Statistical Analysis Tool!\n
    Here, you can enter a password to see how it compares to 1 million other passwords from the RockYou 2024 dataset,
    a collection of real-world passwords from various data breaches.\n
    We'll provide you with insights on how to improve your password's strength and uniqueness.
    You'll also receive an entropy score, which measures the randomness and unpredictability of your password.\n
    Don't worry - we won't store your password. All analysis is done locally.\n
    After analysis, you'll see a graph showing how your password's entropy compares to others.
    Use these insights to create a stronger, more secure password that's less likely to be guessed or hacked.
    """)

     # Page for inserting a password
    st.subheader('Insert Password for Analysis')
    st.write("Please enter your password below.")
    password = st.text_input("Enter a password for analysis",
                             help="Type your password here to see how it compares to common length patterns")
    st.caption("Your password will be analyzed locally and not stored or transmitted.")

    if not password:
        st.write("Enter a password to analyze.")
        return

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
    plot_password_categories_distribution(loaded_statistics, password, special_chars="!@#$%^&*()-_=+[]{|;:'\,.<>?/`~ }")
    # plot_special_character_usage(loaded_statistics, user_special_chars)

    plot_year_usage_histogram(loaded_statistics, user_years)
    plot_entropy_distribution(loaded_statistics, entropy_value)
    display_password_strength_feedback(entropy_value)
    
    # plot_number_position_violin(loaded_statistics, user_number_positions)
    # plot_special_character_position_violin(loaded_statistics, user_special_positions)
    # if password_info and user_special_chars:
    #     plot_specific_special_character_positions(loaded_statistics, user_special_chars)
    # plot_special_characters_by_length(loaded_statistics)

    st.header('Password Clustering Analysis')
    st.write("""
    In this analysis, we use three different techniques—entropy, n-gram log-likelihood, and MinHash—to cluster passwords based on their statistical properties.
    Each method captures a different aspect of password structure, allowing us to better understand password strength and predictability.
    By inputting your password, you will be able to compare it to known password clusters and find its closest matches based on each of these clustering techniques.
    """)

    json_files_path = ''
    entropy_json_name = 'entropy_clusters.json'
    minhash_json_name = 'minhash_clusters.json'
    ngram_prob_name = 'ngram_probs.json'
    ngram_clusters_name = 'ngram_clusters.json'
    minhash_similarity_json_name = 'minhash_clusters_with_similarity.json'

    entropy_json_path = os.path.join(json_files_path, entropy_json_name)
    ngram_clusters_path = os.path.join(json_files_path, ngram_clusters_name)
    ngram_prob_path = os.path.join(json_files_path, ngram_prob_name)
    minhash_json_path = os.path.join(json_files_path, minhash_json_name)
    minhash_similarity_json_path = os.path.join(json_files_path, minhash_similarity_json_name)

    # Load JSON data
    with open(entropy_json_path, 'r', encoding='utf-8') as json_file:
        entropy_clusters = json.load(json_file)

    with open(minhash_json_path, 'r', encoding='utf-8') as json_file:
        minhash_clusters = json.load(json_file)

    with open(ngram_clusters_path, 'r', encoding='utf-8') as json_file:
        ngram_clusters_json = json.load(json_file)

    """ Entropy Clustering """
    st.subheader('Clustering by Entropy')
    st.write("""
    **Entropy** is a critical metric for assessing password strength. It measures the unpredictability or randomness in a password, which directly correlates with how difficult it is to guess. Entropy is calculated based on two key factors:

    1. **Password Length**: Longer passwords have higher entropy because they offer more potential combinations of characters.
    2. **Character Set Size**: The diversity of characters (e.g., lowercase, uppercase, numbers, special symbols) in the password increases the character set size, further boosting entropy.

    The formula to calculate entropy is: 

    `Entropy = Password Length × log₂(Character Set Size)`

    For example, a password that includes lowercase letters (26 characters), uppercase letters (26 characters), and numbers (10 characters), and has a length of 8 characters, will have a character set size of 62. Using the formula, its entropy is:

    `8 × log₂(62) ≈ 47.63 bits`

    The higher the entropy, the stronger the password. To ensure robust security, it is recommended that a password achieve at least 80 bits of entropy, typically requiring a length of at least 12 characters that incorporate a mix of all character types.

    ### How does the entropy clustering work?
    We calculate the entropy for each password in the dataset and cluster together passwords with similar entropy values. When you input your password, we compute its entropy and display its closest cluster in the scatter plot. This helps you visualize where your password stands in terms of security compared to other passwords in the dataset. 

    You will also see the top 10 closest passwords to yours based on entropy, along with their entropy values, providing a clear comparison of how secure your password is.
    """)

    plot_entropy(password, entropy_clusters)

    """ N-gram Clustering """
    st.subheader('Clustering by N-gram Log-Likelihood')
    st.write("""
    **N-gram Log-Likelihood** is a statistical measure used to evaluate the probability of a sequence of characters (in this case, a password) based on patterns observed in large datasets. N-grams refer to contiguous sequences of 'n' characters. For example, a bi-gram analyzes two consecutive characters at a time. 

    The **log-likelihood** value tells us how likely a given sequence is to occur based on common linguistic patterns. Passwords with high log-likelihood values are considered **"meaningful"**, meaning they follow more predictable patterns and resemble natural language, making them easier to guess. On the other hand, passwords with low log-likelihood values are considered **"gibberish"** because they appear more random and less predictable.

    ### Why does this matter?
    While a password with high entropy might seem secure because it uses diverse characters, the presence of common words or patterns can increase the n-gram log-likelihood. This makes it more vulnerable to attacks that rely on language models or dictionaries. For example, "WelcomeBack2022!" might have high entropy, but it contains common words and a predictable number sequence, resulting in a higher log-likelihood and making it more predictable.

    ### Meaningful vs. Gibberish
    When clustering passwords, we differentiate between **meaningful** and **gibberish** sequences based on their average n-gram log-likelihood. 
    - **Meaningful** passwords tend to follow linguistic or common patterns, making them more vulnerable to attacks based on language models.
    - **Gibberish** passwords appear more random and are less predictable, which enhances their security.

    Your password’s n-gram log-likelihood score will determine which cluster it belongs to—**meaningful** or **gibberish**—providing you with insights into how predictable or random your password structure is.
    """)
    # Load n-gram probabilities
    ngram_probs = load_ngram_probabilities(ngram_prob_path)
    plot_ngram(password, ngram_clusters_json, ngram_probs)

    """ MinHash Clustering """
    st.subheader('Clustering by MinHash Similarity')
    st.write("""
    **MinHash** is a technique used to estimate the similarity between sets, which in this context refers to passwords as sets of characters. By treating each password as a set of unique characters and hashing these characters into a compact signature (MinHash signature), we can efficiently compare passwords based on their structural similarities.

    The **Jaccard similarity** between two sets (passwords) is calculated by measuring how many characters they share. MinHash approximates this similarity efficiently, which is especially useful when comparing a large number of passwords. This allows us to identify structurally similar passwords, even if their character sequences or lengths differ.

    ### Why does MinHash matter for passwords?
    MinHash helps uncover passwords that may not be identical but are structurally similar. For example, passwords like "Password123" and "Password456" might have different digits at the end, but their character sets overlap significantly. MinHash captures this similarity and clusters these passwords together.

    ### Visualization and Clustering
    You can adjust the number of MinHash clusters displayed using a slider, ranging from 10 to 100. The scatter plot visualizes these clusters, with each point representing one of the **largest** password clusters. The distances between clusters reflect how different or similar they are based on character composition, and the size of the points corresponds to the number of passwords in each cluster.

    By analyzing the largest 100 clusters, you can get a sense of how passwords are grouped based on their structural similarities. The visualization also shows the **top N** closest passwords to yours, where **N** is the number of clusters you've chosen to display. This allows you to understand how your password compares to others in terms of character overlap and structure.
    """)

    
    st.subheader("MinHash Clusters Visualization")
    top_k = st.slider('Select number of top clusters to display', min_value=5, max_value=100, value=10, step=5)

    # Load clusters and similarities
    if os.path.exists(minhash_similarity_json_path):
        clusters, similarities = load_clusters_and_similarities(minhash_similarity_json_path)

        # Find the closest MinHash cluster to the user's password
        closest_cluster_label, max_similarity = find_closest_minhash_cluster_label(password, clusters)

        if max_similarity == 0:
            st.write("No similar cluster found. Creating a new cluster for your password.")

            user_cluster_label = 'User Password'
            clusters[user_cluster_label] = [password]

            cluster_names, cluster_sizes, cluster_examples, similarity_matrix = get_top_k_clusters(
                clusters, similarities, top_k=top_k)

            cluster_names.append(user_cluster_label)
            cluster_sizes = np.append(cluster_sizes, 1)
            cluster_examples.append([password])

            user_similarities = []
            input_minhash = get_minhash(password)
            for cluster_label in cluster_names[:-1]:
                cluster_minhash = get_minhash(clusters[cluster_label][0])
                sim = input_minhash.jaccard(cluster_minhash)
                user_similarities.append(1 - sim)

            new_row = np.array(user_similarities + [0])
            similarity_matrix = np.vstack([similarity_matrix, new_row[:-1]])
            new_col = np.append(new_row, 0)[:, np.newaxis]
            similarity_matrix = np.hstack([similarity_matrix, new_col])

            closest_cluster_label = user_cluster_label
        else:
            cluster_names, cluster_sizes, cluster_examples, similarity_matrix = get_top_k_clusters(
                clusters, similarities, top_k=top_k, include_clusters=[closest_cluster_label])

        visualize_clusters(cluster_names, cluster_sizes, cluster_examples, similarity_matrix, closest_cluster_label)
    else:
        st.warning("MinHash clusters with similarities file not found.")

    plot_minhash(password, minhash_clusters)

    

