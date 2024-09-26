from collections import defaultdict

from tqdm import tqdm
import password_statistics as ps
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import string
import pickle
import password_statistics as ps


def save_plot(dataset_name, plot_name):
    """ Save the plot to the generated_plots directory """
    os.makedirs("generated_plots", exist_ok=True)
    os.makedirs(os.path.join("generated_plots", dataset_name), exist_ok=True)
    plt.savefig(os.path.join("generated_plots", dataset_name, f"{plot_name}.png"))
    plt.close()

def print_statistics(statistics):
    """ Print dome of the statistics of the passwords """
    print(f"Total Passwords: {statistics['total']}")
    print(f"Lower Case Only: {statistics['lower_case_only_percentage']:.2f}%")
    print(f"Upper Case Only: {statistics['upper_case_only_percentage']:.2f}%")
    print(f"Special Characters Only: {statistics['special_characters_only_percentage']:.2f}%")
    print(f"Numbers Only: {statistics['numbers_only_percentage']:.2f}%")
    print(f"Contains Lower Case: {statistics['contains_lower_case_percentage']:.2f}%")
    print(f"Contains Upper Case: {statistics['contains_upper_case_percentage']:.2f}%")
    print(f"Contains Special Characters: {statistics['contains_special_characters_percentage']:.2f}%")
    print(f"Contains Numbers: {statistics['contains_numbers_percentage']:.2f}%")
    print(f"Lower Case and Numbers: {statistics['lower_case_and_numbers_percentage']:.2f}%")
    print(f"Lower Case and Special Characters: {statistics['lower_case_and_special_characters_percentage']:.2f}%")
    print(f"Lower Case and Upper Case: {statistics['lower_case_and_upper_case_percentage']:.2f}%")
    print(f"Upper Case and Numbers: {statistics['upper_case_and_numbers_percentage']:.2f}%")
    print(f"Upper Case and Special Characters: {statistics['upper_case_and_special_characters_percentage']:.2f}%")
    print(f"Special Characters and Numbers: {statistics['special_characters_and_numbers_percentage']:.2f}%")
    print(f"Lower Upper Case and Numbers: {statistics['lower_upper_case_and_numbers_percentage']:.2f}%")
    print(f"Lower Upper Case and Special Characters: {statistics['lower_upper_case_and_special_characters_percentage']:.2f}%")
    print(f"Lower Special Characters and Numbers: {statistics['lower_special_characters_and_numbers_percentage']:.2f}%")
    print(f"Upper Special Characters and Numbers: {statistics['upper_special_characters_and_numbers_percentage']:.2f}%")
    print(f"All Character Types: {statistics['all_character_types_percentage']:.2f}%")
    print(f"Upper Case Only at Beginning: {statistics['upper_case_only_beginning_percentage']:.2f}%"),
    print(f"Numbers Only at end: {statistics['numbers_only_at_end_percentage']:.2f}%")

def plot_password_length_histogram(length_percentages, dataset_name, highlight_length=None):
    """ Plot the password length histogram """
    lengths = list(length_percentages.keys())
    percentages = list(length_percentages.values())

    # Create the plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(lengths, percentages, color='#4CAF50', alpha=0.7)  # Soft green color
    
    # Highlight the bar corresponding to the entered password length
    if highlight_length is not None:
        highlight_bar = next((bar for bar in bars if bar.get_x() == highlight_length - 0.4), None)
        if highlight_bar:
            highlight_bar.set_color('#FFA726')  # Soft orange color
            highlight_bar.set_alpha(1.0)
    
    ax.set_xlabel('Password Length')
    ax.set_ylabel('Percentage')
    ax.set_title('Password Length Distribution')
    ax.set_xticks(lengths)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text annotation for the entered password length
    if highlight_length is not None:
        ax.text(highlight_length, 50, f'Your password length: {highlight_length}', 
                ha='center', va='bottom', color='#FFA726', fontweight='bold')  # Soft orange color
    
    # Return the figure
    return fig

def plot_ascii_usage_histogram(ascii_counts, dataset_name, input_string=None):
    """ Plot the ASCII usage histogram """
    chars = list(ascii_counts.keys())
    percentages = list(ascii_counts.values()) 
    fig, ax = plt.subplots(figsize=(20, 5))  # Increased width from 10 to 20
    
    # Create a default color list with a more pleasant soft blue
    colors = ['#4682B4' for _ in chars]  # Steel Blue
    
    # If an input string is provided, highlight its characters with a soft orange
    if input_string:
        highlight_color = '#FFA07A'  # Light Salmon
        for i, char in enumerate(chars):
            if char in input_string:
                colors[i] = highlight_color
    
    # Plot the bars with the appropriate colors
    bars = ax.bar(chars, percentages, color=colors, alpha=0.8)
    
    ax.set_xlabel('ASCII Character', fontsize=12)
    ax.set_ylabel('Percentage', fontsize=12)
    ax.set_title('ASCII Character Usage in Passwords', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3, color='#CCCCCC')  # Lighter, less intrusive grid
    
    # If an input string was provided, add a legend with more pleasing colors
    if input_string:
        # Create a custom legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#FFA07A', edgecolor='none', label='Characters in your password'),
            Patch(facecolor='#4682B4', edgecolor='none', label='Characters not in your password')
        ]
        ax.legend(handles=legend_elements, loc='upper right', 
                  facecolor='#F0F0F0', edgecolor='none', fontsize=10)
        
        # Add text annotation for the characters in the password
        chars_in_password = set(input_string)
        ax.text(0.5, 0.95, f"Characters in your password: {''.join(sorted(chars_in_password))}", 
                transform=ax.transAxes, ha='center', va='top', 
                bbox=dict(facecolor='#FFA07A', edgecolor='none', alpha=0.7),
                fontsize=12, fontweight='bold')
    
    # Set a light background color for better contrast
    ax.set_facecolor('#F5F5F5')  # Very light grey
    
    # Adjust layout to prevent cutting off labels
    plt.tight_layout()
    
    save_plot(dataset_name, "plot_ascii_usage_histogram")
    plt.close(fig)
    return fig
    
def plot_year_histogram(year_counts,dataset_name):
    """ Plot the year histogram """
    years = list(year_counts.keys())
    counts = list(year_counts.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(years, counts, color='red', alpha=0.7)
    ax.set_xlabel('Year')
    ax.set_ylabel('Counts')
    ax.set_title('Year Usage in Passwords')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "plot_year_histogram")
    plt.close(fig)
    return fig


def number_position_violin_plot(number_positions, dataset_name):
    """ Plot the position of numbers in passwords """
    number_plot_df = pd.DataFrame(number_positions, columns=['Total length of password', 'Normalized Position'])
    single_color = "skyblue"    
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(x='Total length of password', y='Normalized Position', data=number_plot_df, bw_method=0.2, inner=None, density_norm='width', color=single_color, ax=ax)
    ax.set_title('Position of Numbers in Passwords by Length')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "plot_number_position_violin_plot")
    plt.close(fig)
    return fig

def special_character_position_violin_plot(special_char_positions, dataset_name):
    """ Plot the position of special characters in passwords """
    special_char_plot_df = pd.DataFrame(special_char_positions, columns=['Total length of password', 'Normalized Position'])
    single_color = "skyblue"
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(x='Total length of password', y='Normalized Position', data=special_char_plot_df, bw_method=0.2, inner=None, density_norm='width', color=single_color, ax=ax)
    ax.set_title('Position of Special Characters in Passwords by Length')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "plot_special_character_position_violin_plot")
    plt.close(fig)
    return fig


def special_character_position_violin_plot_for_specific_characters(special_char_positions, char,dataset_name):
    """ Plot the position of a specific char in passwords """
    special_char_plot_df = pd.DataFrame(special_char_positions, columns=['Total length of password', 'Normalized Position'])
    single_color = "skyblue"
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(x='Total length of password', y='Normalized Position', data=special_char_plot_df, bw_method=0.1, inner=None, density_norm='width', color=single_color, ax=ax)
    ax.set_title(f'Position of "{char}" Special Character in Passwords by Length')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "plot_special_character_position_violin_plot_for_specific_characters")
    plt.close(fig)
    return fig


def plot_count_of_special_characters_by_length(special_char_counts, length,dataset_name):
    """ Plot the count of special characters in passwords of different lengths """
    special_chars = list(special_char_counts.keys())
    counts = list(special_char_counts.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(special_chars, counts, color='orange', alpha=0.7)
    ax.set_xlabel('Special Character')
    ax.set_ylabel('Counts')
    ax.set_title(f'Special Character Usage in Passwords of length {length}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "plot_count_of_special_characters_by_length")
    plt.close(fig)
    return fig


def plot_count_of_numbers_by_length(numbers_counts, length,dataset_name):
    """ Plot the count of numbers in passwords of different lengths """
    numbers = list(numbers_counts.keys())
    counts = list(numbers_counts.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(numbers, counts, color='orange', alpha=0.7)
    ax.set_xlabel('Number')
    ax.set_ylabel('Counts')
    ax.set_title('Number Usage in Passwords of length {}'.format(length))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "plot_count_of_numbers_by_length")
    plt.close(fig)
    return fig


def plot_count_of_upper_case_by_length(upper_case_counts, length,dataset_name):
    """ Plot the count of upper case characters in passwords of different lengths """
    upper_case = list(upper_case_counts.keys())
    counts = list(upper_case_counts.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(upper_case, counts, color='orange', alpha=0.7)
    ax.set_xlabel('Upper Case')
    ax.set_ylabel('Counts')
    ax.set_title('Upper Case Usage in Passwords of length {}'.format(length))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "plot_count_of_upper_case_by_length")
    plt.close(fig)
    return fig


def plot_count_of_lower_case_by_length(lower_case_counts, length,dataset_name):
    """ Plot the count of lower case characters in passwords of different lengths """
    lower_case = list(lower_case_counts.keys())
    counts = list(lower_case_counts.values())
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lower_case, counts, color='orange', alpha=0.7)
    ax.set_xlabel('Lower Case')
    ax.set_ylabel('Counts')
    ax.set_title('Lower Case Usage in Passwords of length {}'.format(length))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "plot_count_of_lower_case_by_length")
    plt.close(fig)
    return fig


def plot_categories_bar_plot(statistics,dataset_name):
    labels = ["Contains Lower Case", "Contains Upper Case", "Contains Numbers", "Contains Special",
    "Lower Case Only", "Upper Case Only", "Numbers Only", "Special Only",
    "Lower and Numbers", "Lower and Special", "Lower and Upper",
    "Upper and Numbers", "Upper and Special", "Special and Numbers",
    "Lower,Upper,Numbers", "Lower,Upper,Special", "Lower,Special,Numbers",
    "Upper,Special,Numbers", "All Character Types"
    ]
    percentages = [statistics['contains_lower_case_percentage'], statistics['contains_upper_case_percentage'],
        statistics['contains_numbers_percentage'], statistics['contains_special_characters_percentage'],
        statistics['lower_case_only_percentage'], statistics['upper_case_only_percentage'], statistics['numbers_only_percentage'],
        statistics['special_characters_only_percentage'], statistics['lower_case_and_numbers_percentage'],
        statistics['lower_case_and_special_characters_percentage'], statistics['lower_case_and_upper_case_percentage'], statistics['upper_case_and_numbers_percentage'],
        statistics['upper_case_and_special_characters_percentage'], statistics['special_characters_and_numbers_percentage'], statistics['lower_upper_case_and_numbers_percentage'],
        statistics['lower_upper_case_and_special_characters_percentage'], statistics['lower_special_characters_and_numbers_percentage'], statistics['upper_special_characters_and_numbers_percentage'],
        statistics['all_character_types_percentage']
    ]
    # Plot the bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(labels, percentages, color='skyblue')
    ax.set_ylabel('Percentage', fontsize=14)
    ax.set_xlabel('Characteristics', fontsize=14)
    ax.set_title('Password Characteristics Distribution', fontsize=16)
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax.tick_params(axis='y', labelsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    save_plot(dataset_name, "plot_categories_bar_plot")
    plt.close(fig)
    return fig

def plot_percentage_of_special_characters_by_length(special_char_counts, length, dataset_name):
    """ Plot the percentage of special characters in passwords of different lengths """
    special_chars = list(special_char_counts.keys())
    total_counts = sum(special_char_counts.values())
    percentages = [(count / total_counts) * 100 if total_counts > 0 else 0 for count in special_char_counts.values()]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(special_chars, percentages, color='orange', alpha=0.7)
    ax.set_xlabel('Special Character')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Special Character Usage as Percentage in Passwords of length {length}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"percentage_of_special_characters_by_length_{length}")
    plt.close(fig)
    return fig

def plot_percentage_of_numbers_by_length(numbers_counts, length, dataset_name):
    """ Plot the percentage of numbers in passwords of different lengths """
    numbers = list(numbers_counts.keys())
    total_counts = sum(numbers_counts.values())
    percentages = [(count / total_counts) * 100 if total_counts > 0 else 0 for count in numbers_counts.values()]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(numbers, percentages, color='orange', alpha=0.7)
    ax.set_xlabel('Number')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Number Usage as Percentage in Passwords of length {length}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"percentage_of_numbers_by_length_{length}")
    plt.close(fig)
    return fig

def plot_percentage_of_upper_case_by_length(upper_case_counts, length, dataset_name):
    """ Plot the percentage of upper case characters in passwords of different lengths """
    upper_case = list(upper_case_counts.keys())
    total_counts = sum(upper_case_counts.values())
    percentages = [(count / total_counts) * 100 if total_counts > 0 else 0 for count in upper_case_counts.values()]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(upper_case, percentages, color='orange', alpha=0.7)
    ax.set_xlabel('Upper Case')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Upper Case Usage as Percentage in Passwords of length {length}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"percentage_of_upper_case_by_length_{length}")
    plt.close(fig)
    return fig

def plot_percentage_of_lower_case_by_length(lower_case_counts, length, dataset_name):
    """ Plot the percentage of lower case characters in passwords of different lengths """
    lower_case = list(lower_case_counts.keys())
    total_counts = sum(lower_case_counts.values())
    percentages = [(count / total_counts) * 100 if total_counts > 0 else 0 for count in lower_case_counts.values()]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(lower_case, percentages, color='orange', alpha=0.7)
    ax.set_xlabel('Lower Case')
    ax.set_ylabel('Percentage (%)')
    ax.set_title(f'Lower Case Usage as Percentage in Passwords of length {length}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"percentage_of_lower_case_by_length_{length}")
    plt.close(fig)
    return fig


def plot_categories_pie_plot_via_matplotlib(statistics,dataset_name):
    """ Plot the categories pie plot using matplotlib """
    labels = [
        "Lower Case Only", "Upper Case Only", "Numbers Only", 
        "Lower and Numbers", "Lower and Special",
        "Upper and Numbers", "Upper and Special", "Lower Upper and Numbers", 
        "Special and Numbers", "Lower Upper and Special", "Lower Special and Numbers",
        "Upper Special and Numbers", "All Character Types", "Lower and Upper",
        "special_characters_and_numbers_percentage", "lower_upper_case_and_special_characters_percentage",
        "lower_special_characters_and_numbers_percentage", "upper_special_characters_and_numbers_percentage",
        "all_character_types_percentage", "lower_case_and_upper_case_percentage"
        ]
    percentages = [
        statistics['lower_case_only_percentage'], 
        statistics['upper_case_only_percentage'], 
        statistics['numbers_only_percentage'],
        statistics['lower_case_and_numbers_percentage'],
        statistics['lower_case_and_special_characters_percentage'], 
        statistics['upper_case_and_numbers_percentage'],
        statistics['upper_case_and_special_characters_percentage'], 
        statistics['lower_upper_case_and_numbers_percentage'],
        statistics['special_characters_and_numbers_percentage'],
        statistics['lower_upper_case_and_special_characters_percentage'], 
        statistics['lower_special_characters_and_numbers_percentage'], 
        statistics['upper_special_characters_and_numbers_percentage'],
        statistics['all_character_types_percentage'],
        statistics['lower_case_and_upper_case_percentage'],
        statistics['special_characters_and_numbers_percentage'],
        statistics['lower_upper_case_and_special_characters_percentage'],
        statistics['lower_special_characters_and_numbers_percentage'],
        statistics['upper_special_characters_and_numbers_percentage'],
        statistics['all_character_types_percentage'],
        statistics['lower_case_and_upper_case_percentage']
    ]
    # Function to selectively show percentages with custom font sizes
    def autopct_function(pct):
        return ('%1.1f%%' % pct) if pct > 1 else ('%1.1f%%' % pct)
    fig, ax = plt.subplots(figsize=(10, 5))
    patches, texts, autotexts = ax.pie(
        percentages, labels=labels, autopct=autopct_function, startangle=140, colors=plt.cm.Paired.colors,
        textprops={'fontsize': 12},  # Change font size for labels
        wedgeprops={'linewidth': 0, 'edgecolor': 'black'}  # Optional: Add edge color to wedges
    )
    ax.set_title('Password Characteristics Distribution', fontsize=16, pad=30)  # Increase pad to move title further
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Change the font size of percentages based on their value
    for autotext in autotexts:
        pct = float(autotext.get_text().replace('%', ''))
        if pct < 1:
            autotext.set_fontsize(6)
        else:
            autotext.set_fontsize(8)
    
    save_plot(dataset_name, "plot_categories_pie_plot_via_matplotlib")
    plt.close(fig)
    return fig

def plot_categories_pie_plot_via_plotly(statistics,dataset_name):
    """ Plot the categories pie plot using plotly """
    labels = [
        "Lower Case Only", "Upper Case Only", "Numbers Only", 
        "Lower and Numbers", "Lower and Special",
        "Upper and Numbers", "Upper and Special", "Lower Upper and Numbers", 
        "Special and Numbers", "Lower Upper and Special", "Lower Special and Numbers",
        "Upper Special and Numbers", "All Character Types", "Lower and Upper",
        "special_characters_and_numbers_percentage", "lower_upper_case_and_special_characters_percentage",
        "lower_special_characters_and_numbers_percentage", "upper_special_characters_and_numbers_percentage",
        "all_character_types_percentage", "lower_case_and_upper_case_percentage"
        ]
    percentages = [
        statistics['lower_case_only_percentage'], 
        statistics['upper_case_only_percentage'], 
        statistics['numbers_only_percentage'],
        statistics['lower_case_and_numbers_percentage'],
        statistics['lower_case_and_special_characters_percentage'], 
        statistics['upper_case_and_numbers_percentage'],
        statistics['upper_case_and_special_characters_percentage'], 
        statistics['lower_upper_case_and_numbers_percentage'],
        statistics['special_characters_and_numbers_percentage'],
        statistics['lower_upper_case_and_special_characters_percentage'], 
        statistics['lower_special_characters_and_numbers_percentage'], 
        statistics['upper_special_characters_and_numbers_percentage'],
        statistics['all_character_types_percentage'],
        statistics['lower_case_and_upper_case_percentage'],
        statistics['special_characters_and_numbers_percentage'],
        statistics['lower_upper_case_and_special_characters_percentage'],
        statistics['lower_special_characters_and_numbers_percentage'],
        statistics['upper_special_characters_and_numbers_percentage'],
        statistics['all_character_types_percentage'],
        statistics['lower_case_and_upper_case_percentage']
    ]
    import plotly.express as px
    import plotly.graph_objects as go
    # Create pie chart with Plotly Graph Objects
    fig = go.Figure()
    # Add wedges to the pie chart
    fig.add_trace(go.Pie(
        labels=labels,
        values=percentages,
        textinfo='percent',
        hoverinfo='label+percent',
        textposition='outside',
        textfont=dict(size=50),  # Larger font for percentages
        marker=dict(line=dict(color='#000000', width=1))  # Add edge color to wedges
    ))
    # Update layout to adjust title, ensure equal aspect ratio, and position the legend on the side
    fig.update_layout(
        title_text='Password Characteristics Distribution',
        title_font_size=40,
        title_x=0.5,
        title_y=1,
        title_xanchor='center',
        showlegend=True,
        legend=dict(
            orientation='v',
            x=1,
            y=1,
            title=dict(text='Characteristics', font=dict(size=100)),
            font=dict(size=60),
            itemsizing='trace'
        ),
        margin=dict(t=100, b=100, l=50, r=20),
    )
    return fig
    # fig.show()

def plot_entropy_histogram(entropies, dataset_name):
    """ Plot the histogram of all entropies """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(entropies, bins=20, color='blue', alpha=0.7)
    ax.set_xlabel('Entropy (bits)')
    ax.set_ylabel('Frequency')
    ax.set_title('Entropy Distribution of All Passwords')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "entropy_histogram")
    plt.close(fig)
    return fig

def plot_entropy__percentages_histogram(entropies, dataset_name):
    """ Plot the histogram of all entropies showing percentages instead of frequencies. """
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot histogram with density=True to get the probability density, then multiply by 100 to get percentages
    n, bins, patches = ax.hist(entropies, bins=50, color='blue', alpha=0.7, density=True, weights=np.ones(len(entropies)) / len(entropies))
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))  # This converts y-axis into percentage
    ax.set_xlabel('Entropy (bits)')
    ax.set_ylabel('Percentage of Passwords (%)')
    ax.set_title('Entropy Distribution of All Passwords (Percentage)')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "entropy_histogram_percentage")
    plt.close(fig)
    return fig

def plot_entropy_by_length_histogram(entropy_by_length, length, dataset_name):
    """ Plot the histogram of entropies for a specific password length """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(entropy_by_length, bins=20, color='green', alpha=0.7)
    ax.set_xlabel('Entropy (bits)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Entropy Distribution for Passwords of Length {length}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"entropy_histogram_length_{length}")
    plt.close(fig)
    return fig
    

def plot_entropy_by_length_percentages_histogram(entropies, length, dataset_name):
    """Plot the histogram of entropies for a specific password length with percentages."""
    fig, ax = plt.subplots(figsize=(10, 5))
    # Plot histogram without density=True to get raw counts
    n, bins, patches = ax.hist(entropies, bins=20, color='green', alpha=0.7)
    total = sum(n)  # Sum of counts in all bins, which is total passwords of that length
    n_percentage = [(x / total) * 100 for x in n]  # Convert counts to percentages
    ax.cla()  # Clear the plot
    # Replot with percentage data
    ax.bar(bins[:-1], n_percentage, width=np.diff(bins), color='green', alpha=0.7, align='edge')
    ax.yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter())  # Now percentages will show correctly
    ax.set_xlabel('Entropy (bits)')
    ax.set_ylabel('Percentage of Passwords (%)')
    ax.set_title(f'Entropy Distribution (Percentage) for Passwords of Length {length}')
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"percentage_entropy_histogram_length_{length}")
    plt.close(fig)
    return fig


if __name__ == '__main__':

    # Create the directory if it doesn't exist
    os.makedirs('generated_plots', exist_ok=True)

    #TODO: change to rockyuoi 2024-100K.txt
    dataset_name = 'rockyou_mini.txt'
    dataset_name = 'rockyou2024-100K.txt'
    passwords, statistics = ps.analyze_passwords(dataset_name)

    

    # Save passwords and statistics to a file
    data_to_save = {
        'passwords': passwords,
        'statistics': statistics
    }
    
    with open(f'{dataset_name}_data_passwords_statistics.pkl', 'wb') as f:
        pickle.dump(data_to_save, f)

    print(f"Data saved to {dataset_name}_data_passwords_statistics.pkl")

 

    """ Print some of the statistics of the passwords """
    print_statistics(statistics)
    
    """ Plot the password length histogram """
    plot_password_length_histogram(statistics['length_percentages'], dataset_name)

    """ Plot the ASCII usage histogram """
    plot_ascii_usage_histogram(statistics['ascii_counts'], dataset_name)

    """ Plot Categories """
    plot_categories_bar_plot(statistics, dataset_name)
    plot_categories_pie_plot_via_matplotlib(statistics, dataset_name)
    plot_categories_pie_plot_via_plotly(statistics, dataset_name)

    """ Plot the position of lower, upper, numbers and special chars in all passwords """
    number_position_violin_plot(statistics['number_positions'], dataset_name)
    special_character_position_violin_plot(statistics['special_char_positions'], dataset_name)

    """ Plot the position of special characters in passwords for specific characters """
    special_character_position_violin_plot_for_specific_characters(statistics['special_char_positions_per_char']['.'], '.', dataset_name)
    special_character_position_violin_plot_for_specific_characters(statistics['special_char_positions_per_char']['!'], '!', dataset_name)
    special_character_position_violin_plot_for_specific_characters(statistics['special_char_positions_per_char']['@'], '@', dataset_name)
    special_character_position_violin_plot_for_specific_characters(statistics['special_char_positions_per_char']['#'], '#', dataset_name)
    special_character_position_violin_plot_for_specific_characters(statistics['special_char_positions_per_char']['$'], '$', dataset_name)
    special_character_position_violin_plot_for_specific_characters(statistics['special_char_positions_per_char']['%'], '%', dataset_name)

    """ Plot year histogram """
    plot_year_histogram(statistics['year_counts'], dataset_name)

    """ Plot the count of special characters, numbers, upper case and lower case characters in passwords of different lengths """
    plot_percentage_of_special_characters_by_length(statistics['count_of_special_characters_per_length_per_count_percentages'][8], 8, dataset_name)
    plot_percentage_of_numbers_by_length(statistics['count_of_numbers_per_length_per_count_percentages'][8], 8, dataset_name)
    plot_percentage_of_upper_case_by_length(statistics['count_of_upper_case_per_length_per_count_percentages'][8], 8, dataset_name)
    
    plot_percentage_of_special_characters_by_length(statistics['count_of_special_characters_per_length_per_count_percentages'][9], 9, dataset_name)
    plot_percentage_of_numbers_by_length(statistics['count_of_numbers_per_length_per_count_percentages'][9], 9, dataset_name)
    plot_percentage_of_upper_case_by_length(statistics['count_of_upper_case_per_length_per_count_percentages'][9], 9, dataset_name)

    plot_percentage_of_special_characters_by_length(statistics['count_of_special_characters_per_length_per_count_percentages'][10], 10, dataset_name)
    plot_percentage_of_numbers_by_length(statistics['count_of_numbers_per_length_per_count_percentages'][10], 10, dataset_name)
    plot_percentage_of_upper_case_by_length(statistics['count_of_upper_case_per_length_per_count_percentages'][10], 10, dataset_name)

    plot_percentage_of_special_characters_by_length(statistics['count_of_special_characters_per_length_per_count_percentages'][11], 11, dataset_name)
    plot_percentage_of_numbers_by_length(statistics['count_of_numbers_per_length_per_count_percentages'][11], 11, dataset_name)
    plot_percentage_of_upper_case_by_length(statistics['count_of_upper_case_per_length_per_count_percentages'][11], 11, dataset_name)

    plot_percentage_of_special_characters_by_length(statistics['count_of_special_characters_per_length_per_count_percentages'][12], 12, dataset_name)
    plot_percentage_of_numbers_by_length(statistics['count_of_numbers_per_length_per_count_percentages'][12], 12, dataset_name)
    plot_percentage_of_upper_case_by_length(statistics['count_of_upper_case_per_length_per_count_percentages'][12], 12, dataset_name)

    """ Plot entropy histograms """
    entropy = statistics['entropies']
    entropy_by_length = statistics['entropy_by_length']
    plot_entropy__percentages_histogram(entropy, dataset_name)
    plot_entropy_by_length_percentages_histogram(entropy_by_length[8], 8, dataset_name)

    
    
