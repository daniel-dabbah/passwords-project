from collections import defaultdict

from tqdm import tqdm
import password_statistics as ps
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
import string
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

def plot_password_length_histogram(length_percentages,dataset_name):
    """ Plot the password length histogram """
    lengths = list(length_percentages.keys())
    percentages = list(length_percentages.values())
    plt.figure(figsize=(10, 5))
    plt.bar(lengths, percentages, color='blue', alpha=0.7)
    plt.xlabel('Password Length')
    plt.ylabel('Percentage')
    plt.title('Password Length Distribution')
    plt.xticks(lengths)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    
    save_plot(dataset_name, "plot_password_length_histogram")  # Close the figure to free up memory

def plot_ascii_usage_histogram(ascii_counts,dataset_name):
    """ Plot the ASCII usage histogram """
    chars = list(ascii_counts.keys())
    percentages = list(ascii_counts.values()) 
    plt.figure(figsize=(10, 5))
    plt.bar(chars, percentages, color='green', alpha=0.7)
    plt.xlabel('ASCII Character')
    plt.ylabel('Percentage')
    plt.title('ASCII Character Usage in Passwords')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    save_plot(dataset_name, "plot_ascii_usage_histogram") 
    
def plot_year_histogram(year_counts,dataset_name):
    """ Plot the year histogram """
    years = list(year_counts.keys())
    counts = list(year_counts.values())
    plt.figure(figsize=(10, 5))
    plt.bar(years, counts, color='red', alpha=0.7)
    plt.xlabel('Year')
    plt.ylabel('Counts')
    plt.title('Year Usage in Passwords')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    save_plot(dataset_name, "plot_year_histogram") 


def number_position_violin_plot(number_positions, dataset_name):
    """ Plot the position of numbers in passwords """
    number_plot_df = pd.DataFrame(number_positions, columns=['Total length of password', 'Normalized Position'])
    single_color = "skyblue"    
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='Total length of password', y='Normalized Position', data=number_plot_df, bw_method=0.2, inner=None, density_norm='width', color=single_color)
    plt.title('Position of Numbers in Passwords by Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    save_plot(dataset_name, "plot_number_position_violin_plot")

def special_character_position_violin_plot(special_char_positions, dataset_name):
    """ Plot the position of special characters in passwords """
    special_char_plot_df = pd.DataFrame(special_char_positions, columns=['Total length of password', 'Normalized Position'])
    single_color = "skyblue"
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='Total length of password', y='Normalized Position', data=special_char_plot_df, bw_method=0.2, inner=None, density_norm='width', color=single_color)
    plt.title('Position of Special Characters in Passwords by Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    save_plot(dataset_name, "plot_special_character_position_violin_plot")


def special_character_position_violin_plot_for_specific_characters(special_char_positions, char,dataset_name):
    """ Plot the position of a specific char in passwords """
    special_char_plot_df = pd.DataFrame(special_char_positions, columns=['Total length of password', 'Normalized Position'])
    single_color = "skyblue"
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='Total length of password', y='Normalized Position', data=special_char_plot_df, bw_method=0.1, inner=None, density_norm='width', color=single_color)
    plt.title(f'Position of "{char}" Special Character in Passwords by Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    save_plot(dataset_name, "plot_special_character_position_violin_plot_for_specific_characters")


def plot_count_of_special_characters_by_length(special_char_counts, length,dataset_name):
    """ Plot the count of special characters in passwords of different lengths """
    special_chars = list(special_char_counts.keys())
    counts = list(special_char_counts.values())
    plt.figure(figsize=(10, 5))
    plt.bar(special_chars, counts, color='orange', alpha=0.7)
    plt.xlabel('Special Character')
    plt.ylabel('Counts')
    plt.title('Special Character Usage in Passwords of length {}'.format(length))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    save_plot(dataset_name, "plot_count_of_special_characters_by_length")


def plot_count_of_numbers_by_length(numbers_counts, length,dataset_name):
    """ Plot the count of numbers in passwords of different lengths """
    numbers = list(numbers_counts.keys())
    counts = list(numbers_counts.values())
    plt.figure(figsize=(10, 5))
    plt.bar(numbers, counts, color='orange', alpha=0.7)
    plt.xlabel('Number')
    plt.ylabel('Counts')
    plt.title('Number Usage in Passwords of length {}'.format(length))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    save_plot(dataset_name, "plot_count_of_numbers_by_length")


def plot_count_of_upper_case_by_length(upper_case_counts, length,dataset_name):
    """ Plot the count of upper case characters in passwords of different lengths """
    upper_case = list(upper_case_counts.keys())
    counts = list(upper_case_counts.values())
    plt.figure(figsize=(10, 5))
    plt.bar(upper_case, counts, color='orange', alpha=0.7)
    plt.xlabel('Upper Case')
    plt.ylabel('Counts')
    plt.title('Upper Case Usage in Passwords of length {}'.format(length))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    save_plot(dataset_name, "plot_count_of_upper_case_by_length")


def plot_count_of_lower_case_by_length(lower_case_counts, length,dataset_name):
    """ Plot the count of lower case characters in passwords of different lengths """
    lower_case = list(lower_case_counts.keys())
    counts = list(lower_case_counts.values())
    plt.figure(figsize=(10, 5))
    plt.bar(lower_case, counts, color='orange', alpha=0.7)
    plt.xlabel('Lower Case')
    plt.ylabel('Counts')
    plt.title('Lower Case Usage in Passwords of length {}'.format(length))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    # plt.show()
    save_plot(dataset_name, "plot_count_of_lower_case_by_length")


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
    plt.figure(figsize=(12, 8))
    plt.bar(labels, percentages, color='skyblue')
    plt.ylabel('Percentage', fontsize=14)
    plt.xlabel('Characteristics', fontsize=14)
    plt.title('Password Characteristics Distribution', fontsize=16)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    # plt.show()
    save_plot(dataset_name, "plot_categories_bar_plot")

def plot_percentage_of_special_characters_by_length(special_char_counts, length, dataset_name):
    """ Plot the percentage of special characters in passwords of different lengths """
    special_chars = list(special_char_counts.keys())
    total_counts = sum(special_char_counts.values())
    percentages = [(count / total_counts) * 100 if total_counts > 0 else 0 for count in special_char_counts.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(special_chars, percentages, color='orange', alpha=0.7)
    plt.xlabel('Special Character')
    plt.ylabel('Percentage (%)')
    plt.title(f'Special Character Usage as Percentage in Passwords of length {length}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"percentage_of_special_characters_by_length_{length}")
    plt.close()

def plot_percentage_of_numbers_by_length(numbers_counts, length, dataset_name):
    """ Plot the percentage of numbers in passwords of different lengths """
    numbers = list(numbers_counts.keys())
    total_counts = sum(numbers_counts.values())
    percentages = [(count / total_counts) * 100 if total_counts > 0 else 0 for count in numbers_counts.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(numbers, percentages, color='orange', alpha=0.7)
    plt.xlabel('Number')
    plt.ylabel('Percentage (%)')
    plt.title(f'Number Usage as Percentage in Passwords of length {length}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"percentage_of_numbers_by_length_{length}")
    plt.close()

def plot_percentage_of_upper_case_by_length(upper_case_counts, length, dataset_name):
    """ Plot the percentage of upper case characters in passwords of different lengths """
    upper_case = list(upper_case_counts.keys())
    total_counts = sum(upper_case_counts.values())
    percentages = [(count / total_counts) * 100 if total_counts > 0 else 0 for count in upper_case_counts.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(upper_case, percentages, color='orange', alpha=0.7)
    plt.xlabel('Upper Case')
    plt.ylabel('Percentage (%)')
    plt.title(f'Upper Case Usage as Percentage in Passwords of length {length}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"percentage_of_upper_case_by_length_{length}")
    plt.close()

def plot_percentage_of_lower_case_by_length(lower_case_counts, length, dataset_name):
    """ Plot the percentage of lower case characters in passwords of different lengths """
    lower_case = list(lower_case_counts.keys())
    total_counts = sum(lower_case_counts.values())
    percentages = [(count / total_counts) * 100 if total_counts > 0 else 0 for count in lower_case_counts.values()]
    plt.figure(figsize=(10, 5))
    plt.bar(lower_case, percentages, color='orange', alpha=0.7)
    plt.xlabel('Lower Case')
    plt.ylabel('Percentage (%)')
    plt.title(f'Lower Case Usage as Percentage in Passwords of length {length}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"percentage_of_lower_case_by_length_{length}")
    plt.close()


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
    plt.figure(figsize=(10, 5))
    patches, texts, autotexts = plt.pie(
        percentages, labels=labels, autopct=autopct_function, startangle=140, colors=plt.cm.Paired.colors,
        textprops={'fontsize': 12},  # Change font size for labels
        wedgeprops={'linewidth': 0, 'edgecolor': 'black'}  # Optional: Add edge color to wedges
    )
    plt.title('Password Characteristics Distribution', fontsize=16, pad=30)  # Increase pad to move title further
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    # Change the font size of percentages based on their value
    for autotext in autotexts:
        pct = float(autotext.get_text().replace('%', ''))
        if pct < 1:
            autotext.set_fontsize(6)
        else:
            autotext.set_fontsize(8)
    # plt.show()
    save_plot(dataset_name, "plot_categories_pie_plot_via_matplotlib")

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
    # fig.show()

def plot_entropy_histogram(entropies, dataset_name):
    """ Plot the histogram of all entropies """
    plt.figure(figsize=(10, 5))
    plt.hist(entropies, bins=20, color='blue', alpha=0.7)
    plt.xlabel('Entropy (bits)')
    plt.ylabel('Frequency')
    plt.title('Entropy Distribution of All Passwords')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "entropy_histogram")
    plt.close()

def plot_entropy__percentages_histogram(entropies, dataset_name):
    """ Plot the histogram of all entropies showing percentages instead of frequencies. """
    plt.figure(figsize=(10, 5))
    # Plot histogram with density=True to get the probability density, then multiply by 100 to get percentages
    n, bins, patches = plt.hist(entropies, bins=50, color='blue', alpha=0.7, density=True, weights=np.ones(len(entropies)) / len(entropies))
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter(1))  # This converts y-axis into percentage
    plt.xlabel('Entropy (bits)')
    plt.ylabel('Percentage of Passwords (%)')
    plt.title('Entropy Distribution of All Passwords (Percentage)')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, "entropy_histogram_percentage")
    plt.close()

def plot_entropy_by_length_histogram(entropy_by_length, length, dataset_name):
    """ Plot the histogram of entropies for a specific password length """
    plt.figure(figsize=(10, 5))
    plt.hist(entropy_by_length, bins=20, color='green', alpha=0.7)
    plt.xlabel('Entropy (bits)')
    plt.ylabel('Frequency')
    plt.title(f'Entropy Distribution for Passwords of Length {length}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"entropy_histogram_length_{length}")
    plt.close()
    

def plot_entropy_by_length_percentages_histogram(entropies, length, dataset_name):
    """Plot the histogram of entropies for a specific password length with percentages."""
    plt.figure(figsize=(10, 5))
    # Plot histogram without density=True to get raw counts
    n, bins, patches = plt.hist(entropies, bins=20, color='green', alpha=0.7)
    total = sum(n)  # Sum of counts in all bins, which is total passwords of that length
    n_percentage = [(x / total) * 100 for x in n]  # Convert counts to percentages
    plt.cla()  # Clear the plot
    # Replot with percentage data
    plt.bar(bins[:-1], n_percentage, width=np.diff(bins), color='green', alpha=0.7, align='edge')
    plt.gca().yaxis.set_major_formatter(plt.matplotlib.ticker.PercentFormatter())  # Now percentages will show correctly
    plt.xlabel('Entropy (bits)')
    plt.ylabel('Percentage of Passwords (%)')
    plt.title(f'Entropy Distribution (Percentage) for Passwords of Length {length}')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    save_plot(dataset_name, f"percentage_entropy_histogram_length_{length}")
    plt.close()


if __name__ == '__main__':

    # Create the directory if it doesn't exist
    os.makedirs('generated_plots', exist_ok=True)

    #TODO: change to rockyuoi 2024-100K.txt
    dataset_name = 'rockyou_mini.txt'
    passwords, statistics = ps.analyze_passwords(dataset_name)

    """ Print some of the statistics of the passwords """
    # print_statistics(statistics)
    
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

    
    