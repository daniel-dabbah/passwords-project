import password_statistics as ps
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


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

def plot_password_length_histogram(length_percentages):
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
    plt.show()

def plot_ascii_usage_histogram(ascii_counts):
    """ Plot the ASCII usage histogram """
    chars = list(ascii_counts.keys())
    percentages = list(ascii_counts.values()) 
    plt.figure(figsize=(10, 5))
    plt.bar(chars, percentages, color='green', alpha=0.7)
    plt.xlabel('ASCII Character')
    plt.ylabel('Percentage')
    plt.title('ASCII Character Usage in Passwords')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def plot_year_histogram(year_counts):
    """ Plot the year histogram """
    years = list(year_counts.keys())
    counts = list(year_counts.values())
    plt.figure(figsize=(10, 5))
    plt.bar(years, counts, color='red', alpha=0.7)
    plt.xlabel('Year')
    plt.ylabel('Counts')
    plt.title('Year Usage in Passwords')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def number_position_violin_plot(number_positions):
    """ Plot the position of numbers in passwords """
    number_plot_df = pd.DataFrame(number_positions, columns=['Total length of password', 'Normalized Position'])
    single_color = "skyblue"    
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='Total length of password', y='Normalized Position', data=number_plot_df, bw_method=0.2, inner=None, density_norm='width', color=single_color)
    plt.title('Position of Numbers in Passwords by Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

def special_character_position_violin_plot(special_char_positions):
    """ Plot the position of special characters in passwords """
    special_char_plot_df = pd.DataFrame(special_char_positions, columns=['Total length of password', 'Normalized Position'])
    single_color = "skyblue"
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='Total length of password', y='Normalized Position', data=special_char_plot_df, bw_method=0.2, inner=None, density_norm='width', color=single_color)
    plt.title('Position of Special Characters in Passwords by Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def special_character_position_violin_plot_for_specific_characters(special_char_positions, char):
    """ Plot the position of a specific char in passwords """
    special_char_plot_df = pd.DataFrame(special_char_positions, columns=['Total length of password', 'Normalized Position'])
    single_color = "skyblue"
    plt.figure(figsize=(10, 5))
    sns.violinplot(x='Total length of password', y='Normalized Position', data=special_char_plot_df, bw_method=0.1, inner=None, density_norm='width', color=single_color)
    plt.title(f'Position of "{char}" Special Character in Passwords by Length')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_count_of_special_characters_by_length(special_char_counts, length):
    """ Plot the count of special characters in passwords of different lengths """
    special_chars = list(special_char_counts.keys())
    counts = list(special_char_counts.values())
    plt.figure(figsize=(10, 5))
    plt.bar(special_chars, counts, color='orange', alpha=0.7)
    plt.xlabel('Special Character')
    plt.ylabel('Counts')
    plt.title('Special Character Usage in Passwords of length {}'.format(length))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_count_of_numbers_by_length(numbers_counts, length):
    """ Plot the count of numbers in passwords of different lengths """
    numbers = list(numbers_counts.keys())
    counts = list(numbers_counts.values())
    plt.figure(figsize=(10, 5))
    plt.bar(numbers, counts, color='orange', alpha=0.7)
    plt.xlabel('Number')
    plt.ylabel('Counts')
    plt.title('Number Usage in Passwords of length {}'.format(length))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_count_of_upper_case_by_length(upper_case_counts, length):
    """ Plot the count of upper case characters in passwords of different lengths """
    upper_case = list(upper_case_counts.keys())
    counts = list(upper_case_counts.values())
    plt.figure(figsize=(10, 5))
    plt.bar(upper_case, counts, color='orange', alpha=0.7)
    plt.xlabel('Upper Case')
    plt.ylabel('Counts')
    plt.title('Upper Case Usage in Passwords of length {}'.format(length))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_count_of_lower_case_by_length(lower_case_counts, length):
    """ Plot the count of lower case characters in passwords of different lengths """
    lower_case = list(lower_case_counts.keys())
    counts = list(lower_case_counts.values())
    plt.figure(figsize=(10, 5))
    plt.bar(lower_case, counts, color='orange', alpha=0.7)
    plt.xlabel('Lower Case')
    plt.ylabel('Counts')
    plt.title('Lower Case Usage in Passwords of length {}'.format(length))
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_categories_bar_plot(statistics):
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
    plt.show()


def plot_categories_pie_plot_via_matplotlib(statistics):
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
    plt.show()

def plot_categories_pie_plot_via_plotly(statistics):
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
    fig.show()

if __name__ == '__main__':
    passwords, statistics = ps.analyze_passwords('rockyou2024-100K.txt')
    print_statistics(statistics)
    
    plot_password_length_histogram(statistics['length_percentages'])
    plot_ascii_usage_histogram(statistics['ascii_counts'])
    plot_categories_bar_plot(statistics)
    plot_categories_pie_plot_via_matplotlib(statistics)
    plot_categories_pie_plot_via_plotly(statistics)

    number_position_violin_plot(statistics['number_positions'])
    special_character_position_violin_plot(statistics['special_char_positions'])
    special_character_position_violin_plot_for_specific_characters(statistics['special_char_positions_per_char']['.'],'.')
    plot_year_histogram(statistics['year_counts'])

    plot_count_of_special_characters_by_length(statistics['count_of_special_characters_per_length_per_count_percentages'][8], 8)
    plot_count_of_numbers_by_length(statistics['count_of_numbers_per_length_per_count_percentages'][8], 8)
    plot_count_of_upper_case_by_length(statistics['count_of_upper_case_per_length_per_count_percentages'][8], 8)
    plot_count_of_lower_case_by_length(statistics['count_of_lower_case_per_length_per_count_percentages'][8], 8)
    
    
