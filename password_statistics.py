import matplotlib
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors


# Define the valid ASCII characters (printable characters minus space and last three non-printable ones)
valid_ascii_chars = string.ascii_letters + string.digits + string.punctuation + " "

# Special characters we want to analyze
special_chars = "!@#$%^&*()-_=+[]{}|;:'\",.<>?/`~ "

# Function to check if the password contains only valid ASCII characters
def is_only_ascii(password):
    return all(c in valid_ascii_chars for c in password)

# Function to check if the password is numeric
def is_numeric(password):
    return password.isdigit()

# Function to check if the password is alphabet-only
def is_alphabet_only(password):
    return password.isalpha()

def is_lower_case_only(password):
    return all(c.islower() for c in password)

def is_upper_case_only(password):
    return all(c.isupper() for c in password)

def is_special_characters_only(password):
    return all(c in special_chars for c in password)

def is_lower_case_present(password):
    return any(c.islower() for c in password)

def is_upper_case_present(password):
    return any(c.isupper() for c in password)

def is_special_characters_present(password):
    return any(c in special_chars for c in password)

def is_numbers_present(password):
    return any(c.isdigit() for c in password)

# Function to extract the positions of numbers and special characters in passwords
def extract_positions(passwords):
    number_positions = []
    special_char_positions = []
    for password in passwords:
        if is_numeric(password) or is_alphabet_only(password):
            continue
        length = len(password)
        if length > 30 or length < 6:  # Filter out passwords longer than 30 characters
            continue
        for i, char in enumerate(password):
            if length > 1:  # Avoid division by zero
                position = i / (length - 1)  # Normalize position to range [0, 1]
            else:
                position = 0
            if char.isdigit():
                number_positions.append((length, position))
            if char in special_chars:
                special_char_positions.append((length, position))
    return number_positions, special_char_positions


# Additional analysis functions
def analyze_passwords(file_path):
    passwords = []
    total_passwords = 0
    only_lower_case_count = 0
    only_upper_case_count = 0
    only_special_characters_count = 0
    only_numbers_count = 0
    contains_lower_case_count = 0
    contains_upper_case_count = 0
    contains_special_characters_count = 0
    contains_numbers_count = 0

    lower_case_and_numbers_count = 0
    lower_case_and_special_characters_count = 0
    lower_case_and_upper_case_count = 0
    upper_case_and_numbers_count = 0
    upper_case_and_special_characters_count = 0
    special_characters_and_numbers_count = 0
    lower_upper_case_and_numbers_count = 0
    lower_upper_case_and_special_characters_count = 0
    lower_special_characters_and_numbers_count = 0
    upper_special_characters_and_numbers_count = 0
    all_character_types_count = 0

    upper_case_only_beginning_count = 0
    

    length_counts = {length: 0 for length in range(4, 31)}
    ascii_counts = {char: 0 for char in valid_ascii_chars}

    with open(file_path, 'r', encoding='latin-1', errors='ignore') as file:
        for line in tqdm(file, desc="Analyzing passwords"):
            password = line.strip()
            
            if not is_only_ascii(password):
                continue

            passwords.append(password)
            
            total_passwords += 1

            # Contains Only 
            if is_lower_case_only(password):
                only_lower_case_count += 1
            
            if is_upper_case_only(password):
                only_upper_case_count += 1

            if is_numeric(password):
                only_numbers_count += 1

            if is_special_characters_only(password):
                only_special_characters_count += 1

            # Contains (not only)
            if is_lower_case_present(password):
                contains_lower_case_count += 1

            if is_upper_case_present(password):
                contains_upper_case_count += 1

            if is_special_characters_present(password):
                contains_special_characters_count += 1

            if is_numbers_present(password):
                contains_numbers_count += 1
            
            # Combinations

            # Only lower case and numbers
            if all(c.islower() or c.isdigit() for c in password):
                if any(c.islower() for c in password) and any(c.isdigit() for c in password):
                    lower_case_and_numbers_count += 1
            
            # Only lower case and special characters
            if all(c.islower() or c in special_chars for c in password):
                if any(c.islower() for c in password) and any(c in special_chars for c in password):
                    lower_case_and_special_characters_count += 1
            
            # Only lower case and upper case
            if all(c.islower() or c.isupper() for c in password):
                if any(c.islower() for c in password) and any(c.isupper() for c in password):
                    lower_case_and_upper_case_count += 1

            # Only upper case and numbers
            if all(c.isupper() or c.isdigit() for c in password):
                if any(c.isupper() for c in password) and any(c.isdigit() for c in password):
                    upper_case_and_numbers_count += 1
            
            # Only upper case and special characters
            if all(c.isupper() or c in special_chars for c in password):
                if any(c.isupper() for c in password) and any(c in special_chars for c in password):
                    upper_case_and_special_characters_count += 1
            
            # Only special characters and numbers
            if all(c in special_chars or c.isdigit() for c in password):
                if any(c in special_chars for c in password) and any(c.isdigit() for c in password):
                    special_characters_and_numbers_count += 1
            
            # Only lower case, upper case, and numbers
            if all(c.islower() or c.isupper() or c.isdigit() for c in password):
                if any(c.islower() for c in password) and any(c.isupper() for c in password) and any(c.isdigit() for c in password):
                    lower_upper_case_and_numbers_count += 1

            # Only lower case, upper case, and special characters
            if all(c.islower() or c.isupper() or c in special_chars for c in password):
                if any(c.islower() for c in password) and any(c.isupper() for c in password) and any(c in special_chars for c in password):
                    lower_upper_case_and_special_characters_count += 1
            
            # Only lower case, special characters, and numbers
            if all(c.islower() or c in special_chars or c.isdigit() for c in password):
                if any(c.islower() for c in password) and any(c in special_chars for c in password) and any(c.isdigit() for c in password):
                    lower_special_characters_and_numbers_count += 1
            
            # Only upper case, special characters, and numbers
            if any(c.isupper() for c in password) and any(c in special_chars for c in password) and any(c.isdigit() for c in password):
                upper_special_characters_and_numbers_count += 1

            # All character types
            if all(c.islower() or c.isupper() or c.isdigit() or c in special_chars for c in password):
                if any(c.islower() for c in password) and any(c.isupper() for c in password) and any(c.isdigit() for c in password) and any(c in special_chars for c in password):
                    all_character_types_count += 1

            # More complex combinations
            
            if len(password) > 1 and password[0].isupper() and all(c.islower() or c.isdigit() or c in special_chars for c in password[1:]):
                upper_case_only_beginning_count += 1

            if 4 <= len(password) <= 30:
                length_counts[len(password)] += 1
            
            for char in password:
                if char in ascii_counts:
                    ascii_counts[char] += 1

    return passwords, {
        "total": total_passwords,
        "lower_case_only_percentage": (only_lower_case_count / total_passwords) * 100,
        "upper_case_only_percentage": (only_upper_case_count / total_passwords) * 100,
        "special_characters_only_percentage": (only_special_characters_count / total_passwords) * 100,
        "numbers_only_percentage": (only_numbers_count / total_passwords) * 100,
        "contains_lower_case_percentage": (contains_lower_case_count / total_passwords) * 100,
        "contains_upper_case_percentage": (contains_upper_case_count / total_passwords) * 100,
        "contains_special_characters_percentage": (contains_special_characters_count / total_passwords) * 100,
        "contains_numbers_percentage": (contains_numbers_count / total_passwords) * 100,
        "lower_case_and_numbers_percentage": (lower_case_and_numbers_count / total_passwords) * 100,
        "lower_case_and_special_characters_percentage": (lower_case_and_special_characters_count / total_passwords) * 100,
        "lower_case_and_upper_case_percentage": (lower_case_and_upper_case_count / total_passwords) * 100,
        "upper_case_and_numbers_percentage": (upper_case_and_numbers_count / total_passwords) * 100,
        "upper_case_and_special_characters_percentage": (upper_case_and_special_characters_count / total_passwords) * 100,
        "special_characters_and_numbers_percentage": (special_characters_and_numbers_count / total_passwords) * 100,
        "lower_upper_case_and_numbers_percentage": (lower_upper_case_and_numbers_count / total_passwords) * 100,
        "lower_upper_case_and_special_characters_percentage": (lower_upper_case_and_special_characters_count / total_passwords) * 100,
        "lower_special_characters_and_numbers_percentage": (lower_special_characters_and_numbers_count / total_passwords) * 100,
        "upper_special_characters_and_numbers_percentage": (upper_special_characters_and_numbers_count / total_passwords) * 100,
        "all_character_types_percentage": (all_character_types_count / total_passwords) * 100,
        "upper_case_only_beginning_percentage": (upper_case_only_beginning_count / total_passwords) * 100,
        "length_percentages": {length: (count / total_passwords) * 100 for length, count in length_counts.items()},
        "ascii_counts": {char: (count / total_passwords) * 100 for char, count in ascii_counts.items()}
    }

# Usage
# passwords, statistics = analyze_passwords('crackstation-human-only//realhuman_phill.txt')
passwords, statistics = analyze_passwords('passwords.txt')

# Print statistics
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
print(f"Upper Case Only at Beginning: {statistics['upper_case_only_beginning_percentage']:.2f}%")

# Prepare data for bar plot

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



labels = [
    "Lower Case Only", "Upper Case Only", "Numbers Only", 
    "Lower and Numbers", "Lower and Special",
    "Upper and Numbers", "Upper and Special", "Lower Upper and Numbers", 
    "Special and Numbers", "Lower Upper and Special", "Lower Special and Numbers",
    "Upper Special and Numbers", "All Character Types", "Lower and Upper"
]

"""
other = statistics['special_characters_and_numbers_percentage'] + \
    statistics['lower_upper_case_and_special_characters_percentage'] + \
    statistics['lower_special_characters_and_numbers_percentage'] + \
    statistics['upper_special_characters_and_numbers_percentage'] + \
    statistics['all_character_types_percentage'] + \
    statistics['lower_case_and_upper_case_percentage']
    """

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
    statistics['lower_case_and_upper_case_percentage']
]


# Function to selectively show percentages
def autopct_function(pct):
    return ('%1.1f%%' % pct) if pct > 1 else ('%1.1f%%' % pct)

# Function to selectively show percentages with custom font sizes
def autopct_function(pct):
    return ('%1.1f%%' % pct) if pct > 1 else ('%1.1f%%' % pct)

# Plot the pie chart
plt.figure(figsize=(12, 8))
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


# Plot the pie chart using plotly
import plotly.express as px
import plotly.graph_objects as go

matplotlib.use('TkAgg')

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

# Show the figure
fig.show()


# Plotting Password Length Histogram
def plot_password_length_histogram(length_percentages):
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

# Plotting Password Length Histogram
def plot_password_length_histogram(length_percentages):
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

# Plotting ASCII Character Usage Histogram
def plot_ascii_usage_histogram(ascii_counts):
    chars = list(ascii_counts.keys())
    percentages = list(ascii_counts.values())
    plt.figure(figsize=(12, 8))
    plt.bar(chars, percentages, color='green', alpha=0.7)
    plt.xlabel('ASCII Character')
    plt.ylabel('Percentage')
    plt.title('ASCII Character Usage in Passwords')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()

"""
# Extract positions
number_positions, special_char_positions = extract_positions(passwords)

# Prepare data for plotting
number_plot_df = pd.DataFrame(number_positions, columns=['Total length of password', 'Normalized Position'])
special_char_plot_df = pd.DataFrame(special_char_positions, columns=['Total length of password', 'Normalized Position'])

# colors = list(mcolors.TABLEAU_COLORS.values())
single_color = "skyblue"  # You can change this to any color you prefer


# Plot using seaborn for numbers
plt.figure(figsize=(12, 8))
sns.violinplot(x='Total length of password', y='Normalized Position', data=number_plot_df, inner=None, scale='width', color=single_color)
plt.title('Position of Numbers in Passwords by Length')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# Plot using seaborn for special characters
plt.figure(figsize=(12, 8))
sns.violinplot(x='Total length of password', y='Normalized Position', data=special_char_plot_df, inner=None, scale='width', color=single_color)
plt.title('Position of Special Characters in Passwords by Length')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
"""

# Plot the histograms
plot_password_length_histogram(statistics['length_percentages'])
plot_ascii_usage_histogram(statistics['ascii_counts'])


