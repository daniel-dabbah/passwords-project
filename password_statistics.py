import matplotlib
from tqdm import tqdm
import string
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.colors as mcolors
import numpy as np
import re
import math

# Define the valid ASCII characters
valid_ascii_chars = string.ascii_letters + string.digits + string.punctuation + " "
# Special characters we want to analyze
special_chars = "!@#$%^&*()-_=+[]{|;:'\,.<>?/`~ }"

def is_only_ascii(password):
    """Function to check if the password contains only valid ASCII characters"""
    return all(c in valid_ascii_chars for c in password)

def is_numeric(password):
    """Function to check if the password contains only numbers"""
    return password.isdigit()

def is_alphabet_only(password):
    """Function to check if the password contains only alphabetic characters"""
    return password.isalpha()

def is_lower_case_only(password):
    """Function to check if the password contains only lowercase characters"""
    return all(c.islower() for c in password)

def is_upper_case_only(password):
    """Function to check if the password contains only uppercase characters"""
    return all(c.isupper() for c in password)

def is_special_characters_only(password):
    """Function to check if the password contains only special characters"""
    return all(c in special_chars for c in password)

def is_lower_case_present(password):
    """Function to check if the password contains lowercase characters"""
    return any(c.islower() for c in password)

def is_upper_case_present(password):
    """Function to check if the password contains uppercase characters"""
    return any(c.isupper() for c in password)

def is_special_characters_present(password):
    """Function to check if the password contains special characters"""
    return any(c in special_chars for c in password)

def is_numbers_present(password):
    """Function to check if the password contains numbers"""
    return any(c.isdigit() for c in password)

def extract_num_of_special_characters(password):
    """Function to extract the number of special characters in the password"""
    return sum(1 for c in password if c in special_chars)

def extract_num_of_numbers(password):
    """Function to extract the number of numbers in the password"""
    return sum(1 for c in password if c.isdigit())

def extract_num_of_upper_case(password):
    """Function to extract the number of upper case characters in the password"""
    return sum(1 for c in password if c.isupper())

def extract_num_of_lower_case(password):
    """Function to extract the number of lower case characters in the password"""
    return sum(1 for c in password if c.islower())

def numbers_at_end(s):
    """Function to check if the numbers are only at the end of the string"""
    # This regex matches a string that contains only non-digit characters, 
    # followed by one or more digits at the end.
    pattern = r'^[^\d]*\d+$'
    return bool(re.match(pattern, s))

def extract_positions_of_numbers(password):
    """Function to extract the positions of numbers in the password"""
    length = len(password)
    for i, char in enumerate(password):
        if char.isdigit():
            if length > 1:
                position = i / (length - 1)
            else:
                position = 0
            return length, position

def extract_positions_of_special_characters(password):
    """Function to extract the positions of special characters in the password"""
    length = len(password)
    for i, char in enumerate(password):
        if char in special_chars:
            if length > 1:
                position = i / (length - 1)
            else:
                position = 0
            return length, position
        
def extract_positions_of_special_characters_per_char(password, char):
    """Function to extract the positions of special characters in the password"""
    length = len(password)
    for i, c in enumerate(password):
        if c == char:
            if length > 1:
                position = i / (length - 1)
            else:
                position = 0
            return length, position

def extract_year(password):
    """Function to extract the year from the password"""
    # Find all four-digit sequences in the password
    match = re.findall(r'\d{4}', password)
    # Check if any of the found sequences are within the range 1900-2030
    for year in match:
        if 1930 <= int(year) <= 2030:
            return int(year)
        
def calculate_entropy(password):
    """Calculate the entropy of a password based on the character sets it uses."""
    char_set_size = 0

    if is_lower_case_present(password):
        char_set_size += 26  # Lowercase letters
    if is_upper_case_present(password):
        char_set_size += 26  # Uppercase letters
    if is_numbers_present(password):
        char_set_size += 10  # Numbers
    if is_special_characters_present(password):
        char_set_size += len(special_chars)  # Special characters

    password_length = len(password)
    if char_set_size == 0 or password_length == 0:
        return 0  # Entropy is zero if the password contains no characters or has zero length

    entropy = password_length * math.log2(char_set_size)
    return entropy


def analyze_passwords(file_path):
    """Function to analyze the passwords in the file and return the statistics
       We read the file line by line and analyze each password to get the statistics
       and save them in a dictionary. We also store the passwords in a list for further analysis.
    """

    # List to store all the passwords
    passwords = []
    # List to store the entropies of the passwords
    entropies = [] 
    # Total password count
    total_password_count = 0

    # Only one type of character Count
    only_lower_case_count = 0
    only_upper_case_count = 0
    only_numbers_count = 0
    only_special_characters_count = 0

    # Contains (not only) Count
    contains_lower_case_count = 0
    contains_upper_case_count = 0
    contains_special_characters_count = 0
    contains_numbers_count = 0

    # Combinations Count
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

    # More complex combinations
    upper_case_only_beginning_count = 0
    numbers_only_at_end_count = 0
    
    # Dictionary to store the counts of passwords of different lengths
    length_counts = {length: 0 for length in range(4, 31)}

    # Dictionary to store the counts of ASCII characters in passwords
    ascii_counts = {char: 0 for char in valid_ascii_chars}

    # Disctionary to store years found in passwords
    years_counts = {year: 0 for year in range(1930, 2031)}

    # Positions of numbers and special characters in passwords
    number_positions = []
    special_char_positions = []
    
    # Positions of special characters per character in passwords
    special_char_positions_per_char = {char: [] for char in special_chars}

    # Count of special characters per length
    count_of_passwords_with_special_characters_per_length = {length: 0 for length in range(4, 31)}
    count_of_passwords_with_numbers_per_length = {length: 0 for length in range(4, 31)}
    count_of_passwords_with_upper_case_per_length = {length: 0 for length in range(4, 31)}
    count_of_passwords_with_lower_case_per_length = {length: 0 for length in range(4, 31)}

    # Count of special characters per length per count
    count_of_special_characters_per_length_per_count = {length: {} for length in range(4, 31)}
    for length in count_of_special_characters_per_length_per_count:
        for count in range(1, length+1):
            count_of_special_characters_per_length_per_count[length][count] = 0
    
    # Count of numbers per length per count
    count_of_numbers_per_length_per_count = {length: {} for length in range(4, 31)}
    for length in count_of_numbers_per_length_per_count:
        for count in range(1, length+1):
            count_of_numbers_per_length_per_count[length][count] = 0

    # Count of upper case per length per count
    count_of_upper_case_per_length_per_count = {length: {} for length in range(4, 31)}
    for length in count_of_upper_case_per_length_per_count:
        for count in range(1, length+1):
            count_of_upper_case_per_length_per_count[length][count] = 0
    
    # Count of lower case per length per count
    count_of_lower_case_per_length_per_count = {length: {} for length in range(4, 31)}
    for length in count_of_lower_case_per_length_per_count:
        for count in range(1, length+1):
            count_of_lower_case_per_length_per_count[length][count] = 0

    entropy_by_length = {}  # Dictionary to store entropies by password length

    char_counts_for_log_likelihood = {char: 0 for char in valid_ascii_chars}  # Initialize counts to 0
    total_chars = 0  # Total number of characters processed
    
    """ Going Through Passwords Line by Line """
    # Read the file line by line
    with open(file_path, 'r', encoding='latin-1', errors='ignore') as file:
        for line in tqdm(file, desc="Analyzing passwords"):
            
            # Check password is valid ASCII
            password = line.strip()
            if not is_only_ascii(password):
                continue
            
            # Add password to the list
            passwords.append(password)
            # Increment total password count
            total_password_count += 1

            # Calculate entropy for the password
            entropy = calculate_entropy(password)
            entropies.append(entropy)

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
            
            """comibnations"""
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

            
            """ More Complex Combinations """
            # Upper case only at the beginning
            if len(password) > 1 and password[0].isupper() and all(c.islower() or c.isdigit() or c in special_chars for c in password[1:]):
                upper_case_only_beginning_count += 1

            # Numbers only at the end
            if len(password) > 1 and is_numbers_present(password) and not is_numeric(password):
                if numbers_at_end(password):
                    numbers_only_at_end_count += 1

            """ Password Lengths """
            if 4 <= len(password) <= 30:
                length_counts[len(password)] += 1
        
            """ ASCII Characters """
            for char in password:
                if char in ascii_counts:
                    ascii_counts[char] += 1

            """ Special Characters and Numbers Positions """
            if is_numeric(password) or is_alphabet_only(password) or len(password) > 30 or len(password) < 6:
                continue
            if is_numbers_present(password):
                number_position = extract_positions_of_numbers(password)
                number_positions.append(number_position)
            if is_special_characters_present(password):
                special_char_position = extract_positions_of_special_characters(password)
                special_char_positions.append(special_char_position)  

            """ Special Characters Positions Per Character """
            if is_special_characters_present(password):
                for char in special_chars:
                    if char in password:
                        special_char_position = extract_positions_of_special_characters_per_char(password, char)
                        special_char_positions_per_char[char].append(special_char_position)

            """ Years """
            if is_numbers_present(password):
                year = extract_year(password)
                if year:
                    years_counts[year] += 1
            
            """ Count of special characters, numbers, upper case, and lower case per length """
            if is_special_characters_present(password):
                count_of_passwords_with_special_characters_per_length[len(password)] += 1
                count_of_special_characters_per_length_per_count[len(password)][extract_num_of_special_characters(password)] += 1
            if is_numbers_present(password):
                count_of_passwords_with_numbers_per_length[len(password)] += 1
                count_of_numbers_per_length_per_count[len(password)][extract_num_of_numbers(password)] += 1
            if is_upper_case_present(password):
                count_of_passwords_with_upper_case_per_length[len(password)] += 1
                count_of_upper_case_per_length_per_count[len(password)][extract_num_of_upper_case(password)] += 1
            if is_lower_case_present(password):
                count_of_passwords_with_lower_case_per_length[len(password)] += 1
                count_of_lower_case_per_length_per_count[len(password)][extract_num_of_lower_case(password)] += 1
            
            """ Entropy by Length """
            # Group entropies by password length
            password_length = len(password)
            if password_length not in entropy_by_length:
                entropy_by_length[password_length] = []
            entropy_by_length[password_length].append(entropy)

            """ Log-Likelihood """
            for char in line:
                if char in char_counts_for_log_likelihood:  # Only count characters in the ASCII range
                    char_counts_for_log_likelihood[char] += 1
                    total_chars += 1

    """ If there are no passwords of a certain length, set the count to 1 to avoid division by zero """
    for i in range(4, 31):
        if count_of_passwords_with_special_characters_per_length[i] == 0:
            count_of_passwords_with_special_characters_per_length[i] = 1
        if count_of_passwords_with_numbers_per_length[i] == 0:
            count_of_passwords_with_numbers_per_length[i] = 1
        if count_of_passwords_with_upper_case_per_length[i] == 0:
            count_of_passwords_with_upper_case_per_length[i] = 1
        if count_of_passwords_with_lower_case_per_length[i] == 0:
            count_of_passwords_with_lower_case_per_length[i] = 1

    """ Calculate the percentages of special characters, numbers, upper case, and lower case per length per count """
    count_of_special_characters_per_length_per_count_percentages = {}
    count_of_numbers_per_length_per_count_percentages = {}
    count_of_upper_case_per_length_per_count_percentages = {}
    count_of_lower_case_per_length_per_count_percentages = {}
    for length in range(4, 31):
        count_of_special_characters_per_length_per_count_percentages[length] = {}
        count_of_numbers_per_length_per_count_percentages[length] = {}
        count_of_upper_case_per_length_per_count_percentages[length] = {}
        count_of_lower_case_per_length_per_count_percentages[length] = {}
        for count in range(1, length+1):
            count_of_special_characters_per_length_per_count_percentages[length][count] = (count_of_special_characters_per_length_per_count[length][count] / count_of_passwords_with_special_characters_per_length[length]) * 100
            count_of_numbers_per_length_per_count_percentages[length][count] = (count_of_numbers_per_length_per_count[length][count] / count_of_passwords_with_numbers_per_length[length]) * 100
            count_of_upper_case_per_length_per_count_percentages[length][count] = (count_of_upper_case_per_length_per_count[length][count] / count_of_passwords_with_upper_case_per_length[length]) * 100
            count_of_lower_case_per_length_per_count_percentages[length][count] = (count_of_lower_case_per_length_per_count[length][count] / count_of_passwords_with_lower_case_per_length[length]) * 100

    """ Calculate probabilities (normalize the counts) for log-likelihood """
    char_probabilities_for_log_likelihood = {}
    for char, count in char_counts_for_log_likelihood.items():
        if total_chars > 0:
            char_probabilities_for_log_likelihood[char] = count / total_chars
        else:
            char_probabilities_for_log_likelihood[char] = 0.0  # Avoid division by zero if file is empty

    
    """ Return all the statistics in a dictionary """
    return passwords, {

        # Total password count
        "total": total_password_count,

        # Type of passwords dictionaries
        "lower_case_only_percentage": (only_lower_case_count / total_password_count) * 100,
        "upper_case_only_percentage": (only_upper_case_count / total_password_count) * 100,
        "special_characters_only_percentage": (only_special_characters_count / total_password_count) * 100,
        "numbers_only_percentage": (only_numbers_count / total_password_count) * 100,
        "contains_lower_case_percentage": (contains_lower_case_count / total_password_count) * 100,
        "contains_upper_case_percentage": (contains_upper_case_count / total_password_count) * 100,
        "contains_special_characters_percentage": (contains_special_characters_count / total_password_count) * 100,
        "contains_numbers_percentage": (contains_numbers_count / total_password_count) * 100,
        "lower_case_and_numbers_percentage": (lower_case_and_numbers_count / total_password_count) * 100,
        "lower_case_and_special_characters_percentage": (lower_case_and_special_characters_count / total_password_count) * 100,
        "lower_case_and_upper_case_percentage": (lower_case_and_upper_case_count / total_password_count) * 100,
        "upper_case_and_numbers_percentage": (upper_case_and_numbers_count / total_password_count) * 100,
        "upper_case_and_special_characters_percentage": (upper_case_and_special_characters_count / total_password_count) * 100,
        "special_characters_and_numbers_percentage": (special_characters_and_numbers_count / total_password_count) * 100,
        "lower_upper_case_and_numbers_percentage": (lower_upper_case_and_numbers_count / total_password_count) * 100,
        "lower_upper_case_and_special_characters_percentage": (lower_upper_case_and_special_characters_count / total_password_count) * 100,
        "lower_special_characters_and_numbers_percentage": (lower_special_characters_and_numbers_count / total_password_count) * 100,
        "upper_special_characters_and_numbers_percentage": (upper_special_characters_and_numbers_count / total_password_count) * 100,
        "all_character_types_percentage": (all_character_types_count / total_password_count) * 100,
        "upper_case_only_beginning_percentage": (upper_case_only_beginning_count / total_password_count) * 100,
        "numbers_only_at_end_percentage": (numbers_only_at_end_count / total_password_count) * 100,

        # Password lengths percentages
        "length_percentages": {length: (count / total_password_count) * 100 for length, count in length_counts.items()},

        # ASCII characters percentages
        "ascii_counts": {char: (count / total_password_count) * 100 for char, count in ascii_counts.items()},

        # Years counts and percentages
        "year_percentages": {year: (count / total_password_count) * 100 for year, count in years_counts.items()},
        "year_counts": years_counts,

        # Positions of numbers and special characters
        "number_positions": number_positions,
        # Positions of special characters
        "special_char_positions": special_char_positions,
        # Positions of special characters per character
        "special_char_positions_per_char": special_char_positions_per_char,
        
        # Count of passwords that used special characters, numbers, upper case, and lower case per length
        "count_of_special_characters_per_length": count_of_passwords_with_special_characters_per_length,
        "count_of_numbers_per_length": count_of_passwords_with_numbers_per_length,
        "count_of_upper_case_per_length": count_of_passwords_with_upper_case_per_length,
        "count_of_lower_case_per_length": count_of_passwords_with_lower_case_per_length,

        # Count of special characters, numbers, upper case, and lower case per length per count
        "count_of_special_characters_per_length_per_count": count_of_special_characters_per_length_per_count,
        "count_of_numbers_per_length_per_count": count_of_numbers_per_length_per_count,
        "count_of_upper_case_per_length_per_count": count_of_upper_case_per_length_per_count,
        "count_of_lower_case_per_length_per_count": count_of_lower_case_per_length_per_count,

        # Percentages of the latter - special characters, numbers, upper case, and lower case per length per count
        "count_of_special_characters_per_length_per_count_percentages": count_of_special_characters_per_length_per_count_percentages,
        "count_of_numbers_per_length_per_count_percentages": count_of_numbers_per_length_per_count_percentages,
        "count_of_upper_case_per_length_per_count_percentages": count_of_upper_case_per_length_per_count_percentages,
        "count_of_lower_case_per_length_per_count_percentages": count_of_lower_case_per_length_per_count_percentages,
        
        "entropies": entropies,
        "entropy_by_length": entropy_by_length,

        "char_probabilities_for_log_likelihood": char_probabilities_for_log_likelihood
    }