
import string
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly.express as px
import pandas as pd

date_pattern = r'^\b(?:(\d{1,2})([-/.]?)(\d{1,2})(\2(?:\d{2}|\d{4}))?)\b$'
email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'


directory_path = "password_strength_text_files/"
with open(directory_path + "word_list.txt", 'r') as f:
    word_list = f.readlines()
with open(directory_path + "cracking_number_list_with_dates.txt", 'r') as f:
    number_list = f.readlines()
with open(directory_path + "password_list.txt", 'r') as f:
    password_list = f.readlines()
with open(directory_path + "cracking_name_list.txt", 'r') as f:
    name_list = f.readlines()

word_set = set()
num_set = set()
pass_set = set()
name_set = set()
for w in word_list:
    word_set.add(w.replace('\n', ''))

for w in number_list:
    num_set.add(w.replace('\n', ''))

for w in password_list:
    pass_set.add(w.replace('\n', ''))

for w in name_list:
    name_set.add(w.replace('\n', '').lower())

word_list = sorted(list(word_set), key=len, reverse=True)

number_list = sorted(list(num_set), key=len, reverse=True)

password_list = sorted(list(pass_set), key=len, reverse=True)

name_list = sorted(list(name_set), key=len, reverse=True)

def give_character_type_score(password):
    types = np.zeros(4)
    for c in password:
        if c in string.ascii_uppercase:
            types[0] = 1
        if c in string.ascii_lowercase:
            types[1] = 1
        if c in string.punctuation:
            types[2] = 1
        if c in string.digits:
            types[3] = 1
    return np.count_nonzero(types) - 1


def give_different_character_score(password):
    character_set = set()
    for c in password:
        character_set.add(c)

    return len(character_set)

def contains_date(date_str):
    # Regular expression to match '02-04', '15/06', '21.7', and '0907'
    pattern = r'\b\d{1,2}([-/.]?)\d{1,2}\b'

    # Search the string for a date pattern
    return bool(re.search(pattern, date_str))


def longest_digit_substring(s):
    # Find all consecutive digit sequences
    digit_sequences = re.findall(r'\d+', s)

    # Return the longest one, or an empty string if none are found
    if digit_sequences:
        return len(max(digit_sequences, key=len))
    return 0


def longest_single_char_substring(s):
    max_length = 0
    max_char = ''

    current_char = ''
    current_length = 0

    for char in s:
        if char == current_char:
            current_length += 1
        else:
            if current_length > max_length:
                max_length = current_length
                max_char = current_char

            current_char = char
            current_length = 1

    # Final check at the end of the string
    if current_length > max_length:
        max_length = current_length
        max_char = current_char

    return max_length


def remove_pattern(password):
    password = password.lower()
    if len(password) > 5 and password[-4:] == '.com':
        password = password[:-4]
    for word in word_list:
        password = password.replace(word, "x")
    for name in name_list:
        password = password.replace(name, "x")
        password = password.replace(name.lower(), "x")
    for number in number_list:
        password = password.replace(number, "8")

    return password


def automatic_deny(password):
    lower_password = password.lower()
    if ((password in password_list) or (password in number_list) or
            (password in word_list)):
        return True
    if ((lower_password in password_list) or (lower_password in number_list) or
            (lower_password in word_list) or (lower_password in name_list)):
        return True
    if len(password) < 9 and len(password) <= sum(1 for char in password if char.isdigit()):
        return True
    if len(password) <= longest_single_char_substring(password):
        return True
    if re.search(date_pattern, password):
        return True

    return False

def almost_deny(password):
    if len(password) < 9 and len(password) <= sum(1 for char in password if char.isdigit()) + 1:
        return True
    if len(password) <= longest_single_char_substring(password) + 1:
        return True

    remove_last_letter = password[:-1]
    remove_first_letter = password[1:]
    remove_last_letter_lower = remove_last_letter.lower()
    remove_first_letter_lower = remove_first_letter.lower()
    if ((remove_last_letter in password_list) or (remove_last_letter in number_list) or
            (remove_last_letter in word_list) or (remove_last_letter in name_list)):
        return True
    if ((remove_first_letter in password_list) or (remove_first_letter in number_list) or
            (remove_first_letter in word_list) or (remove_first_letter in name_list)):
        return True
    if ((remove_last_letter_lower in password_list) or (remove_last_letter_lower in number_list) or
            (remove_last_letter_lower in word_list) or (remove_last_letter_lower in name_list)):
        return True
    if ((remove_first_letter_lower in password_list) or (remove_first_letter_lower in number_list) or
            (remove_first_letter_lower in word_list) or (remove_first_letter_lower in name_list)):
        return True


def calc_modified_password_score(p):
    password_score = give_different_character_score(p)
    password_score += len(p) * 1.9
    password_score -= longest_digit_substring(p)
    password_score -= longest_single_char_substring(p)
    if len(p) < 6:
        password_score -= 2
    if len(p) < 8:
        password_score -= 2
    if len(p) < 10:
        password_score -= 2
    if contains_date(p):
        password_score -= 6
    return min(password_score, 31)

def calc_no_special_password_score(p):
    password_score = give_different_character_score(p)
    password_score += len(p) * 1.9
    password_score -= (longest_single_char_substring(p) * 2)
    if len(p) < 6:
        password_score -= 2
    if len(p) < 8:
        password_score -= 2
    if len(p) < 10:
        password_score -= 2
    if contains_date(p):
        password_score -= 6
    return min(password_score, 30)

