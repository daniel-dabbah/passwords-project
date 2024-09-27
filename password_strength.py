import string
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly.express as px
import pandas as pd
from password_strength_helper import (automatic_deny, almost_deny, give_character_type_score, calc_no_special_password_score,
                                      calc_modified_password_score, remove_pattern)

email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'

def calculate_password_strength(password, with_details=False):

    password = password.replace("\n", "")
    if automatic_deny(password):
        if with_details:
            return 0, password, 0, 0, 0
        return 0, password
    if re.fullmatch(email_pattern, password):
        if with_details:
            return 0, password, 0, 0, 0
        return 0, password


    password_score = 3 * give_character_type_score(password)
    unmodified_score = calc_modified_password_score(password)
    p = remove_pattern(password)

    only_letters = re.sub(r'\d{2,}', '', password)
    only_letters = re.sub(r'[^A-Za-z0-9]+', '', only_letters)
    only_letters_score = calc_no_special_password_score(only_letters)

    remove_pattern_score = calc_modified_password_score(p)

    if automatic_deny(only_letters) or almost_deny(password):
        password_score += unmodified_score * 0.25
        final_score = min(max(0, (password_score * 0.3)), 10)

    elif automatic_deny(p):
        password_score += unmodified_score * 0.25
        password_score += only_letters_score * 0.25
        final_score = min(max(0, (password_score * 0.3)), 10)

    else:
        password_score += unmodified_score * 0.25
        password_score += only_letters_score * 0.25
        password_score += remove_pattern_score
        final_score = min(max(0, (password_score * 0.28) - 3.4), 10)

    if with_details:
        return final_score, p, unmodified_score, only_letters_score, remove_pattern_score

    return final_score, p



