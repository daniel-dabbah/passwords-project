import string
import numpy as np
import matplotlib.pyplot as plt
import re
import plotly.express as px
import pandas as pd
from password_strength_helper import (automatic_deny, give_character_type_score, calc_no_special_password_score,
                                      calc_modified_password_score, remove_pattern)

email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b'

def calculate_password_strength(password):
    password = password.replace("\n", "")
    if automatic_deny(password):
        return 0, password
    if re.fullmatch(email_pattern, password):
        return 0, password
    password_score = 2* give_character_type_score(password)

    password_score += calc_modified_password_score(password) * 0.4
    p = remove_pattern(password)

    no_specials = re.sub(r'[^a-zA-Z0-9]', '', password)
    if automatic_deny(no_specials):
        return min(max(0, (password_score * 0.15)), 10), p

    password_score += calc_no_special_password_score(no_specials)* 0.2
    p = remove_pattern(password)
    if automatic_deny(p):
        return min(max(0, (password_score * 0.16)), 10), p
    password_score += calc_modified_password_score(p)


    return min(max(0, (password_score * 0.187) - 1.8), 10), p


