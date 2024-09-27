import streamlit as st
import json
import plotly.express as px
import matplotlib.pyplot as plt
import pandas as pd
import plotly.io as pio
import numpy as np
import plotly.graph_objects as go
import os
import mpld3

def plot_password_strength_bins():

    with open('password_strength_bins.json', 'r') as f:
        bins_list = json.load(f)

    bins = np.array(bins_list)

    fig = plt.figure(figsize = (13, 4))

    plt.bar(range(10), bins, color ='midnightblue')

    plt.xlabel("Password Strength Score", fontsize=18)
    plt.ylabel("Number of Passwords", fontsize=18)
    plt.title("Password Scores Histogram", fontsize=23)
    plt.xticks(ticks = range(0, 10), labels=["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7",
                                             "7-8", "8-9", "9-10"], fontsize=14)


    st.pyplot(fig)

    st.write("""
             The Histogram above visualizes the distribution of password strengths. \n
             We can see that most people use a pretty simple passwords but also not too simple.
        """)

def plot_password_strength_scatter():

    st.subheader('Visualizing the strength of each password')

    df = pd.read_csv('password_strength_dataframe.csv')

    df['Index'] = df.index

    fig = px.scatter(df, x='Index', y='score', hover_data=['password', 'modified'])

    fig.update_layout(
        width=1300,  # Set width in pixels
        height=700,  # Set height in pixels
        # title="Passwords Strengths",
        # title_font_size = 25,
        font_color="black",
        # title_xanchor='center',
        # title_xref = "paper",
        # title_x=0.5,
        xaxis=dict(tickvals=[],
            title='',
        #     titlefont=dict(color='black', size=21)  # Change x-axis title font color to blue
        ),
        yaxis=dict(
            title='Password Strength',
            titlefont=dict(color='black', size=21),  # Change y-axis title font color to red
            tickfont=dict(color='black', size=16)
        ),
        margin=dict(l=5, r=5, t=5, b=5)  # Reduce margins
    )

    st.plotly_chart(fig)

    st.write("""
             The scatter plot above allows us to manually explore what kind of passwords
             achieve a good score and which get a bad score \n
             we can also see the modified version of the password, where we removed common patterns. \n
             Some passwords get a score of zero as they can be cracked in a matter of minutes using
             a simple predefined algorithm. notice that password that have the pattern of an email
             address fall in this category, because it is most likely identical to the users email
             or very close to it 
        """)


def static_strength_page():
    plot_password_strength_bins()

    plot_password_strength_scatter()