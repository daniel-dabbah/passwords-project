import streamlit as st

def website_introduction():
    st.title('Welcome to the Password Strength Analysis Dashboard')

    st.write('''
    In an age of increasing data breaches, understanding password strength is essential to enhancing personal and organizational security.
    Our interactive dashboard provides a comprehensive analysis of breached passwords and offers real-time insights on the security of passwords users create.
    Whether you’re curious about common password vulnerabilities or want to check the robustness of your own password, this tool is designed to help.
    ''')

    st.write('''
    ## Explore Our Dashboard
    Our application offers three key sections:
    
    1. **Breached Password Analysis**:  
       Analyze a dataset of breached passwords to uncover patterns in password creation, including length distribution, character composition, and common vulnerabilities. Visualizations reveal key insights into how passwords from real-world breaches compare in terms of strength and predictability.
    
    2. **Check Your Password Statistics**:  
       Input your password to see how it compares to others in terms of length, character variety, and more. This page provides detailed statistics, showing how your password ranks against breached passwords and highlights areas where it could be improved.
    
    3. **Check Your Password Strength**:  
       Evaluate the strength of your password in real-time. Our algorithm calculates an entropy score, detects patterns or weaknesses, and suggests improvements. The password strength is measured on a scale of 0 to 10, helping you ensure that your password is resilient against cracking attempts.
    ''')

    st.write('''
    ## Key Features:
    - **Visualize Common Password Patterns**: Discover patterns like most used password lengths, popularity of certain characters, and typical positions of numbers or special symbols.
    - **Real-Time Feedback**: Input a password and instantly receive feedback on its strength, character usage, and entropy.
    - **Clustering Analysis**: Explore password clusters based on factors like entropy, MinHash similarity, and N-gram log-likelihood, helping you understand how your password stands in comparison to others.
    ''')

    st.write('''
    ## Our Mission:
    The goal of this tool is not only to highlight weaknesses in common password creation practices but also to help users create stronger, more resilient passwords.
    We aim to empower users with the knowledge they need to stay secure online by providing actionable insights based on extensive analysis of real-world password data.
    ''')

    st.write('''
    ## Get Started
    Navigate through the different sections of the dashboard and start exploring the insights and security tips we provide. Whether you are analyzing breached passwords or testing your own, our platform will guide you in understanding the key elements of password security.
    ''')

if __name__ == "__main__":
    website_introduction()
