# A Needle in a Data Haystack – 678978 – Final Project Report

## Password Strength Analysis Dashboard

### Team Members

- **Dana Aviran: [dana.aviran@mail.huji.ac.il](mailto:dana.aviran@mail.huji.ac.il)**
  
- **Michael Svitlizky:  [michael.svitlizky@mail.huji.ac.il](mailto:michael.svitlizky@mail.huji.ac.il)**
  
- **Daniel Dabbah: [Daniel.dabbah@mail.huji.ac.il](mailto:Daniel.dabbah@mail.huji.ac.il)**

- **Carmit Yehudai: [Carmitch.yehudai@mail.huji.ac.il](mailto:Carmitch.yehudai@mail.huji.ac.il)**



### Problem Description
The increasing frequency of data breaches highlights the vulnerability of commonly used passwords. Weak and predictable passwords make users susceptible to attacks, with hackers easily exploiting them through brute-force techniques or dictionaries of popular passwords. The goal of our project is to create an interactive dashboard that provides users with a comprehensive analysis of their password strength. This analysis uses entropy, clustering algorithms, and statistical models to assess the predictability and security of passwords. The dashboard not only evaluates the password’s current strength but also offers actionable suggestions for improvement.

### Key Questions
- What common patterns do people use in password creation?
- How can we quantify the strength of a password?
- What guidance can we offer users to help them create stronger, more secure passwords?

### Datasets
For this project, we utilized several large datasets comprising both breached passwords and common word lists. These datasets were essential for evaluating password strength, identifying common patterns, and developing cracking algorithms:
- **Rockyou2024**: The most recent and largest collection of passwords, containing 10 billion entries along with word dictionaries. Requires cleaning and ordering. 45GB.
- **Pwnd Passwords**: A dataset of 500 million passwords hashed with MD5 and NTLM, including their frequency counts. Serves as our test set for evaluating our cracking capabilities. 10.3 GB.
- **Antipublic breach dataset**: About 120 million passwords from various breaches. 2 GB.
- **RockYou**: 14 million passwords with counts from the 2009 RockYou breach. 174 MB.
- **Pwdb 100K Top Passwords**: The 100,000 most common passwords with their counts.

### Our Solution - Password Analysis Dashboard
We implemented a wide range of methodologies and provided users with dynamic visualizations that make password strength assessment both engaging and informative. The dashboard was built using Streamlit, a Python-based framework for creating interactive web applications.

#### General Password Analysis
Through the whole project, the analysis focuses on the Rockyou 2024 dataset, which contains over 10 billion passwords. To make the analysis more manageable with the available tools, we randomly selected 1 million passwords as a representative sample. Minimal preprocessing was applied, filtering out passwords longer than 30 characters to better reflect human-created passwords, and retaining only those containing ASCII characters. After reviewing various password breach datasets, we determined that this dataset was the most relevant for modern websites' password requirements, as it featured longer passwords with a wider range of character types (uppercase letters, numbers, and special characters), which older datasets lacked. Even with this more secure dataset, we uncovered significant insights into common password creation patterns, revealing that many passwords remain insecure. The visualizations of this dataset provided valuable insights into these behaviors, helping users recognize patterns that may compromise password strength.

#### Insights Revealed in Our Analysis
- **Password Length Distribution**: Our analysis revealed a curve-like distribution of password lengths, indicating a "sweet spot" that most users gravitate toward. Shorter and longer passwords were less common, forming the tails of the distribution. This pattern provides insights into user behavior and vulnerabilities in password selection.
- **Character Composition**: Lowercase letters were overwhelmingly popular, while special characters like '?' or '>' were used less frequently, indicating an opportunity to improve password security by promoting their use. The most frequent password category combined lowercase letters and numbers, followed by passwords containing only lowercase letters. More complex combinations, such as those incorporating uppercase letters, special characters, and numbers, were relatively rare, highlighting the need for stronger password creation habits.
- **Number and Special Character Placement**: Numbers were often placed at the beginning or end of passwords, while special characters (e.g., from the set: !@#$%^&*()-_=+[]{|;:',.<>?/`~) were more likely to appear in the middle. These patterns remained consistent across different password lengths, revealing common habits in password creation.
- **Year Usage**: Many passwords included years, likely representing birth years or significant dates. Years between 1960-2010 were most common, likely reflecting the age distribution of internet users. This trend highlights how personal information often influences password creation.
- **Entropy and Password Strength**: By calculating entropy, we measured the unpredictability and strength of passwords. The distribution formed a curve, with most passwords falling in the middle range of complexity. Very weak and very strong passwords were less common, but a notable spike in high-entropy passwords suggested the presence of randomly generated passwords, possibly from password managers.

### Clustering Password Analysis
Our clustering analysis aimed to categorize passwords based on various metrics, providing insights into their predictability, security, and structural patterns.
- **Clustering by entropy**: Entropy is a fundamental measure in password security, representing the randomness or unpredictability of a password. It is calculated based on both the length of the password and the diversity of characters used, such as lowercase and uppercase letters, numbers, and special symbols. A larger character set increases the number of potential combinations, making the password more secure. We calculated entropy for each password using the formula: Entropy = Password Length × log₂(Character Set Size). Passwords with an entropy difference of less than 0.5 were grouped into clusters. For users, we provided a visual representation of which entropy cluster their password belonged to, along with examples of similar passwords. This helped users understand the complexity of their password and encouraged them to improve entropy by increasing password length and incorporating diverse characters.
- **Clustering by N-gram log-likelihood**: N-gram log-likelihood is a statistical technique that evaluates the probability of a sequence of characters appearing within a password based on trained language models. This method is particularly useful for detecting passwords that follow predictable linguistic or character sequence patterns, which may render them more vulnerable to attacks. While entropy measures the randomness and complexity of a password by considering factors such as length and character diversity, it does not account for semantic patterns or common linguistic structures that may render a password more susceptible to guessing. For example, the password "WelcomeBack2022!" may exhibit high entropy due to its inclusion of uppercase and lowercase letters, numbers, and special characters. However, it consists of common words and a predictable sequence, which results in a higher n-gram log-likelihood. The higher log-likelihood indicates that the password aligns closely with popular password patterns and is therefore more meaningful and potentially more vulnerable to attacks that exploit linguistic predictability. The n-gram model captures the likelihood of observing an n-character sequence (e.g., bigrams or trigrams) by analyzing large corpora of text or, in our case, password data. For this analysis, we utilized bigrams (2-character sequences) to calculate log-likelihood scores, allowing us to cluster passwords that exhibit similar linguistic patterns.
- **Clustering by MinHash**: MinHash is a technique that efficiently estimates the similarity between sets of elements, making it well-suited for password clustering. By converting each password into a MinHash signature, we were able to compute approximate Jaccard similarities between passwords or clusters. This method allowed us to group structurally similar passwords, even when the order or specific characters differed. To help users better understand these clusters, we used Multidimensional Scaling (MDS) to project the high-dimensional similarity matrix onto a 2D plane. The result was a visual scatter plot where cluster sizes and distances between them indicated the similarity of passwords. Users could explore clusters, hovering over points to see example passwords and their commonalities.

### Input User Password Analysis
The dashboard includes two pages which allow users to input their own passwords and receive real-time interactive analysis and feedback on strength and improvements.
- **Cluster Visualization (Entropy, MinHash, and N-gram Log-Likelihood)**: Users could input a password to see which cluster it fell into. Each cluster, calculated using entropy, MinHash, or n-gram log-likelihood, was displayed in scatter plots. The user's password was highlighted in red, while similar clusters were shown in blue. This helped users understand how common or unique their password was relative to others.
- **Real-Time Password Strength Analysis**: As users typed in passwords, the dashboard provided instant feedback on their strength through a live-updating strength bar, reflecting how changes to length, character types, and complexity influenced overall strength.

### Our Password Strength Algorithm
We define a heuristic approach to evaluate the strength of a given password. Our method primarily examines simple features such as the password’s length, the number of unique characters, and the use of different character types. Additionally, we use advanced techniques, such as regular expressions to detect common patterns and predefined lists to identify passwords containing popular words and numbers.

#### Implementation Steps
- **Step 1 - Initial Disqualification Tests**
  - When evaluating a password, we first run basic tests to quickly disqualify weak passwords, assigning them a score of 0. If the password meets any of these conditions, it is immediately given a score of 0, as such patterns are highly predictable and easily compromised. This initial screening involves checking five key conditions:
    1. Common Passwords: The password appears in a predefined list of common passwords, numbers, popular words or names. These lists were compiled through an analysis of various datasets, common male, female, and family names. This approach allows us to detect passwords that are superficially complex but are actually based on easily guessable patterns.
    2. Numeric Passwords: The password is shorter than 9 characters and consists only of numbers.
    3. Repeated Characters: The password consists of a single character repeated multiple times.
    4. Date Formats: The password matches a common date format (e.g., 24/07, 07-24, 2407, 24.07.1999, etc.).
    5. Email Format: The password matches the user's email address format (e.g., xxxx@yyy.zz), which is considered highly insecure.
- **Step 2 - Detecting Fake Complexity**
  - If the password passes the initial disqualification tests, we proceed to analyze its true complexity. We create two additional variations of the password to detect "fake complexity," where a password may seem strong at first glance but is actually composed of common patterns that are vulnerable to brute-force attacks.
    - In the first variation, we remove punctuation symbols and sequences of digits, transforming the password into a recognizable word or phrase (if it exists in our predefined word lists). 
    - In the second variation, we replace any detected word or number substring with a single character, based on the rationale that any distinct word serves as a single additional unit of complication, much like the complexity that is added by the addition of a single random character.
- **Step 3 - Final Strength Calculation**
  - After analyzing the password and its two variations, we compute a strength score for each version based on its features, such as length, character diversity, and detected patterns. We then combine these scores using a weighted sum and normalize the result to a scale of 0 to 10, where:
    - 0 represents a password that can be cracked within seconds using basic brute-force techniques.
    - 10 represents a highly secure password that does not exhibit any significant weaknesses. 

### Evaluation - Cracking Method
To assess the effectiveness of our analysis and recommendations, we developed a smart brute-force algorithm and tested it on the "Pwned" dataset, which contains over 500 million MD5/NTLM hashed passwords along with their frequency counts, collected from various breaches over the years. While cracking a single strong password can take years, targeting a significant portion of a large database is feasible and a common tactic in cyber threats. Our algorithm, leveraging intelligent brute-forcing techniques, carefully selected dictionaries, and statistical insights, was designed to crack as many passwords as possible in a reasonable timeframe.

### Key Concepts
- **Identifying Common Password Patterns**: One of the core aspects of our method involves analyzing large password datasets to detect frequently used patterns. These patterns typically follow predictable structures, such as lowercase letters followed by numbers, or the inclusion of special characters in specific positions. By extracting statistical insights from these datasets, we can identify these patterns and focus our cracking efforts on the most vulnerable hashes, maximizing efficiency with minimal computational effort.
- **Recognizing Common Words, Names, and Numbers**: An extensive analysis of password datasets reveals trends such as the overuse of popular names, keyboard combinations (e.g., "asdf," "qwerty"), and predictable numeric sequences. By compiling lists of these frequently used elements, we direct our efforts toward promising password candidates.
- **Additional Data Sources**: Our method goes beyond password datasets by incorporating data from unrelated fields. For example, we analyzed names and popular terms from social media, sports, and pop culture, which often provide insight into common password choices, and can be used for a specific attack vector.
- **Incorporating Patterns**: This involves inserting common elements into predictable structures that users typically follow. For instance, a popular pattern might be lowercase letters followed by digits, or a name combined with a number. These combinations are then crafted to match common password templates that appear in many datasets.
  - **Common Patterns**:
    - **Sequences of Digits**: One of the simplest and most predictable patterns involves sequences of digits, with combinations like "123456" appearing frequently.
    - **Sequences of Words**: Another common but slightly more complex pattern includes predictable sequences of words, such as "ILoveMyMom." Although this phrase might be endearing, it provides little security against dictionary cracking attempts.
    - **Concatenation of Lowercase Letters and Digits**: A more sophisticated pattern includes a word from a small pool of common words followed by a few digits or punctuation marks like "!" or ".".
    - **Word Separated by Special Characters**: In this pattern, a common word is followed by a special character (e.g., "_" or "@") and then a frequently used number.

### Future Work
In the future, our system could be enhanced by integrating real-time data from newly breached password sets, providing even more up-to-date feedback to users. Additionally, expanding the system to handle multi-language support would make it more applicable to a global user base. Further improvements could include integrating password manager functionalities to help users generate secure passwords directly through the dashboard.
