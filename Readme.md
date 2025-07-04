# Employee Sentiment Analysis Project: A Deep Dive

**Date:** July 4, 2025

**Project Lead:** [Your Name/Team Name]

## 1. Executive Overview

In today's competitive business landscape, understanding employee morale is not just an HR metric; it's a critical driver of productivity, innovation, and retention. This project, **"Employee Sentiment Analysis,"** is a comprehensive initiative designed to move beyond traditional surveys and gain real-time, data-driven insights into the sentiment of our workforce.

By programmatically analyzing the text from internal email communications, we can identify sentiment patterns, track morale over time, and build predictive models to understand the key factors that influence employee happiness and engagement. This allows for proactive, rather than reactive, management and strategy.

The project is structured as a sequential pipeline of six data analysis tasks. We begin with raw data, clean and process it, perform sentiment analysis, visualize the findings, identify key trends and influencers, and culminate in a machine learning model that attempts to predict sentiment. This document serves as a detailed guide to the project's architecture, methodology, and technical implementation.

---

## 2. The Project's "Why": Goals and Objectives

The primary goals of this project are:

* **To Quantify Sentiment:** Move from anecdotal evidence to a quantitative understanding of employee sentiment by assigning a numerical score to email communications.
* **To Track Trends Over Time:** Analyze how sentiment evolves on a monthly basis, allowing us to correlate morale with company events, policy changes, or seasonal factors.
* **To Identify Key Influencers:** Pinpoint employees who consistently exhibit highly positive or negative sentiment, providing opportunities for recognition or support.
* **To Build Predictive Insights:** Use machine learning to discover the underlying drivers of sentiment. For example, does message length or frequency correlate with positive or negative feelings?
* **To Empower Decision-Making:** Provide leadership with a clear, evidence-based tool to monitor the pulse of the organization and make more informed decisions regarding employee welfare and engagement strategies.

---

## 3. The Workflow: A Six-Step Journey

The entire analysis is broken down into six modular Jupyter Notebooks. It is **critical** to run these notebooks in sequential order, as each task generates an output file that serves as the input for the next.



---

## 4. Detailed Task Descriptions

Here we break down each step of the analysis pipeline, explaining its purpose, processes, and outcomes.

### ➤ **Task 1: The Foundation - Text Processing and Sentiment Analysis (`task1.ipynb`)**

* **Purpose:** This is the foundational step. Its goal is to take the raw, unstructured email text and transform it into a structured dataset with a calculated sentiment score for each message.
* **Methodology:**
    1.  **Data Loading:** The process begins by loading the `enron-spam-master.csv` file, focusing on the sender, date, and body of each email.
    2.  **Text Preprocessing:** To ensure our analysis is accurate, we must "clean" the text. A custom function performs several key operations using the **Natural Language Toolkit (NLTK)** library:
        * **Tokenization:** Breaking down each email body into individual words or "tokens."
        * **Stopword Removal:** Removing common English words that provide little semantic value (e.g., "the," "a," "in," "is").
        * **Lemmatization:** Reducing words to their base or root form (e.g., "running" becomes "run," "better" becomes "good"). This helps in standardizing the vocabulary.
    3.  **Sentiment Scoring:** We use the powerful **VADER (Valence Aware Dictionary and sEntiment Reasoner)** tool. VADER is specifically attuned to sentiments expressed in social media and informal text, making it a great choice for emails. It analyzes the processed text and produces a `compound` score ranging from -1 (most negative) to +1 (most positive).
    4.  **Classification:** Based on the compound score, each email is classified into one of three simple categories: **'positive'**, **'negative'**, or **'neutral'**.
* **Input:** `enron-spam-master.csv`
* **Output:** A new file, `task1_result.csv`, which contains the original data enriched with the processed text, the raw sentiment score, and the final sentiment category.

### ➤ **Task 2: The Big Picture - Visualizing Sentiment Distribution (`task2.ipynb`)**

* **Purpose:** To get a quick, high-level understanding of the overall sentiment across all analyzed emails. Is the general tone of our company's communication positive, negative, or balanced?
* **Methodology:**
    1.  The notebook loads the results from the previous task.
    2.  Using the **Matplotlib** library, it generates a histogram of the raw sentiment scores. A histogram is a bar chart that shows the frequency of data points in a set of ranges.
* **Output:** A visual plot is displayed. This chart allows for an immediate visual assessment of the company's overall emotional pulse.

### ➤ **Task 3: Finding the Pattern - Monthly Sentiment Aggregation (`task3.ipynb`)**

* **Purpose:** To shift our focus from individual emails to individual employees and track their sentiment over time. This helps us understand employee journeys and identify trends.
* **Methodology:**
    1.  The notebook loads the sentiment data from Task 1.
    2.  It converts the 'Date' column into a proper datetime format to enable chronological analysis.
    3.  Using the **Pandas** library, it groups all emails first by the employee (the 'From' address) and then by the month.
    4.  It then calculates the *average* sentiment score for each employee within each month.
* **Output:** A new file, `task3_result.csv`, is created. This file provides a summarized view, showing the average monthly sentiment for every employee, making it easy to track sentiment evolution.

### ➤ **Task 4: Highlighting the People - Employee Ranking (`task4.ipynb`)**

* **Purpose:** To use the aggregated monthly data to identify and rank employees at the extremes of the sentiment spectrum. This can be used to recognize positive ambassadors or to identify individuals who may need support.
* **Methodology:**
    1.  The notebook loads the monthly aggregated data from Task 3.
    2.  For each month in the dataset, it sorts the employees by their average sentiment score.
    3.  It then selects the **top 3 employees with the most positive scores** and the **bottom 3 employees with the most negative scores**.
* **Output:** A new file, `task4_result.csv`, that presents a clear, month-by-month leaderboard of the most positive and negative communicators.

### ➤ **Task 5: Getting Ready for AI - Feature Engineering (Part 1) (`task5.ipynb`)**

* **Purpose:** To prepare our dataset for the final machine learning task. A predictive model needs more than just the text; it needs numerical features to find patterns. This notebook creates some basic but important features.
* **Methodology:**
    1.  The notebook returns to the detailed data from Task 1.
    2.  It calculates two new features for every single email:
        * `message_length`: The total number of characters in the email body.
        * `word_count`: The total number of words in the email body.
* **Output:** A new file, `task5_result.csv`, which is an enriched version of our core dataset, now including these new numerical features.

### ➤ **Task 6: The Final Step - Predictive Modeling (`task6.ipynb`)**

* **Purpose:** To build a machine learning model that attempts to *predict* an email's sentiment score based on its characteristics. This helps us answer the question: "What factors contribute to a positive or negative sentiment?"
* **Methodology:**
    1.  **Advanced Feature Engineering:** The notebook first creates several more sophisticated features for each employee, such as their overall message frequency and average message length. It also counts the number of explicitly positive and negative words in each message.
    2.  **Model Selection:** We use a **Linear Regression** model from the **Scikit-learn** library. This is a fundamental model that tries to find a linear relationship between the input features and the output (the sentiment score).
    3.  **Training and Testing:** The data is split into a training set (which the model learns from) and a testing set (which is used to evaluate its performance on unseen data).
    4.  **Evaluation:** The model's predictive power is measured using two key metrics:
        * **Mean Squared Error (MSE):** The average squared difference between the predicted scores and the actual scores. Lower is better.
        * **R-squared (R2):** A score that indicates how much of the variance in the sentiment score is explained by the model. It ranges from 0 to 1, with higher being better.
* **Output:** The notebook prints the final performance metrics (MSE and R2).
* **Finding:** The initial Linear Regression model shows underperformance, indicating it cannot effectively capture the complex patterns in the data. This is a valuable insight in itself, suggesting that more sophisticated, non-linear models should be explored in future iterations of this project.

---

## 5. Technical Stack

This project relies exclusively on the Python ecosystem and several industry-standard open-source libraries:

* **Python 3.x**
* **Jupyter Notebook:** For interactive development and analysis.
* **Pandas:** The cornerstone for data manipulation and analysis.
* **NLTK (Natural Language Toolkit):** The primary tool for text preprocessing.
* **Matplotlib:** For data visualization.
* **Scikit-learn:** For building and evaluating the machine learning model.

---

## 6. How to Run the Project

To replicate this analysis, please follow these steps:

1.  **Prerequisites:** Ensure you have Python 3 installed on your system.

2.  **Install Dependencies:** Install all the required libraries using pip.
    ```bash
    pip install jupyter pandas nltk matplotlib scikit-learn
    ```
3.  **Download NLTK Data:** The NLTK library requires some additional data packages for its functions to work. Run the following commands in a Python interpreter:
    ```python
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('vader_lexicon')
    ```
4.  **Place Data:** Ensure the initial `enron-spam-master.csv` file is in the same directory as the notebooks.
5.  **Execute Notebooks:** Launch Jupyter Notebook and run the notebooks in order, from `task1.ipynb` to `task6.ipynb`.