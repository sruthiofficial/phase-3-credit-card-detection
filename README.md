# AI-Powered Credit Card Fraud Detection System

## Overview

This project develops an AI-powered system to accurately detect and prevent fraudulent credit card transactions in real time. By analyzing user behavior and transaction patterns using machine learning algorithms, the system aims to overcome the limitations of traditional rule-based fraud detection methods, which are often slow, inaccurate, and prone to false alerts.

## Problem Statement

Credit card fraud is a significant and growing threat in the digital transaction landscape, leading to substantial financial losses for both financial institutions and consumers. Existing fraud detection systems often struggle to keep pace with the increasingly sophisticated and adaptive nature of fraudulent activities. This project addresses this challenge by leveraging the power of Artificial Intelligence to build a more robust and efficient fraud detection system.

## Abstract

The rapid expansion of digital payment methods has made credit card fraud a critical concern. Conventional fraud detection systems, relying on static rules and manual reviews, are proving inadequate against evolving fraud techniques. This project proposes an AI-driven solution that employs machine learning algorithms to analyze transaction data, identify anomalous behavior, and detect potential fraud in real time. By continuously learning from historical data and adapting its detection model, the system aims to significantly reduce false positives and improve overall fraud detection accuracy.

## Project Structure

The project repository includes the following key components (based on the limited code snippets provided):

* **Data Loading and Preprocessing:** Code for loading transaction data (likely from a CSV file named `Churn_Modelling.csv`).
* **Data Cleaning:** Implementation of data cleaning steps, including dropping irrelevant columns such as `RowNumber`, `CustomerId`, and `Surname`.
* **Exploratory Data Analysis (EDA):** Code to check for missing values and analyze the distribution of the target variable (e.g., 'Exited' which might represent fraudulent vs. non-fraudulent transactions in this context, though the filename is suggestive of churn). Visualizations, such as a countplot of the churn distribution, are generated and saved as `churn_distribution.png`.
* **Feature Engineering:** Application of one-hot encoding to categorical features like 'Geography' and 'Gender'.
* **Model Development:** Implementation of a machine learning model for fraud detection. The code snippet suggests the use of `RandomForestClassifier` from the `sklearn.ensemble` library.
* **Model Evaluation:** Code for evaluating the performance of the trained model using metrics like `classification_report` and `confusion_matrix` from `sklearn.metrics`.
* **Model Persistence:** Implementation to save the trained model using `joblib`.

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn (sklearn)
* Matplotlib
* Seaborn (implied by `plt.figure` and `sns.countplot`)
* Plotly Express

## Setup and Installation

1.  Clone the GitHub repository:
    ```bash
    git clone [https://github.com/sruthiofficial/phase-3-credit-card-detection.git](https://github.com/sruthiofficial/phase-3-credit-card-detection.git)
    cd phase-3-credit-card-detection
    ```
2.  Install the required Python libraries:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn plotly joblib
    ```
3.  Ensure the dataset (`Churn_Modelling.csv`) is present in the project directory.

## Usage

1.  Run the Python scripts in the repository to perform data preprocessing, model training, and evaluation. The specific execution order might be indicated by the filenames or the overall project workflow.

## Future Scope

The project can be further enhanced in the following areas:

1.  **Real-time deep learning for dynamic fraud detection:** Implementing deep learning models capable of adapting to evolving fraud patterns in real time.
2.  **Federated learning for privacy-preserving training:** Enabling collaborative model training across multiple data sources without sharing sensitive information.
3.  **Adaptive models that learn new fraud patterns continuously:** Developing models that can automatically detect and learn from new types of fraudulent activities.
4.  **Behavioral biometrics for enhanced user verification:** Incorporating user behavior data (e.g., typing speed, mouse movements) for more accurate authentication and fraud detection.
5.  **Graph analytics to detect fraud networks:** Utilizing graph-based techniques to identify interconnected fraudulent activities and accounts.
6.  **Explainable AI to build user trust:** Implementing methods to provide insights into the model's decision-making process, enhancing transparency and user trust.
7.  **Blockchain for secure, traceable transactions:** Exploring the use of blockchain technology to create a more secure and transparent transaction environment.

## Team Members and Roles

* Data cleaning - Trisha .M
* EDA - Varuneesri.A
* Feature engineering - Pavithra.S.Y
* Model development - Sruthi.L
* Documentation and reporting - Swetha .J and Priyanka .P

## Repository Link

[https://github.com/sruthiofficial/phase-3-credit-card-detection.git](https://github.com/sruthiofficial/phase-3-credit-card-detection.git)

## Date of Submission

09/05/2025
