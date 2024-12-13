# SCT_DS_3
This is my third task in SkillCraft Technology Internship.
Decision Tree Classifier for Bank Marketing Dataset

This project uses a Decision Tree Classifier to predict whether a customer will purchase a product or service based on demographic and behavioral data. The dataset used is the Bank Marketing Dataset from the UCI Machine Learning Repository.

Objective

The primary goal is to analyze customer data, build a decision tree model, and predict the likelihood of a customer purchasing a product or service.

Dataset Overview

The dataset contains information about customers, including:

Demographics: Age, job, marital status, education.

Behavioral Data: Balance, housing loan status, duration of contact.

Target Variable: y (binary: yes = purchased, no = not purchased).

Key Features:

age: Age of the customer.

job: Type of job.

marital: Marital status.

education: Level of education.

balance: Account balance.

duration: Duration of the last contact in seconds.

campaign: Number of contacts performed during the campaign.

pdays: Number of days since the customer was last contacted.

previous: Number of contacts performed before this campaign.

Steps in the Script

1. Data Loading and Exploration

The dataset is loaded using pandas.

Basic information about the dataset is displayed, including column names, data types, and missing values.

2. Data Preprocessing

Categorical variables are encoded using one-hot encoding.

The target variable (y) is mapped to binary values (0 for no, 1 for yes).

The dataset is split into training and test sets (70% training, 30% test).

3. Model Training

A Decision Tree Classifier is trained using the training data.

The Gini index is used as the criterion for splitting.

The model is restricted to a maximum depth of 5 to prevent overfitting.

4. Model Evaluation

Accuracy, precision, recall, and F1-score are calculated.

A confusion matrix is generated.

5. Visualization

The decision tree is visualized to understand the decision-making process.

6. Hyperparameter Tuning

Grid search is used to find the best parameters (max_depth, min_samples_split, criterion).

7. Prediction for New Data

An example new customer is used to demonstrate the prediction process.

The input data is aligned with the training feature structure.
