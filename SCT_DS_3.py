#!/usr/bin/env python
# coding: utf-8

# # Script: Decision Tree Classifier for Bank Marketing Dataset
# Purpose: Predict whether a customer will purchase a product or service based on demographic and behavioral data.
# Dataset: Bank Marketing Dataset (UCI Machine Learning Repository)

# In[49]:


import pandas as pd


# In[26]:


data = pd.read_csv(r"C:\Users\ramsi\Downloads\bank+marketing\bank\bank-full.csv",sep=";", quotechar='"')


# In[27]:


data


# # Understand the Data

# In[11]:


data.head()


# In[12]:


data.info


# In[16]:


data.describe()


# # Check missing values

# In[18]:


data.isnull().sum()


# # Encode Categorical Variables

# In[28]:


# Identify categorical columns
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# Perform one-hot encoding
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)



# In[29]:


#Encode the target variable
data['y'] = data['y'].map({'no': 0, 'yes': 1})


# In[30]:


data.head()


# # Split the Data

# In[32]:


from sklearn.model_selection import train_test_split
# Features and target
X = data.drop('y', axis=1)
y = data['y']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")



# # Decision Tree classifier
# 

# In[34]:


from sklearn.tree import DecisionTreeClassifier, plot_tree
# Initialize the Decision Tree Classifier
dt = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=42)

# Train the model
dt.fit(X_train, y_train)


# In[35]:


y_pred = dt.predict(X_test)


# # Measure accuracy and other metrics.

# In[36]:


from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[37]:


# Accuracy score
print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# # Visualize the decision tree

# In[38]:


plt.figure(figsize=(20, 10))
plot_tree(dt, feature_names=X.columns, class_names=['No', 'Yes'], filled=True, rounded=True)
plt.title("Decision Tree Visualization")
plt.show()


# # Hyperparameter Tuning

# In[40]:


from sklearn.model_selection import GridSearchCV

# Define parameter grid
param_grid = {
    'max_depth': [3, 5, 7, 10],
    'min_samples_split': [2, 5, 10],
    'criterion': ['gini', 'entropy']
}

# Perform grid search
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best parameters
print("Best Parameters:", grid_search.best_params_)

# Train the best model
best_dt = grid_search.best_estimator_


# # Prediction on new data

# In[46]:


# Example new customer data
new_customer_data = {
    'age': [35],
    'balance': [5000],
    'duration': [120],
    'campaign': [2],
    'pdays': [-1],
    'previous': [0],
    'job_technician': [1],
    'job_unknown': [0],
    'marital_married': [1],
    'marital_single': [0],
    'education_secondary': [1],
    'education_tertiary': [0],
    'default_yes': [0],
    'housing_yes': [1],
    'loan_yes': [0],
    'contact_cellular': [1],
    'contact_unknown': [0],
    'month_jun': [0],
    'month_may': [1],
    'month_nov': [0],
    'poutcome_success': [0]
}

# Convert to DataFrame
new_customer = pd.DataFrame(new_customer_data)

# Ensure all columns match
for col in X.columns:
    if col not in new_customer:
        new_customer[col] = 0  # Add missing columns with default value 0
new_customer = new_customer[X.columns]  # Reorder columns to match training data


# In[47]:


# Predict for the new customer
prediction = best_dt.predict(new_customer)
print("Prediction:", "Yes" if prediction[0] == 1 else "No")


# In[44]:





# In[ ]:





# In[ ]:




