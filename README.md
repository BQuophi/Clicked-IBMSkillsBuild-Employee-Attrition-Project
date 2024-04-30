# Clicked-IBMSkillsBuild-Employee-Attrition-Project

### Machine Learning for Predictive Analytics: Solving Employee Attrition at Bain and Compan

You've been hired as a data analyst on a team working for Bain and Company, a prestigious management consulting firm. One of Bain's high-profile clients is facing an unprecedented amount of employee attrition, or churn, and they want to leverage their company's HR data to gain insights into this problem. Your task is to work closely with Kim Wexler, the Head of HR at the client's organization, to analyze the provided HR dataset, predict employee attrition, and provide actionable recommendations for improving employee retention
![image](https://github.com/BQuophi/Clicked-IBMSkillsBuild-Employee-Attrition-Project/assets/92530942/952db675-11cb-4c7f-84d4-34cb98501f90)

### Understanding the Challenge

Employee attrition is a critical issue that can have far-reaching consequences for any organization. High turnover rates can lead to decreased productivity, low team morale, and increased costs associated with hiring and training new employees. Bain's client is grappling with this very problem, and they are seeking a data-driven approach to address it.
![image](https://github.com/BQuophi/Clicked-IBMSkillsBuild-Employee-Attrition-Project/assets/92530942/cb2db0d9-6c2f-43bd-802e-1a7084fb662b)

The goal of this project is twofold: 
- first, to identify the factors that contribute to employee attrition, and
- second, to develop a predictive model that can accurately forecast which employees are at risk of leaving the company.
Armed with these insights, the client can take proactive measures to improve employee satisfaction, implement targeted retention strategies, and ultimately reduce costly attrition rates.

Exploring the HR Dataset
The first step in this project is to thoroughly understand the HR dataset provided by Kim Wexler. This dataset contains various features that could potentially influence an employee's decision to stay or leave the company. Let's delve into the specifics:
Features
The dataset includes a comprehensive set of features that capture various aspects of an employee's profile, job characteristics, performance, and satisfaction levels. Here are the key features:

![image](https://github.com/BQuophi/Clicked-IBMSkillsBuild-Employee-Attrition-Project/assets/92530942/2793f4ab-1aaf-4a4a-9993-464666788d72)

One of the first things to do as a analyst is to critically take some time to look at the data to understand its features. This step is crucial because it helps to identify inconsistencies and missing value scenarios. This helps analysts to ask relevant questions during subsequent meetings with the client to understand the data and requirements properly. 

As shown below, critically examining the data helped to identify the case of missing values

![missing-val-img](https://github.com/BQuophi/Clicked-IBMSkillsBuild-Employee-Attrition-Project/assets/92530942/db147317-a4c8-46a3-985e-6dda5a94f692)

### Target Variable

The target variable in this dataset is Attrition, which is a binary indicator (Yes or No) representing whether an employee has left the company or not. This variable is the outcome we aim to predict using the available features.

### Descriptive Statistics

Before diving into any analysis, it's crucial to understand the characteristics of the dataset. We'll calculate descriptive statistics such as mean, median, standard deviation, and quartiles for numerical features, and frequency distributions for categorical features. These insights will help us identify any anomalies, outliers, or skewed distributions that may require further attention during the data preprocessing stage.

![image](https://github.com/BQuophi/Clicked-IBMSkillsBuild-Employee-Attrition-Project/assets/92530942/ead92c4e-c4d0-48a6-af2f-f1e4f6b84614)


### Data Cleaning and Preprocessing
Real-world datasets often contain imperfections that need to be addressed before any meaningful analysis can be performed. This step is crucial as it ensures that our models are trained on high-quality, reliable data, leading to more accurate and robust predictions.

### Handling Missing Values
Missing data is a common issue in datasets, and it can arise due to various reasons, such as data entry errors, incomplete surveys, or system failures. We'll explore different imputation techniques to fill in these missing values, such as mean/median imputation, k-nearest neighbors imputation, or more advanced methods like multiple imputation by chained equations (MICE).
The choice of imputation technique will depend on the nature and extent of missing data, as well as the underlying assumptions and distributions of the features. For example, if the missing data is scattered randomly across the dataset, mean/median imputation may be a suitable approach. However, if the missing data exhibits patterns or is concentrated in specific subgroups, more sophisticated methods like MICE may be required to preserve the relationships between features.

- **In this project analysis, the missing values were minimal and thus were dropped**.
- 
##  Feature Engineering
While the existing features in the HR dataset provide valuable insights, we modified and dropped certain features to better capture the underlying patterns and relationships within the data. Feature engineering is a crucial step in enhancing the predictive power of our models.

### Encoding Categorical Features
Several features in the HR dataset are categorical, such as Department, EducationField, JobRole, and MaritalStatus. To incorporate these features into our machine learning models, we'll need to encode them as numerical values. Common encoding techniques include:

- One-Hot Encoding: This method creates a new binary column for each unique category in the feature, effectively transforming a single categorical variable into multiple binary features.
- Label Encoding: This technique assigns a unique numerical label to each category in the feature, allowing the model to treat the categories as ordered values.
- Target Encoding: Instead of encoding categories based on their labels, target encoding uses the target variable (in this case, Attrition) to encode the categories based on their likelihood of belonging to the positive or negative class.

The choice of encoding technique will depend on the nature of the categorical features, the assumptions of the machine learning models, and the potential impact of the encoding on the model's performance and interpretability. For example, one-hot encoding can be useful when the categorical feature has a large number of unique categories and there is no inherent ordering among them. However, it can also lead to a high-dimensional feature space, which may increase the computational complexity and risk of overfitting.

Label encoding, on the other hand, can be more efficient in terms of memory usage, but it assumes an ordinal relationship between the categories, which may not always be appropriate. Target encoding can be a useful alternative when the relationship between the categorical feature and the target variable is more important than the relationship between the categories themselves.

### Feature Selection 
Certain feature like 'Age' were dropped
Looking at the correlation of our variables, we can see that we have a 77% correlation between the PercentSalaryHike and PerformanceRating variable, so let’s eliminate the PercentSalryHike variable. There are others so remove one of the highly correlated values each.

Reasons to Remove:
- Multicollinearity: Highly correlated independent variables can lead to multicollinearity, which can cause problems in statistical models like regression. It can make it difficult to determine the independent effect of each variable on the target variable.
  
## Model Selection & Building
With our cleaned and engineered dataset in hand, the next step is to select the appropriate machine learning models to tackle the employee attrition prediction problem. Since our target variable (Attrition) is binary, we'll focus on classification models. Two popular choices for this task are Decision Trees and Logistic Regression.

Modeling Techniques
1. Decision Trees
   
Strengths:
- Captures complex patterns.
- Easy to interpret.
- Handles missing values.
  
Weaknesses:
- Prone to overfitting.
- Instability.
Bias towards dominant classes.

2. Logistic Regression
Strengths:
- Simple and interpretable.
- Well-suited for binary classification.
- Regularization options.
  
Weaknesses:
- Assumes linear relationship.
- Limited expressiveness.
- Sensitive to outliers.

### Evaluation Metrics
To evaluate the performance of our classification models, we'll utilize a suite of metrics suitable for binary classification problems. These metrics include:

- Accuracy: The proportion of correctly classified instances (both true positives and true negatives).
- Precision: The fraction of true positives among the instances classified as positive (i.e., the model's ability to avoid false positives).
- Recall (Sensitivity): The fraction of true positives among the actual positive instances (i.e., the model's ability to detect all positive cases).
Let’s analyze the performance of both the Decision Tree and Logistic Regression models based on the provided metrics:

![results](https://github.com/BQuophi/Clicked-IBMSkillsBuild-Employee-Attrition-Project/assets/92530942/6739e3d4-fc3b-48de-a07b-63aef01a94b6)


The Decision Tree model has higher recall for attrition (captures more actual cases) but sacrifices precision (more false positives).
The Logistic Regression model has balanced precision and recall but misses actual attrition cases.
Considering the importance of identifying attrition, the Decision Tree model may be preferred with hyperparameter tuning to avoid overfitting.

![recommend](https://github.com/BQuophi/Clicked-IBMSkillsBuild-Employee-Attrition-Project/assets/92530942/6ca2ce43-09fa-44a7-94f7-afca4eca16c0)

![Screenshot (1997)](https://github.com/BQuophi/Clicked-IBMSkillsBuild-Employee-Attrition-Project/assets/92530942/410f24e5-dc1f-4122-a59c-668a8dca9fa0)

## Recommendations
Detailed recommendations for addressing employee attrition based on the specific factors:

### Total Working Years:
Employees with more years of experience tend to stay longer with the company.

- Recommendations:
Career Development: Invest in continuous learning and development opportunities for employees.

Mentorship Programs: Pair experienced employees with newer ones to foster growth.

Recognition: Acknowledge and celebrate long-serving employees to boost morale

### Job Satisfaction:
Satisfied employees are less likely to churn.

- Recommendations:
Job Design: Align job roles with employees’ strengths and interests.

Recognition and Rewards: Recognize achievements and provide fair rewards.

Career Growth: Offer clear career paths and opportunities for advancement
