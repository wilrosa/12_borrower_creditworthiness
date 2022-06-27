# 12_borrower_creditworthiness

This repo contains the results of the module 12 challenge. To view the file, open the "credit_risk_resampling" folder and then the "credit_risk_resampling.ipynb" file. 

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. To deal with this issue, I used various techniques to train and evaluate models with imbalanced classes. Specifically, I used a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers.

Using my knowledge of the imbalanced-learn library, I applied a logistic regression model to compare two versions of the dataset. First, using the original dataset. Second, resampling the data by using the `RandomOverSampler` module from the imbalanced-learn library.

For both cases, I counted the target classes, trained a logistic regression classifier, calculated the balanced accuracy score, generated a confusion matrix, and generated a classification report. Below you will also find a credit risk analysis report.

To accomplish these goals, I completed the following steps:

(1) Split the Data into Training and Testing Sets 
(2) Created a Logistic Regression Model with the Original Data
(3) Predicted a Logistic Regression Model with Resampled Training Data
(4) Wrote a Credit Risk Analysis Report

## Installation Guide

I needed to add the imbalanced-learn and PyDotPlus libraries to my dev virtual environment.  The imbalanced-learn library has models that were developed specifically to deal with class imbalance.

```python
!conda install -c conda-forge imbalanced-learn
```
---

## Technologies

This project leverages python 3.7 with the following libraries and dependencies:

* [pandas](https://github.com/pandas-dev/pandas) - For manipulating data

* [numpy](https://github.com/numpy/numpy) - Fundamental package for scientific computing with Python

* [sklearn](https://github.com/scikit-learn/scikit-learn) - Module for machine learning built on top of SciPy

* [imblearn](https://github.com/scikit-learn-contrib/imbalanced-learn) - Package offering a number of re-sampling techniques

---

### **Step 1: Split the Data into Training and Testing Sets**

(1) Read the `lending_data.csv` data from the `Resources` folder into a Pandas DataFrame.

(2) Created the labels set `(y)` from the “loan_status” column, and then create the features `(X)` DataFrame from the remaining columns. In the “loan_status” column, a value of `0` means that the loan is healthy. A value of `1` means that the loan has a high risk of defaulting.

(3) Checked the balance of the labels variable `(y)` by using the `value_counts` function.

(4) Split the data into training and testing datasets by using `train_test_split`.

### **Step 2: Create a Logistic Regression Model with the Original Data**

Employed knowledge of logistic regression to complete the following steps:

(1) Fit a logistic regression model by using the training data (`X_train` and `y_train`).

(2) Saved the predictions on the testing data labels by using the testing feature data (`X_test`) and the fitted model. 

(3) Evaluated the model’s performance by doing the following: 

    - Calculated the accuracy score of the model.

    - Generated a confusion matrix.

    - Printed the classification report.
    
(4) Answered the following question: 
    
    - How well does the logistic regression model predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

### **Step 3: Predict a Logistic Regression Model with Resampled Training Data**

Because of the small number of high-risk loan labels, I considered that a model that uses resampled data may perform better. Thus, I resampled the training data and then reevaluated the model. Specifically, I used `RandomOverSampler`.

To do so, I completed the following steps:

(1) Used the `RandomOverSampler` module from the imbalanced-learn library to resample the data. Confirmed that the labels had an equal number of data points.

(2) Used the `LogisticRegression` classifier and the resampled data to fit the model and make predictions.

(3) Evaluated the model’s performance by doing the following:

    - Calculated the accuracy score of the model.

    - Generated a confusion matrix.

    - Printed the classification report.

(4) Answered the following question: 
      
    - How well does the logistic regression model, fit with oversampled data, predict both the 0 (healthy loan) and 1 (high-risk loan) labels?

### **Step 4: Credit Risk Analysis Report**

Here, I wrote a brief report that includes a summary and an analysis of the performance of both machine learning models that I used. The report contains the following:

(1) An Overview of the Analysis 

   (A) Purpose of the analysis.
   
Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. Thus, I used various techniques to train and evaluate models with imbalanced classes. 
   
   (B) What financial information the data was on, and what you needed to predict.
   
I used a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers, seperating borrowers into two classes, those with a "healthy" loan profile and those with a "high-risk" loan profile. Based on the financial infromation available in the dataset, such as loan size, borrower income, debt to income, number of accounts, interst rate, and total debt, I needed to predict which loans were healthy and which loans were high risk. 

   (C) Information about the variables you were trying to predict (e.g., `value_counts`).
   
Based on the model, as expected, there was a significant imbalance in the variables I tried to predict. Specifically, from the orginional dataset, the balance of "healthy loans" was 75,036 vs. 2,500 "high-risk loans". 

   (D) Stages of the machine learning process you went through as part of this analysis.
   
As part of the analysis, I had to go through the three stages of the machine learning process. First, I split the data into training and testing sets. Second, I created a logistics regression model and fit the data to the model. Finally, I made predictions with the origional data and the resampled data. 

   (E) Methods (e.g., `LogisticRegression`, or any resampling method).
   
I fit a Logistics Regression model by using the training data. Then I saved the predictions on the testing data labels by using the testing feature data and fitted the model. Moreover, I used the RandomOverSampler module from the imbalanced-learn library to resample the data. 

(2) Results - Balanced accuracy scores and the precision and recall scores of all machine learning models.

   (A) Machine Learning Model 1:
   
    - Balanced accuracy score = .952
    - Precision = .99
    - Recall = .99
    
   (B) Machine Learning Model 2:
    
    - Balanced accuracy score = .993
    - Precision = .99
    - Recall = .99
    
(3) Summary

   (A) Which one seems to perform best? How do you know it performs best?
   
Both models perform well. However, Machine Learning Model 2 performs better, as it has a slightly higher balanced accuracy score at .993 vs. .99.

   (B) Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )
   
Yes, the goal is to predict which loans are high risk. While both models have comprable precision ratings when classifying `1` "high-risk" loans (.85 vs. .84), Model 2 has a higher recall score when classifying "high-risk" loans, .99 vs. .91.    

   (C) Recommendation and reasoning.

I recommend using Machine Learning Model 2 for the reasons mentioned above. It has a slightly higher balanced accuracy score and a hgiher recall score when classifying "high-risk" loans. 
   
---
## Contributors

Brought to you by Wilson Rosa. https://www.linkedin.com/in/wilson-rosa-angeles/.

---
## License

MIT
