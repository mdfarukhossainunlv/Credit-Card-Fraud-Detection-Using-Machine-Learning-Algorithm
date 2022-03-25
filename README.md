# Credit-Card-Fraud-Detection-Using-Machine-Learning-Algorithm

# Problem Statement:
The Credit Card Fraud Detection Problem includes modeling past credit card transactions with the knowledge of the ones that turned out to be fraud. This model is then used to identify whether a new transaction is fraudulent or not. Our aim here is to detect 100% of the fraudulent transactions while minimizing the incorrect fraud classifications.
# Observations
- Very few transactions are actually fraudulent (less than 1%). The data set is highly skewed, consisting of 492 frauds in a total of 284,807 observations. This resulted in only 0.172% fraud cases. This skewed set is justified by the low number of fraudulent transactions.
- The dataset consists of numerical values from the 28 ‘Principal Component Analysis (PCA)’ transformed features, namely V1 to V28. Furthermore, there is no metadata about the original features provided, so pre-analysis or feature study could not be done.
- The ‘Time’ and ‘Amount’ features are not transformed data.
- There is no missing value in the dataset.
# Why does class imbalance affect model performance?
- In general, we want to maximize the recall while capping FPR (False Positive Rate), but you can classify a lot of charges wrong and still maintain a low FPR because you have a large number of true negatives.
- This is conducive to picking a relatively low threshold, which results in a high recall but extremely low precision.
# What is the catch?
- Training a model on a balanced dataset optimizes performance on validation data.
- However, the goal is to optimize performance on the imbalanced production dataset. You ultimately need to find a balance that works best in production.
- One solution to this problem is: Use all fraudulent transactions, but subsample non-fraudulent transactions as needed to hit our target rate.
# Business questions to brainstorm:
Since all features are anonymous, we will focus our analysis on non-anonymized features: Time, Amount
- How different is the amount of money used in different transaction classes?
- Do fraudulent transactions occur more often during certain frames?
# The only non-transformed variables to work with are:
- Time
- Amount
- Class (1: fraud, 0: not_fraud)

png

Notice how imbalanced is our original dataset! Most of the transactions are non-fraud. If we use this DataFrame as the base for our predictive models and analysis we might get a lot of errors and our algorithms will probably overfit since it will “assume” that most transactions are not a fraud. But we don’t want our model to assume, we want our model to detect patterns that give signs of fraud!

Determine the number of fraud and valid transactions in the entire dataset.
- Shape of Fraudulant transactions: (492, 31)
- Shape of Non-Fraudulant transactions: (284315, 31)
# Distributions:
By seeing the distributions we can have an idea of how skewed are these features, we can also see further distributions of the other features. There are techniques that can help the distributions be less skewed which will be implemented in this notebook in the future.

Doesn’t seem like the time of transaction really matters here as per the above observation. Now let us take a sample of the dataset for our modeling and prediction

PNG

PNG

# Correlation Matrices
Correlation matrices are the essence of understanding our data. We want to know if there are features that influence heavily in whether a specific transaction is a fraud. However, it is important that we use the correct DataFrame (subsample) in order for us to see which features have a high positive or negative correlation with regard to fraud transactions.
# Summary and Explanation:
- Negative Correlations: V17, V14, V12, and V10 are negatively correlated. Notice how the lower these values are, the more likely the end result will be a fraud transaction.
- Positive Correlations: V2, V4, V11, and V19 are positively correlated. Notice how the higher these values are, the more likely the end result will be a fraud transaction.
- BoxPlots: We will use boxplots to have a better understanding of the distribution of these features in fraudulent and non-fradulent transactions.

Note: We have to make sure we use the subsample in our correlation matrix or else our correlation matrix will be affected by the high imbalance between our classes. This occurs due to the high-class imbalance in the original DataFrame.

PNG

# Model Building

# Artificial Neural Network (ANNs)

2671/2671 [==============================] - 11s 4ms/step - loss: 0.0037 - fn: 29.0000 - fp: 12.0000 - tn: 85295.0000 - tp: 107.0000 - precision: 0.8992 - recall: 0.7868
[0.003686850192025304, 29.0, 12.0, 85295.0, 107.0, 0.8991596698760986, 0.7867646813392639]

PNG

PNG ANNS

# XGBoost

png xgboost

# Random Forest

png random forest

# CatBoost

png catboost

# LigthGBM

png ligthgbm

# Model Comparison

png

# Conclusions:
We learned how to develop our credit card fraud detection model using machine learning. We used a variety of ML algorithms including ANNs and Tree-based models. At the end of the training, out of 85443 validation transaction, XGBoost perform better than other models:
- Correctly identifying 111 of them as fraudulent
- Missing 9 fraudulent transactions
- At the cost of incorrectly flagging 25 legitimate transactions
