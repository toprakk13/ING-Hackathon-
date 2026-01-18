# ING-Hackathon-
This project is a machine learning model that predicts the probability of customers leaving (churn) a bank/service provider by using their demographic information and financial transaction history. Customized metrics (Lift, Recall@k) are used to handle imbalanced datasets and to identify the correct target audience.

About the Project

Predicting customer churn in advance enables companies to develop proactive strategies. In this project:
	•	Customers’ transaction frequencies, transaction amounts, and service tenure are analyzed.
	•	Missing values are filled with median values, and categorical variables are processed.
	•	A Random Forest Classifier algorithm is used to train the model.
	•	Model performance is evaluated using Recall@10% and Lift@10% metrics, which are especially critical for marketing campaigns.

📂 File Structure
	•	ModelAI.py: The main Python script that performs data preprocessing, feature engineering, model training, and prediction.
	•	Metrics.py: Contains custom metric functions used to evaluate model performance (recall_at_k, lift_at_k, convert_auc_to_gini).
	•	customers.csv: Customer demographic data (age, gender, occupation, etc.).
	•	referance_data.csv: Reference dates and churn labels for training data.
	•	submission.csv: Prediction outputs of the model on the test data.
