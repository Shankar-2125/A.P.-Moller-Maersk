# A.P.-Moller-Maersk AIML Assessment

# Overview

Welcome to the A.P.-Moller-Maersk AIML Assessment Project! This repository contains code for predicting sourcing costs using various machine learning algorithms. In this project, we explore a dataset containing historical data on sourcing costs and build predictive models to forecast costs for future periods.

# Libraries Used
The project utilizes several Python libraries for data analysis, visualization, and machine learning:

Pandas: For data manipulation and analysis.

Seaborn and Matplotlib: For data visualization.

Plotly Express: For interactive visualizations.

Scikit-learn: For machine learning algorithms.

XGBoost: For gradient boosting algorithms.

# Analysis and Visualization

Descriptive statistics, histograms, count plots, scatter plots, box plots, and time series plots are used to explore and visualize the data.
Geospatial analysis is performed to visualize the distribution of sourcing costs based on area codes.

# Model Training

In this project, we employ various machine learning algorithms to predict sourcing costs. Here's a breakdown of the steps involved:

1. Model Selection:

   We consider several regression algorithms suitable for this task:

   Gradient Boosting Regressor:A powerful ensemble learning technique that builds models sequentially, each one correcting the errors of its predecessor.

   Random Forest Regressor: Utilizes a multitude of decision trees to reduce overfitting and improve accuracy.

   XGBoost Regressor: An optimized gradient boosting library known for its speed and performance.

   Decision Tree Regressor: A simple yet effective model that partitions the data recursively based on feature splits.

3. Training:

   Each model is trained using the training dataset, which contains historical data on sourcing costs. During training, the algorithm learns the patterns and relationships within the data to make accurate
   predictions.

4. Hyperparameter Tuning:

   Hyperparameters are parameters that are not directly learned by the model during training but rather set beforehand. We fine-tune these hyperparameters to optimize the performance of each model. Techniques 
   such as grid search or random search can be employed for this purpose.

6. Evaluation Metrics:

   To assess the performance of each model, we use the following evaluation metrics:

   Mean Absolute Error (MAE):
   Represents the average absolute difference between the predicted and actual values. It provides a measure of the average magnitude of errors in the predictions.

   Mean Squared Error (MSE):
   Calculates the average of the squares of the errors between the predicted and actual values. It penalizes larger errors more heavily than MAE, making it sensitive to outliers.

5. Cross-Validation:

   To ensure the robustness of our models and avoid overfitting, we perform cross-validation. This technique involves splitting the training data into multiple subsets, training the model on different 
   combinations of these subsets, and evaluating its performance on the remaining data.


7. Model Comparison:

   After training and evaluating each model, we compare their performance based on the evaluation metrics. This comparison helps us identify the best-performing algorithm for the task of sourcing cost prediction.

8. Final Model Selection:

   Based on the evaluation results, we select the most suitable model for forecasting sourcing costs. The selected model is then used to make predictions on the test dataset for future periods.

# Future Work
While the current models provide accurate predictions for sourcing costs, there are opportunities for further improvement. Future work may involve:
Feature engineering: Exploring additional features or transformations to enhance model performance.

Advanced algorithms: Experimenting with more sophisticated machine learning techniques, such as neural networks or ensemble methods.

Incorporating external data: Integrating external datasets, such as economic indicators or market trends, to capture additional factors influencing sourcing costs.

By continuously refining our models and techniques, we aim to develop robust and accurate forecasting models for sourcing cost prediction.

# Forecasting
Trained models are used to forecast sourcing costs for June 2021 using the test dataset.
