# Diabetes-prediction-project-
DiabeticPrediction
The Diabetes Prediction project aims to predict whether a patient is diabetic or not based on their health attributes such as insulin intake, blood pressure, age, glucose levels, skin thickness, BMI, diabetes pedigree function, and the number of pregnancies. This prediction is made using machine learning algorithms and the project utilizes libraries such as pandas, scikit-learn (sklearn), and seaborn.

Dataset The project utilizes a dataset downloaded from Kaggle, which contains the necessary information about patients for predicting their diabetic condition. The data is loaded into a pandas DataFrame, where preprocessing steps are applied to clean the data, remove outliers, perform normalization, and reduce dimensionality.

Functions The project includes the following functions:

isnull() and drop_duplicates(): These functions are used to check the presence of null values in the DataFrame and remove any duplicated columns, ensuring data quality.

describe(): This function provides a statistical overview of the dataset, including count, mean, standard deviation, minimum, maximum, median, and first and third quartiles of the data.

jointplot(): This function is used to visualize the correlation between two dimensions of the data, providing insights into the relationship between different health attributes.

remove_outliers(): The remove_outliers() function takes the DataFrame, target columns, and a threshold as inputs. It identifies and removes outliers from the dataset based on the specified threshold, ensuring the accuracy of the prediction model.

MinMaxScaler(): This function performs data normalization using the Min-Max scaling technique. It scales the data within a specific range, making it suitable for the machine learning model.

PCA (Principal Component Analysis): The pca() function is used to perform dimensionality reduction by selecting the most appropriate columns for the prediction model. It uses PCA to extract the most significant features from the dataset.

train_test_split(): This function splits the preprocessed dataset into training and testing datasets. The training dataset is used to train the prediction model, while the testing dataset is used to evaluate the model's accuracy.

classification_report(): This function generates a classification report that provides metrics such as precision, recall, F1-score, and support values. It assesses the performance of the machine learning model.

Machine Learning Models The project utilizes two machine learning models for binary classification: Support Vector Machine (SVM) and Logistic Regression.

Support Vector Machine (SVM): SVM is a powerful algorithm used for binary classification. It separates data points into two classes or constructs multiple hyperplanes for multi-class classification. SVM is well-suited for this project's binary classification task of predicting whether a patient is diabetic or not.

Logistic Regression: Logistic regression models the relationship between input features and the probability of an instance belonging to a specific class. It uses a logistic function to map real-valued numbers to a value between 0 and 1, representing the probability. Logistic regression is another suitable algorithm for binary classification.

Evaluation Metrics The evaluation metric used in this project is the classification report, which provides precision, recall, F1-score, and support values for the machine learning models. The SVM model achieved an accuracy of 77%, while the logistic regression model achieved an accuracy of 78%.

Usage To use this project, follow these steps:

Download the dataset from the provided link on Kaggle. Load the dataset into a pandas DataFrame. Apply data preprocessing steps, including checking for null values, removing duplicates, and handling outliers. Perform data normalization using the Min-Max scaling technique. Apply dimensionality reduction using PCA and select the most appropriate columns for the prediction model. Split the preprocessed dataset into training and testing datasets using the train_test_split() function. This step ensures that the model is trained on a portion of the data and evaluated on unseen data.

Train the machine learning models on the training dataset. Utilize the SVM algorithm and logistic regression algorithm to classify the data and predict the diabetic condition of the patients.

Evaluate the performance of the trained models using the testing dataset. Generate a classification report using the classification_report() function, which provides metrics such as precision, recall, F1-score, and support values.

Analyze the results and assess the accuracy of the models. The SVM model achieved an accuracy of 77%, and the logistic regression model achieved an accuracy of 78%.

Use the trained models to predict the diabetic condition of new, unseen patient data by providing their health attributes as input.

Dependencies The project requires the following libraries:

pandas: Used for data manipulation and analysis. sklearn (scikit-learn): Provides machine learning algorithms and tools for data preprocessing, model training, and evaluation. seaborn: Used for data visualization. Make sure to have these libraries installed in your Python environment before running the project.

File Structure The project file structure is as follows:

diabetes_prediction.ipynb: Jupyter Notebook containing the code implementation and explanation of the project. dataset.csv: The dataset downloaded from Kaggle. You can find the Jupyter Notebook file and the dataset file in the project repository.

Running the Project To run the project, follow these steps:

Ensure that you have the required libraries installed in your Python environment.

Download the dataset (dataset.csv) from the provided link on Kaggle and place it in the same directory as the Jupyter Notebook file.

Open the Jupyter Notebook (diabetes_prediction.ipynb) in Jupyter Notebook or any compatible environment.

Execute the cells in the notebook sequentially to run the code and observe the results.

Make sure to provide the necessary input data or modify the code as per your requirements.

Explore the code, visualizations, and classification reports to gain insights into the prediction process and the performance of the models.

Conclusion The Diabetes Prediction project aims to predict the diabetic condition of patients based on their health attributes. By utilizing machine learning algorithms such as SVM and logistic regression, along with preprocessing techniques and evaluation metrics, the project provides a reliable method for determining whether a patient is diabetic or not.

Feel free to explore the code, experiment with different algorithms or evaluation metrics, and further enhance the project to meet your specific needs.
