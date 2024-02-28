# Diabetes Prediction using Support Vector Machine (SVM)

## Overview
This project utilizes a Support Vector Machine (SVM) model to predict the onset of diabetes in patients based on various input features such as Pregnancies, Glucose level, Blood Pressure, Skin Thickness, Insulin level, BMI (Body Mass Index), Diabetes Pedigree Function, and Age. The dataset used for training and testing the model is the PIMA Diabetes Dataset obtained from Kaggle.

## Project Structure
The project is organized as follows:
- `diabetes_prediction.ipynb`: Jupyter Notebook containing the code for data preprocessing, model training, evaluation, and prediction.
- `data/`: Directory containing the dataset file.
- `README.md`: This README file providing an overview of the project.

## Dataset
The PIMA Diabetes Dataset consists of several features related to health metrics and lifestyle factors of female patients, along with an outcome variable indicating whether the patient developed diabetes within 5 years of data collection.

Features:
- Pregnancies: Number of times pregnant
- Glucose: Plasma glucose concentration a 2 hours in an oral glucose tolerance test
- BloodPressure: Diastolic blood pressure (mm Hg)
- SkinThickness: Triceps skin fold thickness (mm)
- Insulin: 2-Hour serum insulin (mu U/ml)
- BMI: Body mass index (weight in kg/(height in m)^2)
- DiabetesPedigreeFunction: Diabetes pedigree function
- Age: Age in years

Outcome:
- 0: Non-diabetic
- 1: Diabetic

## Methodology
Data Preprocessing: The dataset is preprocessed to handle missing values, outliers, and feature scaling to ensure optimal performance of the SVM model.

Model Training: The SVM model is trained using the scikit-learn library in Python. After loading and preprocessing the dataset, the data is split into training and testing sets. The SVM model is then trained on the training data and evaluated using various performance metrics such as accuracy, precision, recall, and F1-score.
The SVM model is trained on the preprocessed dataset to learn the patterns and relationships between the input features and the target variable.

Model Evaluation: The trained model's performance is evaluated using appropriate metrics such as accuracy, precision, recall, and F1-score to assess its predictive ability.



## Usage
To use this project, follow these steps:

Clone the Repository: git clone https://github.com/shivamj2001/diabetes-predictor.git
Install Dependencies: Ensure you have Python installed, and then install the required libraries listed in requirements.txt using pip install -r requirements.txt.
Run the Notebook: Open the Jupyter notebook diabetes_prediction.ipynb and follow the instructions to preprocess the data, train the SVM model, and evaluate its performance.


## Results
The SVM model achieves a 78% accuracy score in predicting diabetes, indicating its effectiveness in distinguishing between diabetic and non-diabetic patients.

## Future Improvements
- Explore feature engineering techniques to enhance the model's predictive power.
- Experiment with different machine learning algorithms to compare performance and identify the most suitable approach.
- Incorporate additional data sources or features for a more comprehensive analysis.
- Deploy the model as a web application for broader accessibility.
- Contributing :
    Contributions to this project are welcome! If you have suggestions for improvements, new features, or bug fixes, please 
     submit a pull request. For major changes, please open an issue first to discuss the proposed changes.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements
I acknowledge Kaggle for providing the PIMA Diabetes Dataset, which serves as the foundation for this project. 
Scikit-learn, Pandas, NumPy, Matplotlib, and Seaborn for their invaluable contributions to this project.
Additionally, thank the open-source community for their valuable contributions to the tools and libraries used in this project.
