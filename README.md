
# 🕮 Student Exam Performance Predictor 🕮

Welcome to the "Student Exam Performance Predictor" repository! Here, you will find a comprehensive implementation of the Student Exam Performance dataset, using a web application developed with Flask and Render.

This project covers a wide range of functionalities, including exploratory data analysis (EDA), feature engineering, model selection, and an organized pipeline-based approach to data processing.



## 📃Table of contents

1. [Data](#-Data)
2. [Project Structure](#-Project-Structure)
3. [EDA](#-EDA) 
4. [Future work](#-Future-work)
## 🗃️ Project Structure
This explains the project structure and the role of each file or folder.
```
├── artifacts
│   ├── data.csv - The dataset is stored as a dataframe 
│   ├── model.pkl - The best model stored in pickle file
│   ├── preprocessor.pkl - Data transformation in pickle file
│   ├── test.csv - The test set 
│   └── train.csv -  The training set 
│ 
├── notebooks  
│   ├── data
│   │   └── StudentsPerformance.csv - The original dataset
│   │
│   └── EDA_Student_Performance.ipynb - EDA in Jupyter notebook
│
├── src  - This is the source folder which contains all the code.
│   ├── components
│   │   ├── data_ingestion.py - Reading the dataset and spliting into train and test sets
│   │   ├── data_transformation.py - the column transformation with pipelines for both numerical and categorical columns. Data encoding and normalization is done here.
│   │   └── model_trainer.py - choosing the best model &training the model
│   │
│   ├── pipeline 
│   │   └── predict_pipeline.py - implements the  preprocessor and model pickle files and predicts for new data.
│   │
│   ├── exception.py - for custom exception handling 
│   ├── logger.py - for creating logs 
│   └── utils.py - any common functions used across the project
│
├── static
│   └── style.css - styling the app pages
│
├── templates 
│   ├── home.html - home page of app
│   └── prediction.html - page for predicting new scores
│
├── app.py - the flask app file
│ 
├── requirements.txt - all the required packages
│ 
└── setup.py - builds the project as a package
```
##  📊 Data

[![09Darshita09-Data](https://img.shields.io/badge/09Darshita09_--_Data-111111?style=flat-square&logo=github&logoColor=white)](https://github.com/09Darshita09/Student_Performance/blob/main/notebooks/data/StudentsPerformance.csv) 

The Student Exam Performance dataset provides valuable insights into various factors that may affect students' performance in exams. It includes features such as student demographics, parental education, test preparation, and scores in different subjects.



## 📈 Exploratory Data Analysis (EDA)

[![09Darshita09-EDA](https://img.shields.io/badge/09Darshita09_--_EDA-111111?style=flat-square&logo=github&logoColor=white)](https://github.com/09Darshita09/Student_Performance/blob/main/notebooks/EDA_Student_Performance.ipynb)

The Juypter notebook mentioned above, gives valuable insights into student exam performance by delving into the dataset. The EDA process involves analyzing the dataset to gain a better understanding of its features and their relationships. 

The data is checked for any missing values or duplicates here. All numerical and categorical features are analysed. This step also includes informative visualizations, summary statistics, and data exploration techniques to uncover patterns, correlations, and potential outliers.

The observations of each unique values of every feature.
![alt text](images/Pieplot.png)

Feature-wise visualization is also done to analyse :

- the group-wise distribution (univariate analysis)
    ![alt text](images/Univariate.png)
- the impact of each feature on the students' perfmance (bivariate analysis)
    ![alt text](images/Bivariate.png)

Many other visual analysis are done and insights about each observation are given, such as:

The plot to check potential outliers:
![alt text](images/Outliers.png)
## 🚀 Future work
## 🔗 Reach me at:

[![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/darshita-pangam/) 

[![github](https://img.shields.io/badge/github-111111?style=for-the-badge&logo=github&logoColor=white)](https://github.com/09Darshita09) 

[![kaggle](https://img.shields.io/badge/kaggle-46d2ff?style=for-the-badge&logo=kaggle&logoColor=white)](https://www.kaggle.com/darshitapangam )

[![edgeimpulse](https://custom-icon-badges.demolab.com/badge/edge_impulse-007272?style=for-the-badge&logo=edge_impulse)](https://studio.edgeimpulse.com/studio/profile/projects)