# predicting-breast-cancer-logistic-regression
 
## Introduction

Welcome to my first kernel on Kaggle. In this notebook, I explore the Breast Cancer dataset and develop a Logistic Regression model to try classifying suspected cells to Benign or Malignant. This notebook was inspired by Mehgan Risdal's kernel on the Titanic data, and Pedro Marcelino's kernel on the Housing Prices data.

The contents of this notebook will follow the outline below:

- [The Data - Exploratory Data Analysis](#the-data-exploratory-data-analysis)
- [The Variables - Feature Selection](#the-variables-feature-selection)
- [The Model - Building a Logistic Regression Model](#the-model-building-a-logistic-regression-model)
- [The Prediction - Making Predictions with the Model](#the-prediction-making-predictions-with-the-model)

Throughout the notebook, I will try to aid your understanding with some visualizations where necessary. I hope you enjoy reading through this notebook, and please leave comments below if you have any questions or feedbacks. I am a total beginner to the field of Data Science, so any feedback is welcome since it helps me realize my mistakes and also allows me to pick up new insights.

## The Data - Exploratory Data Analysis

Extracted from the UCI ML repository.

Attribute Information:
- id
- diagnosis: M = malignant, B = benign
- Columns 3 to 32

Ten real-valued features are computed for each cell nucleus:
- radius: distances from center to points on the perimeter
- texture: standard deviation of gray-scale values
- perimeter
- area
- smoothness: local variation in radius lengths
- compactness: perimeter^2 / area - 1.0
- concavity: severity of concave portions of the contour
- concave points: number of concave portions of the contour
- symmetry
- fractal dimension: "coastline approximation" - 1

The mean, standard error, and "worst" or largest (mean of the three largest values) of these features were computed for each image, resulting in 30 features. For instance, field 3 is Mean Radius, field 13 is Radius SE, field 23 is Worst Radius.

## Importing Libraries

```python
# import dependencies
# data cleaning and manipulation 
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# machine learning
from sklearn.preprocessing import StandardScaler

import sklearn.linear_model as skl_lm
from sklearn import preprocessing
from sklearn import neighbors
from sklearn.metrics import confusion_matrix, classification_report, precision_score
from sklearn.model_selection import train_test_split

import statsmodels.api as sm
import statsmodels.formula.api as smf
```

## Importing Dataset

```python
# read in the data and check the first 5 rows
df = pd.read_csv('../input/data.csv', index_col=0)
df.head()
```

## Exploratory Data Analysis (EDA)

The exploratory data analysis (EDA) section, including data visualization and an analysis of the data, can be added here.

## Feature Engineering

The feature selection and preprocessing section can be added here. You can explain the process of selecting relevant features and preprocessing the data.

## Model Development and Evaluation

The model development section and evaluation metrics can be included here. You can explain the choice of the logistic regression model, how the model is built, and the evaluation of the model's performance.

## Conclusion

Summarize the key findings and results of the project, and mention potential areas for future improvements or additional analysis.
 
