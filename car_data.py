
import streamlit as st
st.set_page_config(layout="wide")
st.title("🚗 Car Price Prediction App")
st.write("Welcome! Fill in the sidebar to predict the car's price.")

import os
os.chdir(os.path.dirname(__file__))

import numpy as np
import pandas as pd
import pickle
import plotly.express as px
from PIL import Image

# Page configuration
st.set_page_config(layout="wide")

# Load your pre-trained model
with open('linear_model.pkl', 'rb') as f:
    lm2 = pickle.load(f)

# Load feature importance from an Excel file
def load_feature_importance(file_path):
    return pd.read_excel(file_path)

# Load the feature importance DataFrame
final_fi = load_feature_importance("feature_importance.xlsx")  # Replace with your file path

# Sidebar setup
# image_sidebar = Image.open('Pic 1.png')  # Replace with your image file
# st.sidebar.image(image_sidebar, use_column_width=True)
# st.sidebar.header('Vehicle Features')

# Feature selection on sidebar
def get_user_input():
    horsepower = st.sidebar.number_input('Horsepower (No)', min_value=0, max_value=1000, step=1, value=300)
    torque = st.sidebar.number_input('Torque (No)', min_value=0, max_value=1500, step=1, value=400)

    make = st.sidebar.selectbox('Make', ['Aston Martin', 'Audi', 'BMW', 'Bentley', 'Ford', 'Mercedes-Benz', 'Nissan'])
    body_size = st.sidebar.selectbox('Body Size', ['Compact', 'Large', 'Midsize'])
    body_style = st.sidebar.selectbox('Body Style', [
        'Cargo Minivan', 'Cargo Van', 'Convertible', 'Convertible SUV', 'Coupe', 'Hatchback',
        'Passenger Minivan', 'Passenger Van', 'Pickup Truck', 'SUV', 'Sedan', 'Wagon'
    ])
    engine_aspiration = st.sidebar.selectbox('Engine Aspiration', [
        'Electric Motor', 'Naturally Aspirated', 'Supercharged', 'Turbocharged', 'Twin-Turbo', 'Twincharged'
    ])
    drivetrain = st.sidebar.selectbox('Drivetrain', ['4WD', 'AWD', 'FWD', 'RWD'])
    transmission = st.sidebar.selectbox('Transmission', ['automatic', 'manual'])

    user_data = {
        'Horsepower_No': horsepower,
        'Torque_No': torque,
        f'Make_{make}': 1,
        f'Body Size_{body_size}': 1,
        f'Body Style_{body_style}': 1,
        f'Engine Aspiration_{engine_aspiration}': 1,
        f'Drivetrain_{drivetrain}': 1,
        f'Transmission_{transmission}': 1,
    }
    return user_data

# Top banner
#image_banner = Image.open('Pic 2.png')  # Replace with your image file
#st.image(image_banner, use_column_width=True)

# Centered title
st.markdown("<h1 style='text-align: center;'>Vehicle Price Prediction App</h1>", unsafe_allow_html=True)

# Split layout into two columns
left_col, right_col = st.columns(2)

# Left column: Feature Importance Interactive Bar Chart
with left_col:
    st.header("Feature Importance")

    # Sort feature importance DataFrame by 'Feature Importance Score'
    final_fi_sorted = final_fi.sort_values(by='Feature Importance Score', ascending=True)

    # Create interactive bar chart with Plotly
    fig = px.bar(
        final_fi_sorted,
        x='Feature Importance Score',
        y='Variable',
        orientation='h',
        title="Feature Importance",
        labels={'Feature Importance Score': 'Importance', 'Variable': 'Feature'},
        text='Feature Importance Score',
        color_discrete_sequence=['#48a3b4']  # Custom bar color
    )
    fig.update_layout(
        xaxis_title="Feature Importance Score",
        yaxis_title="Variable",
        template="plotly_white",
        height=500
    )
    st.plotly_chart(fig, use_container_width=True)

# Right column: Prediction Interface
with right_col:
    st.header("Predict Vehicle Price")

    # User inputs from sidebar
    user_data = get_user_input()

    # Transform the input into the required format
    def prepare_input(data, feature_list):
        input_data = {feature: data.get(feature, 0) for feature in feature_list}
        return np.array([list(input_data.values())])

    # Feature list (same order as used during model training)
    features = [
        'Horsepower_No', 'Torque_No', 'Make_Aston Martin', 'Make_Audi', 'Make_BMW', 'Make_Bentley',
        'Make_Ford', 'Make_Mercedes-Benz', 'Make_Nissan', 'Body Size_Compact', 'Body Size_Large',
        'Body Size_Midsize', 'Body Style_Cargo Minivan', 'Body Style_Cargo Van',
        'Body Style_Convertible', 'Body Style_Convertible SUV', 'Body Style_Coupe',
        'Body Style_Hatchback', 'Body Style_Passenger Minivan', 'Body Style_Passenger Van',
        'Body Style_Pickup Truck', 'Body Style_SUV', 'Body Style_Sedan', 'Body Style_Wagon',
        'Engine Aspiration_Electric Motor', 'Engine Aspiration_Naturally Aspirated',
        'Engine Aspiration_Supercharged', 'Engine Aspiration_Turbocharged',
        'Engine Aspiration_Twin-Turbo', 'Engine Aspiration_Twincharged',
        'Drivetrain_4WD', 'Drivetrain_AWD', 'Drivetrain_FWD', 'Drivetrain_RWD',
       'Transmission_automatic', 'Transmission_manual'
    ]

    # Predict button
    if st.button("Predict"):
        input_array = prepare_input(user_data, features)
        prediction = lm2.predict(input_array)
        st.subheader("Predicted Price")
        st.write(f"${prediction[0]:,.2f}")

# streamlit run Regr_model_cars.py




# # -*- coding: utf-8 -*-
# """car_data.ipynb

# Automatically generated by Colab.

# Original file is located at
#     https://colab.research.google.com/drive/1FLtBLObEPmELG4_NPyXY98thyDlIt1m5

# 2. Problem Formulation

# *   We want to understand which variables affect the car prices.

# *   We want to be able to predict car prices.

# 3.  Loading the Raw Data
# """

# import os
# import pandas as pd
# import numpy as np
# import seaborn as sns
# from matplotlib import pyplot as plt
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.model_selection import train_test_split
# import pickle
# import plotly.express as px
# import streamlit as st

# # loading excel
# car_data = pd.read_csv("car_data.csv")

# # to see your directory: os.getcwd()
# pd.set_option('display.max_rows', None) # display all rows
# pd.set_option('display.max_columns', None) # display all columns

# car_data.head()

# """4. Data Pre-processing

# 4.1. Understanding our data
# """

# # Investigate all the elements whithin each Feature

# for column in car_data:
#     unique_vals = np.unique(car_data[column].fillna('0'))
#     nr_values = len(unique_vals)
#     if nr_values <= 12:
#         print('The number of values for feature {} :{} -- {}'.format(column, nr_values,unique_vals))
#     else:
#         print('The number of values for feature {} :{}'.format(column, nr_values))

# # Checking for null values
# car_data.isnull().sum()

# """4.2. Dealing with missing values"""

# # droping the columns that have lot's of nulls
# car_data = car_data.drop(['Invoice Price', 'Cylinders','Highway Fuel Economy'], axis=1)

# # dealing with the columns that have missing values
# car_data.head()

# # creating a new column just for the number
# car_data['Horsepower_No'] = car_data['Horsepower'].str[0:3].astype(float)

# # viewing the null values
# car_data[car_data['Horsepower_No'].isna()]

# # caclulating the mean for ford cars
# mean_horsepower = car_data['Horsepower_No'][car_data['Make'] == 'Ford'].mean()

# # filling in the null values with the mean
# car_data['Horsepower_No'] = car_data['Horsepower_No'].fillna(mean_horsepower)
# car_data['Horsepower'] = car_data['Horsepower'].fillna(mean_horsepower)


# car_data.isnull().sum()

# # creating a new column just for the number
# car_data['Torque_No'] = car_data['Torque'].str[0:3].astype(float)

# # viewing the null values
# car_data[car_data['Torque_No'].isna()]

# # caclulating the mean for all cars
# mean_torque = car_data['Torque_No'].mean()

# # filling in the null values with the mean
# car_data['Torque_No'] = car_data['Torque_No'].fillna(mean_torque)
# car_data['Torque'] = car_data['Torque'].fillna(mean_torque)

# car_data.isnull().sum()

# """4.3. Cleaning the data types"""

# car_data.dtypes

# # cleaning MSRP
# car_data['MSRP'] = car_data['MSRP'].str.replace('$','')
# car_data['MSRP'] = car_data['MSRP'].str.replace(',','').astype(float)

# car_data['Used/New Price'] = car_data['Used/New Price'].str.replace('$','')
# car_data['Used/New Price'] = car_data['Used/New Price'].str.replace(',','').astype(float)

# """4.4. Visualizing the data"""

# # Example 1 - Visualize the data using seaborn Pairplots
# g = sns.pairplot(car_data)

# # Example 2 - Visualising a Subset of our data - important features
# g = sns.pairplot(car_data[['MSRP', 'Horsepower_No', 'Torque_No', 'Engine Aspiration']], hue = 'Engine Aspiration', height = 5)

# # Example 3 - Visualising a Subset of our data - important features
# g = sns.pairplot(car_data[['MSRP', 'Horsepower_No', 'Torque_No', 'Make']], hue = 'Make', height = 5) #, kind="reg")

# # visualizing the categorical values

# categories = ['Make','Body Size','Body Style', 'Engine Aspiration', 'Drivetrain','Transmission']

# # Increases the size of sns plots
# sns.set(rc={'figure.figsize':(8,5)})

# for c in categories:

#     ax = sns.barplot(x=c, y="MSRP", data=car_data, errorbar=('ci', False)) #, hue = 'Model')
#     for container in ax.containers:
#         ax.bar_label(container)
#     plt.title(c)
#     plt.show()

# # Investigating the distribution of all fields, adding the mean

# n_variables = ['MSRP','Used/New Price','Horsepower_No','Torque_No']

# # Increases the size of sns plots
# sns.set(rc={'figure.figsize':(8,5)})

# for n in n_variables:
#     x = car_data[n].values
#     sns.displot(x, color = 'blue');

#     # Calculating the mean
#     mean = car_data[n].mean()

#     #ploting the mean
#     plt.axvline(mean, 0,1, color = 'red')
#     plt.title(n)
#     plt.show()

# # Investigating the distribution of all Numerical values

# # Increases the size of sns plots
# sns.set(rc={'figure.figsize':(8,5)})

# for c in n_variables:
#     x = car_data[c].values
#     ax = sns.boxplot(x, color = '#D1EC46')
#     print('The meadian is: ', car_data[c].median())
#     plt.title(c)
#     plt.show()

# # Investigating the distribution of MSRP by categorical variables - by data points

# import warnings
# warnings.simplefilter(action='ignore', category=FutureWarning)

# categories = ['Make','Body Size','Body Style', 'Engine Aspiration', 'Drivetrain','Transmission']

# # Increases the size of sns plots
# sns.set(rc={'figure.figsize':(8,5)})

# for c in categories:
#     ax = sns.boxplot(x=c, y="MSRP", data=car_data, color = '#D1EC46')
#     ax = sns.swarmplot(x=c, y="MSRP", data=car_data, color="red", s=2)
#     plt.title(c)
#     plt.show()

# """4.5. Preparing the final DF"""

# # dropping the columns we do not need
# new_car_data = car_data.drop(['index', 'Model','Year', 'Trim', 'Used/New Price', 'Horsepower', 'Torque'], axis=1)

# new_car_data.head()

# # dropping the columns we do not need
# new_car_data = car_data.drop(['index', 'Model','Year', 'Trim', 'Used/New Price', 'Horsepower', 'Torque'], axis=1)

# # Making categorical variables into numeric representation
# new_car_data = pd.get_dummies(new_car_data, columns = ['Make','Body Size','Body Style', 'Engine Aspiration', 'Drivetrain','Transmission'])

# new_car_data.head()

# """4.6. Correlation & Feature Importances"""

# ## Correlations with Heatmap

# # Increases the size of sns plots
# sns.set(rc={'figure.figsize':(15,10)})

# n_variables = ['MSRP','Horsepower_No','Torque_No']

# pc = new_car_data[n_variables].corr(method ='pearson')

# cols = n_variables

# ax = sns.heatmap(pc, annot=True,
#                  yticklabels=cols,
#                  xticklabels=cols,
#                  annot_kws={'size':10},
#                  cmap="Blues")

# """**Feature Importance**

# Steps of Running Feature Importance









# *   Split the data into X & y
# *   Run a Tree-based estimators (i.e. decision trees & random forests)

# *   Run Feature Importance
# *   We measure the importance of a feature by calculating the increase in the model’s prediction error after permuting the feature




# """

# # Split the data into X & y

# X = new_car_data.drop(['MSRP'], axis = 1).values
# X_columns = new_car_data.drop(['MSRP'], axis = 1)
# y = new_car_data['MSRP'].astype(int)

# print(X.shape)
# print(y.shape)

# # Run a Tree-based estimators (i.e. decision trees & random forests)

# dt = DecisionTreeClassifier(random_state=15, criterion  = 'entropy', max_depth = 10)
# dt.fit(X,y)

# # Calculating FI
# for i, column in enumerate(new_car_data.drop('MSRP', axis=1)):
#     print('Importance of feature {}:, {:.3f}'.format(column, dt.feature_importances_[i]))

#     fi = pd.DataFrame({'Variable': [column], 'Feature Importance Score': [dt.feature_importances_[i]]})

#     try:
#         final_fi = pd.concat([final_fi,fi], ignore_index = True)
#     except:
#         final_fi = fi


# # Ordering the data
# final_fi = final_fi.sort_values('Feature Importance Score', ascending = False).reset_index()
# final_fi

# """5. Splitting the Raw Data - Hold-out validation"""

# # Hold-out validation
# X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.80, test_size = 0.2, random_state=15)

# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)

# """6. Running Regression"""

# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score, explained_variance_score, mean_absolute_error, mean_squared_error
# from math import sqrt

# # Training the Regression
# lm = LinearRegression(fit_intercept = True)
# lm.fit(X_train, y_train)

# y_pred = lm.predict(X_train)
# all_df_predict = lm.predict(X)

# # Model Accuracy on training dataset

# print('The Accuracy  on the training dataset is: ', lm.score(X_train, y_train) )
# print('The Accuracy r2  on the training dataset prediction is: ',r2_score(y_train,y_pred) )

# print("")
# # Model Accuracy on testing dataset
# print('The Accuracy  on the testing dataset is: ', lm.score(X_test, y_test) )

# print("")
# # The Root Mean Squared Error (RMSE)
# print('The RMSE  on the training dataset is: ',sqrt(mean_squared_error(y_train,y_pred)))
# print('The RMSE  on the testing dataset is: ',sqrt(mean_squared_error(y_test,lm.predict(X_test))))

# print("")
# # The Mean Absolute Error (MAE)
# print('The MAE  on the training dataset is: ',mean_absolute_error(y_train,y_pred))
# print('The MAE  on the testing dataset is: ',mean_absolute_error(y_test,lm.predict(X_test)))


# print("")
# # Coefficients
# print('Coefficients: ', lm.coef_ )

# print("")
# # The Intercept
# print('Intercept: ', lm.intercept_)

# """7. Storing Our Model & Results"""

# # Storing the ML Model
# with open('linear_model.pkl', 'wb') as f:
#     pickle.dump(lm, f)

# print(os.getcwd())

# # Storing the Feature Importances
# final_fi['Feature Importance Score'] = final_fi['Feature Importance Score'].round(4)
# final_fi = final_fi.head(27)
# final_fi.to_excel("feature_importance.xlsx")

# # Adding the predicted values
# car_data['MSRP Predictions'] = all_df_predict

# # Expoprting all the data with predictions
# car_data.to_excel("data_with_pred.xlsx")

# """8. Streamlit App - Deployment

# 8.1. Gathering the inputs
# """

# X_columns.head()

# # Transform the input into the required format
# def prepare_input(data, feature_list):
#     input_data = {feature: data.get(feature, 0) for feature in feature_list}
#     return np.array([list(input_data.values())])

# # Feature list (same order as used during model training)
# features = [
#     'Horsepower_No', 'Torque_No', 'Make_Aston Martin', 'Make_Audi', 'Make_BMW', 'Make_Bentley',
#     'Make_Ford', 'Make_Mercedes-Benz', 'Make_Nissan', 'Body Size_Compact', 'Body Size_Large',
#     'Body Size_Midsize', 'Body Style_Cargo Minivan', 'Body Style_Cargo Van',
#     'Body Style_Convertible', 'Body Style_Convertible SUV', 'Body Style_Coupe',
#     'Body Style_Hatchback', 'Body Style_Passenger Minivan', 'Body Style_Passenger Van',
#     'Body Style_Pickup Truck', 'Body Style_SUV', 'Body Style_Sedan', 'Body Style_Wagon',
#     'Engine Aspiration_Electric Motor', 'Engine Aspiration_Naturally Aspirated',
#     'Engine Aspiration_Supercharged', 'Engine Aspiration_Turbocharged',
#     'Engine Aspiration_Twin-Turbo', 'Engine Aspiration_Twincharged',
#     'Drivetrain_4WD', 'Drivetrain_AWD', 'Drivetrain_FWD', 'Drivetrain_RWD',
#     'Transmission_automatic', 'Transmission_manual']

# # new_car_data.columns

# # array of inputs
# input_array = prepare_input(user_data, features)
# input_array

# """8.3. Making the Prediction"""

# # prediction
# prediction = lm.predict(input_array)
# prediction

# """8.4. Creating the Bar Graph"""

# # creating our Bar chart

# # Sort feature importance DataFrame by 'Feature Importance Score'
# final_fi_sorted = final_fi.sort_values(by='Feature Importance Score', ascending=True)

# # Create interactive bar chart with Plotly
# fig = px.bar(
#     final_fi_sorted,
#     x='Feature Importance Score',
#     y='Variable',
#     orientation='h',
#     title="Feature Importance",
#     labels={'Feature Importance Score': 'Importance', 'Variable': 'Feature'},
#     text='Feature Importance Score',
#     color_discrete_sequence=['#48a3b4']  # Custom bar color
# )
# fig.update_layout(
#     xaxis_title="Feature Importance Score",
#     yaxis_title="Variable",
#     template="plotly_white",
#     height=500
# )
# # st.plotly_chart(fig, use_container_width=True)

# fig.show()
