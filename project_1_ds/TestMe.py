# Imports
import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import mean_squared_error, r2_score
import pickle

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def preprocess_data(data):
    data = data.dropna() # Drop rows with missing values
    data = data.drop_duplicates() # Drop duplicate rows
    
    # Calculate the IQR
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1

    # get data only with iterations column value < 20
    data_outliers = data[data['Iterations'] < 20]

    # Get observations with above or qual to 1.5 x IQR rule 
    data_outliers = ((data_outliers < (Q1 - 1.5 * IQR)) | (data_outliers > (Q3 + 1.5 * IQR))) 

    # Print observations with above or qual to 1.5 x IQR rule 
    outliers_to_visualize = data_outliers[data_outliers.any(axis=1)]

    # Remove observations in the outliers_to_visualize from the data
    data_wout_outliers = data.drop(outliers_to_visualize.index, axis=0)
    return data_wout_outliers

def split_data(data):
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    return X,y

def load_model(filename):
    loaded_model = pickle.load(open(filename, "rb"))
    return loaded_model

def else_prediction(data, model):
    else_pred = model.predict(data)
    return else_pred    

def metric_evaluation(y_test, y_pred):
    # RMSE 
    print("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))
    # R2
    print("R2: ", r2_score(y_test, y_pred))

def load_scale_object():
    # Load the scaler object from the file
    with open("project_1\scaler.pkl", "rb") as file:
        scaler_loaded = pickle.load(file)
    return scaler_loaded

def main():

    # Load data
    data = load_data("project_1\Lab6-Proj1_Dataset.csv")

    # Preprocess data
    data = preprocess_data(data)

    # Split data
    X,y = split_data(data)

    # Load scaler object
    scaler = load_scale_object()
    X = scaler.transform(X)

    # Load model
    model = load_model("project_1\model.pickle")

    # ELSE prediction vector 
    else_pred = else_prediction(X, model)

    # Root mean squared deviation
    metric_evaluation(y, else_pred)

if __name__ == "__main__":
    main()