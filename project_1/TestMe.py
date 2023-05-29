import numpy as np
import pandas as pd
import sklearn as sk
from sklearn.metrics import mean_squared_error
import pickle

def load_data(filename):
    data = pd.read_csv(filename)
    return data

def preprocess_data(data):
    data = data.dropna() # Drop rows with missing values
    data = data.drop_duplicates() # Drop duplicate rows
    # 
    return data

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


def main():

    # Load data
    data = load_data("Lab6-Proj1_TestSet.csv")

    # Preprocess data
    data = preprocess_data(data)

    # Split data
    X,y = split_data(data)

    # Load model
    model = load_model("model.picke")

    # ELSE prediction vector 
    else_pred = else_prediction(X, model)

    # Root mean squared deviation
    print("RMSE of ELSE prediction: ", np.sqrt(mean_squared_error(y, else_pred)))


if __name__ == "__main__":
    main()