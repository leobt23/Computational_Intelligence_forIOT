# Load dataset with float values to pandas dataframe
distances_test = pd.read_csv('distances.csv', header=None)

# pd to numpy array
distances_test = distances_test.to_numpy() 