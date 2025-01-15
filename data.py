from pandas import read_csv
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer

# Loading data
training_file = "home-data-for-ml-course/train.csv"
testing_file = "home-data-for-ml-course/test.csv"
train_raw_data = open(training_file, 'rt')
testing_raw_data = open(testing_file, 'rt')
training_dataset = read_csv(train_raw_data)
testing_dataset = read_csv(testing_raw_data)

print("Welcome to the the data page for this program, where we can look further in the house data being used:")
print("To see the Dimensions for both datasets click 1, To see the actual data for both datasets(only the "
      "first 20) click 2")
print("To see the statistical summary for both datasets click 3, enter 4 for some "
      "visualization for the datasets To exit enter e: ")
user_input = input("Response: ")

# looking into both datasets
if user_input == "1":
    # looking into both datasets
    print("Showing the training data dimensions")
    print(training_dataset.shape)
    print("Showing the testing data dimensions")
    print(testing_dataset.shape)
elif user_input == "2":
    # looking into the actual data
    print("Showing the actual training data")
    print(training_dataset.head(20))
    print("Showing the actual testing data")
    print(testing_dataset.head(20))
elif user_input == "3":
    # getting statistical summary for the dataset
    print("Showing statistical summary for the training dataset")
    print(training_dataset.describe())
    print("Showing statistical summary for the testing dataset")
    print(testing_dataset.describe())
elif user_input == "4":
    # visualization of data
    # Using a Histogram to better understand each attribute
    print("Showing the histogram for the training dataset")
    training_dataset.hist()
    plt.show()
    print("Showing the histogram for the testing dataset")
    testing_dataset.hist()
    plt.show()
    # Using Scatter plot matrix to better understand the relationship between attributes
    print("Showing the scatter_matrix for the training dataset")
    scatter_matrix(training_dataset)
    plt.show()
    print("Showing the scatter_matrix for the testing dataset")
    scatter_matrix(testing_dataset)
    plt.show()

# preprocessing the data for non-numeric features
# separating the features and target variables
target_column = "SalePrice"
X_training = training_dataset.drop(columns=[target_column])
y_training = training_dataset[target_column]
X_testing = testing_dataset
# Defining the numeric and categorical columns
numeric_features = X_training.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X_training.select_dtypes(include=[object]).columns
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)
