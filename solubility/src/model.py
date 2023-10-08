import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


"""
    # Solubility of Different Molecules
"""

dv = pd.read_csv("delaney_solubility.csv")


""" 
    Output of `dv`:

    - logS is our `y` variable.
    - MolLogP, MolWt, NumRoratableBonds, 
      AromaticProportions will all be our `x` variables.


       MolLogP       MolWt     NumRotatableBonds  AromaticProportion     logS

1     2.37650       133.405          0.0            0.000000            -2.000
0     2.59540       167.850          0.0            0.000000            -2.180
2     2.59380       167.850          1.0            0.000000            -1.740
3     2.02890       133.405          1.0            0.000000            -1.480
4     2.91890       187.375          1.0            0.000000            -3.040
...     ...           ...            ...               ...                ...
"""


"""
    # Data Separation of X and Y.
"""
# The Y variable is a function of all the X variables.
y = dv["logS"]  # Y is the last column; or the resulting solubility.
x = dv.drop("logS", axis=1)  # X is everything else that correlates to solubility

# Data Splitting using sklearn and train_test_split.
# The test data size will be 20% of the data set
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=100)


def load_data(file_path):
    """
        Load the dataset from a specified file path.
        
        Parameters:
            file_path (str): The path to the data file.
        
        Returns:
            pd.DataFrame: The loaded dataset.
    """
    return pd.read_csv(file_path)


def explore_data(data):
    """
        Perform basic data exploration.
        
        Parameters:
            data (pd.DataFrame): The dataset to explore.
    """
    print(data.head())
    print(data.describe())
    print(data.info())
    # Additional exploration code as per requirements...


def show_plots(data):
    """
        Visualize the relationship between solubility and various molecular descriptors.
        
        Parameters:
            data (pd.DataFrame): The dataset containing the molecular descriptor data and solubility values.
    """
    # Create a 2x2 grid of subplots with a specific figure size.
    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # Flatten the 2x2 grid into a 1-D array to iterate over it.
    axs = axs.flatten()

    # Define a list of independent variables for which we want to visualize the relationship with logS.
    x_vars = ['MolLogP', 'MolWt', 'NumRotatableBonds', 'AromaticProportion']

    # Define a list of colors to be used in the plots.
    colors = ['blue', 'green', 'red', 'purple']

    # Iterate over each subplot, variable, and color.
    for i, (ax, x_var, color) in enumerate(zip(axs, x_vars, colors)):
        # Create a scatter plot on the current subplot (ax).
        ax.scatter(data[x_var], data['logS'], alpha=0.6, color=color)

        # Add trend lines for the first two subplots
        if i < 2:
            z = np.polyfit(data[x_var], data['logS'], 1)
            p = np.poly1d(z)
            ax.plot(data[x_var], p(data[x_var]), linestyle='--', color='black')

        # Set the title of the subplot.
        ax.set_title(f'logS vs {x_var}')

        # Label the x and y axes.
        ax.set_xlabel(x_var)
        ax.set_ylabel('logS')

        # Add a grid for better readability of the plot.
        ax.grid(True, linestyle='--', alpha=0.7)

    # Adjust the layout to prevent overlap between subplots.
    plt.tight_layout()

    # Add a super title for all subplots.
    plt.suptitle('Solubility vs Molecular Descriptors', fontsize=16, y=1.02)

    # Display the plots.
    plt.show()



def preprocess_data(data, target_col):
    """
        Preprocess the data by separating the target variable and splitting the data into training and test sets.
        
        Parameters:
            data (pd.DataFrame): The dataset to preprocess.
            target_col (str): The name of the target column.
        
        Returns:
            tuple: Contains the training and test data (x_train, x_test, y_train, y_test).
    """
    y = data[target_col]
    x = data.drop(target_col, axis=1)
    return train_test_split(x, y, test_size=0.2, random_state=100)


def train_model(model, x_train, y_train):
    """
        Train the specified model using the training data.
        
        Parameters:
            model (model instance): The model to train.
            x_train (pd.DataFrame): The training data features.
            y_train (pd.Series): The training data target variable.
        
        Returns:
            model instance: The trained model.
    """
    model.fit(x_train, y_train)
    return model


def evaluate_model(name, model, x_train, x_test, y_train, y_test):
    """
        Evaluate the model using both training and test data.
        
        Parameters:
            name (str): The name of the model.
            model (model instance): The model to evaluate.
            x_train (pd.DataFrame): The training data features.
            x_test (pd.DataFrame): The test data features.
            y_train (pd.Series): The training data target variable.
            y_test (pd.Series): The test data target variable.
        
        Returns:
            pd.DataFrame: A DataFrame containing the evaluation metrics.
    """
    predictions_train = model.predict(x_train)
    predictions_test = model.predict(x_test)

    # Compute metrics for training data
    mse_train = mean_squared_error(y_train, predictions_train)
    r2_train = r2_score(y_train, predictions_train)

    # Compute metrics for test data
    mse_test = mean_squared_error(y_test, predictions_test)
    r2_test = r2_score(y_test, predictions_test)

    # Display results
    print(f"\n{name} Model Evaluation:")
    print(f"Training: MSE = {mse_train:.2f}, R2 = {r2_train:.2f}")
    print(f"Test: MSE = {mse_test:.2f}, R2 = {r2_test:.2f}")

    # Create and return results DataFrame
    results_df = pd.DataFrame({
        "Model": [name],
        "Training MSE": [mse_train],
        "Training R2": [r2_train],
        "Test MSE": [mse_test],
        "Test R2": [r2_test]
    })
    return results_df


def main():
    """
        Main execution function to orchestrate the data 
        loading, exploration, preprocessing, model training, and evaluation.
    """
    file_path = "delaney_solubility.csv"
    target_col = "logS"

    data = load_data(file_path)
    explore_data(data)
    show_plots(data)
    x_train, x_test, y_train, y_test = preprocess_data(data, target_col)

    lr_model = train_model(LinearRegression(), x_train, y_train)
    lr_results = evaluate_model(
        "Linear Regress.", lr_model, x_train, x_test, y_train, y_test)

    rf_model = train_model(RandomForestRegressor(
        max_depth=2, random_state=100), x_train, y_train)
    rf_results = evaluate_model(
        "Random Forest", rf_model, x_train, x_test, y_train, y_test)

    # Concatenating the results
    all_results = pd.concat([lr_results, rf_results],
                            axis=0).reset_index(drop=True)
    print("\nAll Models Evaluation Results:")
    print(all_results)


if __name__ == "__main__":
    main()
