import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

# Load the iris dataset
iris = load_iris()

# Get the feature names and target names
feature_names = iris.feature_names
target_names = iris.target_names

def create_data(feature_names, target_names, num_rows=50, std_mod=0):
    '''
    Create synthetic data using the mean and standard deviation of the real data.
    The synthetic data will be normally distributed with the same mean and standard deviation as the real data.
    The standard deviation can be modified to increase or decrease the variation in the synthetic data.

    Parameters
    ----------
    feature_names : list
        A list of the feature names
    target_names : list
        A list of the target names
    num_rows : int, optional
        The number of rows to generate for each target class. The default is 50.
    std_mod : float, optional
        The standard deviation modifier. The default is 0.

    Returns
    -------
    df : pandas dataframe
        A dataframe containing the synthetic data and target values
    '''

    # Create empty lists to store the synthetic data
    synthetic_data = []
    synthetic_targets = []

    # Generate synthetic data for each target class
    for target_idx, target_name in enumerate(target_names):
        # Get the real data for this target class
        real_data = iris.data[iris.target == target_idx]
        # Calculate the mean and standard deviation of each feature for this target class
        feature_means = np.mean(real_data, axis=0)
        feature_stds = np.std(real_data, axis=0)

        # feature_stds = [x+std_mod for x in feature_stds]

        # Generate synthetic data using a normal distribution with the same mean and standard deviation as the real data
        # divide std for lower variation. Multiply for more?
        synthetic_data.append(np.random.normal(loc=feature_means, 
                                            scale=feature_stds*std_mod, 
                                            size=(num_rows, 4)))
        
        # Assign the target value for this synthetic data
        synthetic_targets.append([target_name]*num_rows)
        print(f'name: {target_name} - mean: {feature_means} - std: {feature_stds}')

    # Combine the synthetic data for each target class into a single array and shuffle the rows
    synthetic_data = np.concatenate(synthetic_data, axis=0)
    synthetic_targets = np.concatenate(synthetic_targets, axis=0)
    shuffle_idx = np.random.permutation(len(synthetic_data))
    synthetic_data = synthetic_data[shuffle_idx]
    synthetic_targets = synthetic_targets[shuffle_idx]

    # Create a dataframe with the synthetic data and target values
    df = pd.DataFrame(synthetic_data, columns=feature_names)
    df['target'] = synthetic_targets

    return df

    # drop target column
    # df = df.drop('target', axis=1)

