import numpy as np
import pandas as pd
import tensorflow as tf
import warnings
from tensorflow import feature_column
from tensorflow.keras import layers
from matplotlib import pyplot as plt


def create_model(my_learning_rate, feature_layer):
    """ Create and compile a simple linear regression model. """
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add the layer containing the feature columns to the model.
    model.add(feature_layer)

    # Add one linear layer to the model to yield a simple linear regressor.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Construct the layers into a model that TensorFlow can execute.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    """ Feed a dataset into the model in order to train it. """

    features = {name: np.array(value) for name, value in dataset.items()}
    label = np.array(features.pop(label_name))
    history = model.fit(x=features, y=label, batch_size=batch_size,
                        epochs=epochs, shuffle=True)

    # The list of epochs is stored separately from the rest of history.
    epochs = history.epoch

    # Isolate the mean absolute error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse


def plot_the_loss_curve(epochs, rmse):
    """ Plot a curve of loss vs. epoch. """

    # Define the labels
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    # Draw the lines
    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.94, rmse.max()*1.05])
    plt.show()


# Main function
if __name__ == '__main__':

    # This is a bad idea
    warnings.filterwarnings("always")

    # The following lines adjust the granularity of reporting.
    pd.options.display.max_rows = 10
    pd.options.display.float_format = "{:.1f}".format

    tf.keras.backend.set_floatx('float32')

    # Load the datasets
    train_df = pd.read_csv("california_housing_train.csv")
    test_df = pd.read_csv("california_housing_test.csv")

    # Set the scale factor for the labels.
    scale_factor = 1000.0

    # Define to size or resolution of the latitude and longitude buckets.
    resolution_in_degrees = 0.1

    # Scale the labels in the dataframes
    train_df["median_house_value"] /= scale_factor
    test_df["median_house_value"] /= scale_factor

    # Shuffle the examples
    train_df = train_df.reindex(np.random.permutation(train_df.index))

    # Create an empty list that will eventually hood all  feature columns.
    feature_columns = []

    # Create a numerical feature column to represent latitude.
    # latitude = feature_column.numeric_column("latitude")
    # feature_columns.append(latitude)

    # Create a bucket feature column for latitude.
    latitude_as_a_numeric_column = feature_column.numeric_column("latitude")
    latitude_boundaries = list(np.arange(int(min(train_df['latitude'])), int(max(train_df['latitude'])), resolution_in_degrees))
    latitude = feature_column.bucketized_column(latitude_as_a_numeric_column, latitude_boundaries)
    # feature_columns.append(latitude)

    # Create a bucket feature column for longitude.
    longitude_as_a_numeric_column = feature_column.numeric_column("longitude")
    longitude_boundaries = list(np.arange(int(min(train_df['longitude'])), int(max(train_df['longitude'])), resolution_in_degrees))
    longitude = feature_column.bucketized_column(longitude_as_a_numeric_column, longitude_boundaries)
    # feature_columns.append(longitude)

    # Create a feature cross of latitude and longitude.
    latitude_x_longitude = feature_column.crossed_column([latitude, longitude], hash_bucket_size=1000)
    crossed_feature = feature_column.indicator_column(latitude_x_longitude)
    feature_columns.append(crossed_feature)

    # Create a numerical feature column to represent longitude.
    # longitude = feature_column.numeric_column("longitude")
    # feature_columns.append(longitude)

    # Convert the list of feature columns into a layer that will ultimately become
    # part of the model.
    # fp_feature_layer = layers.DenseFeatures(feature_columns)
    # buckets_feature_layer = layers.DenseFeatures(feature_columns)
    feature_cross_feature_layer = layers.DenseFeatures(feature_columns)

    # Define the Hyperparameters.
    learning_rate = 0.05
    epochs = 75
    batch_size = 100
    label_name = "median_house_value"

    # Create and compile the model's topography.
    my_model = create_model(learning_rate, feature_cross_feature_layer)

    # Train the model on the training set.
    epochs, rmse = train_model(my_model, train_df, epochs, batch_size, label_name)

    # Plot the loss curve
    plot_the_loss_curve(epochs, rmse)

    # Test the model against the test set
    print("\n: Evaluate the new model against the test set:")
    test_features = {name: np.array(value) for name, value in test_df.items()}
    test_label = np.array(test_features.pop(label_name))
    my_model.evaluate(x=test_features, y=test_label, batch_size=batch_size)