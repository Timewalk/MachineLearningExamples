import numpy as np
import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import warnings


def build_model(my_learning_rate):
    """ Create and compile a simple linear regression model. """
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Add one linear layer to the model to yield a simple linear regressor.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Compile the model topography into code that TensorFlow can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, df, feature, label, my_epochs,
                my_batch_size=None, my_validation_split=0.01):
    """ Feed a dataset into the model in order to train it. """

    history = model.fit(x=df[feature],
                        y=df[label],
                        batch_size=my_batch_size,
                        epochs=my_epochs,
                        validation_split=my_validation_split)

    # Gather the model's trained weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored separately from the
    # rest of the history.
    epochs = history.epoch

    # Isolate the root mean squared error for each epoch.
    hist = pd.DataFrame(history.history)
    rmse = hist["root_mean_squared_error"]

    return epochs, rmse, history.history


def plot_the_loss_curve(epochs, mae_training, mae_validation):
    """ Plot a curve of loss vs. epoch."""

    # Set up the graph.
    plt.figure()
    plt.xlabel("Epoch")
    plt.ylabel("Root Mean Squared Error")

    # Plot the lines.
    plt.plot(epochs[1:], mae_training[1:], label="Training Loss")
    plt.plot(epochs[1:], mae_validation[1:], label="Validation Loss")
    plt.legend()

    # Don't plot the first epoch since the loss on the first epoch
    # is often substantially greater than the loss for other epochs.
    merged_mae_lists = mae_training[1:] + mae_validation[1:]
    highest_loss = max(merged_mae_lists)
    lowest_loss = min(merged_mae_lists)
    delta = highest_loss - lowest_loss
    print(delta)

    # Define graph limits.
    top_of_y_axis = highest_loss + (delta * 0.05)
    bottom_of_y_axis = lowest_loss - (delta * 0.05)

    # Set the graph limits.
    plt.ylim([bottom_of_y_axis, top_of_y_axis])
    plt.show()


# Main function
if __name__ == '__main__':
    warnings.filterwarnings("ignore")

    # Initialize display option.
    pd.options.display.max_rows = 10
    pd.options.display.float_format = "{:.1f}".format

    # Load datasets.
    train_df = pd.read_csv("california_housing_train.csv")
    test_df = pd.read_csv("california_housing_test.csv")

    # Define a scaler to improve model performance.
    scale_factor = 1000.0

    # Scale the training set's label.
    train_df["median_house_value"] /= scale_factor

    # Scale the test set's label.
    test_df["median_house_value"] /= scale_factor

    # Define the hyperparameters.
    learning_rate = 0.05
    epochs = 300
    batch_size = 2000

    # Split the original training set into a reduced training set
    # and a validation set.
    validation_split = 0.3

    # Identify the feature and the label.
    my_feature = "median_income"     # The median income on a specific city block.
    my_label = "median_house_value"  # The median house value on a specific city block.
    # The model will predict a house value based solely on the neighborhoods'
    # median income.

    # Shuffle the data in the training set.
    shuffled_train_df = train_df.reindex(np.random.permutation(train_df.index))

    # Invoke the functions to build and train the model.
    my_model = build_model(learning_rate)
    epochs, rmse, history = train_model(my_model, shuffled_train_df, my_feature,
                                        my_label, epochs, batch_size, validation_split)

    # Select an item form the test data.
    x_test = test_df[my_feature]
    y_test = test_df[my_label]

    # Run the test data trough the model.
    results = my_model.evaluate(x_test, y_test, batch_size=batch_size)
    print(results)

    # Call the plot functions.
    plot_the_loss_curve(epochs, history["root_mean_squared_error"],
                        history["val_root_mean_squared_error"])

