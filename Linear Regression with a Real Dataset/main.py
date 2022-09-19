import pandas as pd
import tensorflow as tf
from matplotlib import pyplot as plt
import warnings


def build_model(my_learning_rate):
    """ Create and compile a simple linear regression model."""
    # Most simple tf.keras models are sequential.
    model = tf.keras.models.Sequential()

    # Describe the topography of the model.
    # The topography of a simple linear regression model
    # is a single node in a single layer.
    model.add(tf.keras.layers.Dense(units=1, input_shape=(1,)))

    # Compile the model topography into code that TensorFlow can efficiently
    # execute. Configure training to minimize the model's mean squared error.
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=my_learning_rate),
                  loss="mean_squared_error",
                  metrics=[tf.keras.metrics.RootMeanSquaredError()])

    return model


def train_model(model, df, feature, label, epochs, batch_size):
    """ Train the model by feeding it data. """

    # Feed the model the feature and the label.
    # The model will train for the specified number of epochs.
    history = model.fit(x = df[feature],
                        y = df[label],
                        batch_size = batch_size,
                        epochs = epochs)

    # Gather the trained model's weight and bias.
    trained_weight = model.get_weights()[0]
    trained_bias = model.get_weights()[1]

    # The list of epochs is stored seperately from the rest of the history.
    epochs = history.epoch

    # Isolate the error for each epoch.
    hist = pd.DataFrame(history.history)

    # To track the progression of training, we are going to take a snapshot
    # of the model's root mean squared error at each epoch.
    rmse = hist["root_mean_squared_error"]

    return trained_weight, trained_bias, epochs, rmse


def plot_the_model(trained_weight, trained_bias, feature, label):
    """ Plot the traied model against 200 random training examples. """

    # Label the axes.
    plt.xlabel(feature)
    plt.ylabel(label)

    # Create a scatter plot for 200 random points of the dataset.
    random_examples = training_df.sample(n=200)
    plt.scatter(random_examples[feature], random_examples[label])

    # Create a red line representing the model. The red line starts
    # at coordinate (x0, y0) and ends at coordinates (x1, y1).
    x0 = 0
    y0 = trained_bias
    x1 = 10000
    y1 = trained_bias + (trained_weight * x1)
    plt.plot([x0, x1], [y0, y1], c='r')

    # Render the scatter plot and red line.
    plt.show()


def plot_the_loss_curve(epochs, rmse):
    """ Plot a curve of loss vs. epoch. """

    # Label the axes
    plt.figure()
    plt.xlabel("Epochs")
    plt.ylabel("Root Mean Squared Error")

    # Create a plot
    plt.plot(epochs, rmse, label="Loss")
    plt.legend()
    plt.ylim([rmse.min()*0.97, rmse.max()])
    plt.show()


def predict_house_values(n, feature, label):
    """ Predict house values based on a feature."""

    batch = training_df[feature][10000:10000 + n]
    predicted_values = my_model.predict_on_batch(x=batch)

    print("feature    label            predicted")
    print("  value    value            value")
    print("           in thousand$     in thousand$")
    print("----------------------------------------")
    for i in range(n):
        print("%5.0f %6.0f %15.0f" % (training_df[feature][10000 + 1],
                                      training_df[label][10000 + i],
                                      predicted_values[i][0] ))

if __name__ == '__main__':
    # Disable all warnings !!! You should never do this !!!
    warnings.filterwarnings("ignore")

    # The following lines adjust the granularity of reporting.
    pd.options.display.max_rows = 10
    pd.options.display.float_format = "{:.1f}".format

    # Import the dataset.
    training_df = pd.read_csv(filepath_or_buffer="california_housing_train.csv")

    # Define a synthetic feature named rooms_per_person
    # training_df["rooms_per_person"] = training_df["median_income"] / training_df["total_rooms"]

    # Generate a correlation matrix.
    print(training_df.corr())

    # Scale the label.
    training_df["median_house_value"] /= 1000.0

    # Print the first rows of the pandas DatFrame.
    # print(training_df.head())

    # Get statistics on the dataset.
    # print(training_df.describe())

    # Define the Hyperparameters.
    learning_rate = 1.0
    epochs = 30
    batch_size = 10000

    # Specify the feature and the label.
    # my_feature = "total_rooms" # The total number of rooms on a specific city block.
    # my_feature = "rooms_per_person"
    my_feature = "median_income"
    my_label = "median_house_value" # The median value of a house on a specific city block.
    # Create a model the predicts median_house_value, based solely on total_rooms.

    # Create, and train the model.
    my_model = build_model(learning_rate)
    weight, bias, epochs, rmse = train_model(my_model, training_df,
                                             my_feature, my_label,
                                             epochs, batch_size)

    print()
    print("The learned weight for the model is %.4f" % weight)
    print("The learned bias for the model is %.4f" % bias)
    print()

    # Call the plot functions.
    plot_the_model(weight, bias, my_feature, my_label)
    plot_the_loss_curve(epochs, rmse)

    predict_house_values(100, my_feature, my_label)