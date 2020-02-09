# build a neural network that classifies images
# 1. Build a neural network that classifies images
# 2. Train the neural network
# 3. And, finally, evaluate the accuracy of the model

# import libraries
import tensorflow as tf


# load data
def loadData():
    mnist = tf.keras.datasets.mnist
    (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    xTrain, xTest = xTrain / 255, xTest / 255
    return xTrain, yTrain, xTest, yTest


# build model
def buildModel():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    return model


# load data
xTrain, yTrain, xTest, yTest = loadData()

# build model
model = buildModel()

# define loss function
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# compile model
model.compile(optimizer='adam', loss=lossFn, metrics=['accuracy'])

# fit model to training dataset
model.fit(xTrain, yTrain, epochs=5)

# evaluate model
model.evaluate(xTest, yTest, verbose=2)

# probability model for probabilities
probabilityModel = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

probabilityModel(xTest[:5])
