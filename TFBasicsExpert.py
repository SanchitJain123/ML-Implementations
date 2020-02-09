# import libraries

import tensorflow as tf

mnist = tf.keras.datasets.mnist

(xTrain, yTrain), (xTest, yTest) = mnist.load_data()
xTrain, xTest = xTrain / 255.0, xTest / 255.0

# add a channels dimension
xTrain = xTrain[..., tf.newaxis]
xTest = xTest[..., tf.newaxis]

# shuffle and batch data
train = tf.data.Dataset.from_tensor_slices(
    (xTrain, yTrain)
).shuffle(10000).batch(32)

test = tf.data.Dataset.from_tensor_slices(
    (xTest, yTest)
).batch(32)


class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, 3, activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.d1 = tf.keras.layers.Dense(128, activation='relu')
        self.d2 = tf.keras.layers.Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)
        return x


# create an instance of model
model = MyModel()

# define loss function
lossFn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = tf.keras.optimizers.Adam()

trainLoss = tf.keras.metrics.Mean(name='train_loss')
trainAccuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

testLoss = tf.keras.metrics.Mean(name='test_loss')
testAccuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


@tf.function
def trainStep(images, labels):
    with tf.GradientTape() as tape:
        # training=True is needed when the layers need to behave differently for training
        # and inference (e.g. Dropout)
        predictions = model(images, training=True)
        loss = lossFn(labels, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    trainLoss(loss)
    trainAccuracy(labels, predictions)


@tf.function
def testStep(images, labels):
    predictions = model(images, training=False)
    loss = lossFn(labels, predictions)

    testLoss(loss)
    testAccuracy(labels, predictions)


EPOCHS = 5

for epoch in range(EPOCHS):
  # Reset the metrics at the start of the next epoch
  trainLoss.reset_states()
  trainAccuracy.reset_states()
  testLoss.reset_states()
  testAccuracy.reset_states()

  for images, labels in train:
    trainStep(images, labels)

  for test_images, test_labels in test:
    testStep(test_images, test_labels)

  template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
  print(template.format(epoch+1,
                        trainLoss.result(),
                        trainAccuracy.result()*100,
                        testLoss.result(),
                        testAccuracy.result()*100))
