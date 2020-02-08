import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import time


class NeuralStyleTransfer:

    def __init__(self):
        mpl.rcParams['figure.figsize'] = (6, 6)
        mpl.rcParams['axes.grid'] = False
        self.contentPath = ''
        self.stylePath = ''

    def downloadImages(self):
        self.contentPath = tf.keras.utils.get_file(
            'YellowLabradorLooking_new.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/' +
            'example_images/YellowLabradorLooking_new.jpg'
        )

        self.stylePath = tf.keras.utils.get_file(
            'kandinsky5.jpg',
            'https://storage.googleapis.com/download.tensorflow.org/' +
            'example_images/Vassily_Kandinsky%2C_1913_-_Composition_7.jpg'
        )

        print('Content Image Path: ', self.contentPath)
        print('Style Image Path: ', self.stylePath)
        return

    @staticmethod
    def loadImage(pathToImage):
        maxDim = 512
        image = tf.io.read_file(pathToImage)
        image = tf.image.decode_image(image, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)

        shape = tf.cast(tf.shape(image)[:-1], tf.float32)
        longDim = max(shape)
        scale = maxDim / longDim

        newShape = tf.cast(shape * scale, tf.int32)

        image = tf.image.resize(image, newShape)
        image = image[tf.newaxis, :]
        print(image.shape)
        return image

    @staticmethod
    def showImage(image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.show(block=True)
        return

    @staticmethod
    def tensor_to_image(tensor):
        tensor = tensor * 255
        tensor = np.array(tensor, dtype=np.uint8)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        return tensor


if __name__ == '__main__':
    ns = NeuralStyleTransfer()
    ns.downloadImages()
    contentImage = ns.loadImage(ns.contentPath)
    styleImage = ns.loadImage(ns.stylePath)
    # ns.showImage(contentImage, "Dog Picture")
    # ns.showImage(styleImage, "Style Picture")
    import tensorflow_hub as hub

    hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')
    stylized_image = hub_module(tf.constant(contentImage), tf.constant(styleImage))[0]
    stylized_image = ns.tensor_to_image(stylized_image)
    ns.showImage(stylized_image)
