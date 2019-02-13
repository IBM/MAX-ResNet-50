from PIL import Image
from keras.backend import clear_session
from keras import models
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import io
import numpy as np
import logging

logger = logging.getLogger()


class ModelWrapper(object):
    """Model wrapper for Keras models"""

    MODEL_NAME = 'resnet50'
    DEFAULT_MODEL_PATH = 'assets/{}.h5'.format(MODEL_NAME)
    MODEL_INPUT_IMG_SIZE = (224, 224)
    MODEL_LICENSE = 'MIT'

    MODEL_META_DATA = {
        'id': '{}-keras-imagenet'.format(MODEL_NAME.lower()),
        'name': '{} Keras Model'.format(MODEL_NAME),
        'description': '{} Keras model trained on ImageNet'.format(MODEL_NAME),
        'type': 'image_classification',
        'license': '{}'.format(MODEL_LICENSE)
    }

    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))
        clear_session()
        self.model = models.load_model(path)
        # this seems to be required to make Keras models play nicely with threads
        self.model._make_predict_function()
        logger.info('Loaded model: {}'.format(self.model.name))

    def read_image(self, image_data):
        return Image.open(io.BytesIO(image_data))

    def _preprocess_image(self, image, target, mode='tf'):
        image = image.resize(target)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return imagenet_utils.preprocess_input(image, mode=mode)

    def _post_process_result(self, preds):
        return imagenet_utils.decode_predictions(preds)[0]

    def predict(self, x):
        x = self._preprocess_image(x, target=self.MODEL_INPUT_IMG_SIZE, mode='caffe')
        preds = self.model.predict(x)
        return self._post_process_result(preds)
