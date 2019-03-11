import io
import logging
from PIL import Image
from keras.backend import clear_session
from keras import models
from keras.preprocessing.image import img_to_array
from keras.applications import imagenet_utils
import numpy as np
from maxfw.model import MAXModelWrapper
from config import DEFAULT_MODEL_PATH
from flask import abort


logger = logging.getLogger()


class ModelWrapper(MAXModelWrapper):
    """Model wrapper for Keras models"""

    MODEL_NAME = 'resnet50'
    MODEL_INPUT_IMG_SIZE = (224, 224)
    MODEL_LICENSE = 'MIT'
    MODEL_MODE = 'caffe'
    MODEL_META_DATA = {
        'id': '{}-keras-imagenet'.format(MODEL_NAME.lower()),
        'name': '{} Keras Model'.format(MODEL_NAME),
        'description': '{} Keras model trained on ImageNet'.format(MODEL_NAME),
        'type': 'image_classification',
        'license': '{}'.format(MODEL_LICENSE),
        'source': 'https://developer.ibm.com/exchanges/models/all/max-resnet-50/'
    }

    def __init__(self, path=DEFAULT_MODEL_PATH):
        logger.info('Loading model from: {}...'.format(path))
        clear_session()
        self.model = models.load_model(path)
        # this seems to be required to make Keras models play nicely with threads
        self.model._make_predict_function()
        logger.info('Loaded model: {}'.format(self.model.name))

    def read_image(self, image_data):
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except IOError:
            abort(400, 'Invalid file type/extension. Please provide a valid image (supported formats: JPEG, PNG, TIFF).')

    def _pre_process(self, image):
        image = image.resize(self.MODEL_INPUT_IMG_SIZE)
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        return imagenet_utils.preprocess_input(image, mode=self.MODEL_MODE)

    def _post_process(self, preds):
        return imagenet_utils.decode_predictions(preds)[0]

    def _predict(self, x):
        return self.model.predict(x)
