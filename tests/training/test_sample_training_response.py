#
# Copyright 2018-2019 IBM Corp. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX Object Detector'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'resnet50-keras-imagenet'
    assert metadata['name'] == 'resnet50 Keras Model'
    assert metadata['description'] == 'resnet50 Keras model trained on ImageNet'
    assert metadata['license'] == 'MIT'
    assert metadata['type'] == 'image_classification'
    assert metadata['source'] == 'https://developer.ibm.com/exchanges/models/all/max-resnet-50/'


def test_predict():
    model_endpoint = 'http://localhost:5000/model/predict'
    file_path = 'samples/toy_trained.jpg'

    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200
    response = r.json()

    assert response['status'] == 'ok'
    assert response['predictions'][0]['label'] == 'toy'


if __name__ == '__main__':
    pytest.main([__file__])
