import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'MAX ResNet 50'
    assert json.get('info') and json.get('info').get('description') == 'Identify objects in images using a ' + \
                                                                       'first-generation deep residual network.'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'resnet50-keras-imagenet'
    assert metadata['name'] == 'resnet50 Keras Model'
    assert metadata['description'] == 'resnet50 Keras model trained on ImageNet'
    assert metadata['license'] == 'MIT'

def _check_response(r):
    assert r.status_code == 200

    response = r.json()

    assert response['status'] == 'ok'
    assert response['predictions'][0]['label_id'] == 'n07697313'
    assert response['predictions'][0]['label'] == 'cheeseburger'
    assert response['predictions'][0]['probability'] > 0.75

def test_predict():
    model_endpoint = 'http://localhost:5000/model/predict'
    formats = ['jpg', 'png']
    file_path = 'tests/burger.{}'

    for f in formats:
        p = file_path.format(f)
        with open(p, 'rb') as file:
            file_form = {'image': (p, file, 'image/{}'.format(f))}
            r = requests.post(url=model_endpoint, files=file_form)
        _check_response(r)

    # Test invalid
    file_path = 'README.md'

    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__])
