import pytest
import requests


def test_swagger():

    model_endpoint = 'http://localhost:5000/swagger.json'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200
    assert r.headers['Content-Type'] == 'application/json'

    json = r.json()
    assert 'swagger' in json
    assert json.get('info') and json.get('info').get('title') == 'Model Asset Exchange Server'


def test_metadata():

    model_endpoint = 'http://localhost:5000/model/metadata'

    r = requests.get(url=model_endpoint)
    assert r.status_code == 200

    metadata = r.json()
    assert metadata['id'] == 'resnet50-keras-imagenet'
    assert metadata['name'] == 'resnet50 Keras Model'
    assert metadata['description'] == 'resnet50 Keras model trained on ImageNet'
    assert metadata['license'] == 'MIT'


def test_predict():
    model_endpoint = 'http://localhost:5000/model/predict'
    file_path = 'assets/burger.jpg'

    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 200

    response = r.json()

    assert response['status'] == 'ok'
    assert response['predictions'][0]['label_id'] == 'n07697313'
    assert response['predictions'][0]['label'] == 'cheeseburger'
    assert response['predictions'][0]['probability'] > 0.75

    # Test invalid
    file_path = 'assets/README.md'

    with open(file_path, 'rb') as file:
        file_form = {'image': (file_path, file, 'image/jpeg')}
        r = requests.post(url=model_endpoint, files=file_form)

    assert r.status_code == 400


if __name__ == '__main__':
    pytest.main([__file__])
