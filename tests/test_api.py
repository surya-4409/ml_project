import pytest
from src.inference_api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_health_check(client):
    """Test the health check endpoint."""
    rv = client.get('/health')
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert json_data['status'] == 'healthy'

def test_predict_valid(client):
    """Test the predict endpoint with valid data."""
    # Single sample from breast cancer dataset
    data = {
        "features": [[17.99, 10.38, 122.8, 1001.0, 0.1184, 0.2776, 0.3001, 0.1471, 
                      0.2419, 0.0787, 1.095, 0.9053, 8.589, 153.4, 0.006399, 
                      0.04904, 0.05373, 0.01587, 0.03003, 0.006193, 25.38, 17.33, 
                      184.6, 2019.0, 0.1622, 0.6656, 0.7119, 0.2654, 0.4601, 0.1189]]
    }
    rv = client.post('/predict', json=data)
    assert rv.status_code == 200
    json_data = rv.get_json()
    assert 'predictions' in json_data
    assert isinstance(json_data['predictions'], list)

def test_predict_invalid_input(client):
    """Test the predict endpoint with missing features."""
    data = {"wrong_key": []}
    rv = client.post('/predict', json=data)
    assert rv.status_code == 400