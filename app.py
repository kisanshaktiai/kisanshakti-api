import ee
import os
from flask import Flask, request, jsonify

# Load GEE credentials (Render stores this as env var or uploaded file)
CREDENTIALS_PATH = "/etc/secrets/credentials.json"
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = CREDENTIALS_PATH

# Initialize Earth Engine
ee.Initialize()

app = Flask(__name__)

def get_soil_value(lat, lon, image_id):
    try:
        point = ee.Geometry.Point([lon, lat])
        image = ee.Image(image_id)
        sample = image.sample(region=point, scale=250).first()
        if sample:
            props = sample.getInfo().get('properties', {})
            return list(props.values())[0]
        else:
            return None
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/soil', methods=['GET'])
def soil_data():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    if not lat or not lon:
        return jsonify({'error': 'lat and lon are required'}), 400

    layers = {
        "ph": 'projects/soilgrids-isric/phh2o_0-5cm_mean',
        "organic_carbon": 'projects/soilgrids-isric/ocd_0-5cm_mean',
        "clay": 'projects/soilgrids-isric/clay_0-5cm_mean',
        "sand": 'projects/soilgrids-isric/sand_0-5cm_mean',
        "silt": 'projects/soilgrids-isric/silt_0-5cm_mean',
        "bulk_density": 'projects/soilgrids-isric/bdod_0-5cm_mean',
        "cec": 'projects/soilgrids-isric/cec_0-5cm_mean'
    }

    result = {"latitude": lat, "longitude": lon, "depth": "0–5 cm"}

    for key, image_id in layers.items():
        value = get_soil_value(lat, lon, image_id)
        result[key] = value if value is not None else "N/A"

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
