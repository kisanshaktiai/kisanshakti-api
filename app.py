
import ee
import os
from flask import Flask, request, jsonify

# ✅ Use service account from Render secret path
SERVICE_ACCOUNT = 'kisanshaktiai-n@exalted-legacy-456511-b9.iam.gserviceaccount.com'
CREDENTIALS_PATH = '/etc/secrets/credentials.json'

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_file=CREDENTIALS_PATH)
ee.Initialize(credentials)

app = Flask(__name__)

# ✅ Use shared SoilGrids image with band selection
def get_soil_value(lat, lon, band_name):
    try:
        point = ee.Geometry.Point([lon, lat])
        image = ee.Image("projects/soilgrids-isric/soilgrids_mean").select(band_name)
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

    if lat is None or lon is None:
        return jsonify({'error': 'lat and lon are required'}), 400

    layers = {
        "ph": "phh2o_0-5cm_mean",
        "organic_carbon": "ocd_0-5cm_mean",
        "clay": "clay_0-5cm_mean",
        "sand": "sand_0-5cm_mean",
        "silt": "silt_0-5cm_mean",
        "bulk_density": "bdod_0-5cm_mean",
        "cec": "cec_0-5cm_mean"
    }

    result = {
        "latitude": lat,
        "longitude": lon,
        "depth": "0–5 cm"
    }

    for key, band in layers.items():
        result[key] = get_soil_value(lat, lon, band) or "N/A"

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
