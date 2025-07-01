import ee
import os
from flask import Flask, request, jsonify

# Initialize Earth Engine with service account credentials
SERVICE_ACCOUNT = 'kisanshaktiai-n@exalted-legacy-456511-b9.iam.gserviceaccount.com'
CREDENTIALS_PATH = '/etc/secrets/credentials.json'

credentials = ee.ServiceAccountCredentials(SERVICE_ACCOUNT, key_file=CREDENTIALS_PATH)
ee.Initialize(credentials)

app = Flask(__name__)
CORS(app)

# Try to get first available depth from multiple bands
def get_soil_value(lat, lon, image_id_prefix):
    try:
        point = ee.Geometry.Point([lon, lat])
        region = point.buffer(300).bounds()

        # Depth priority
        depths = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]

        for depth in depths:
            band_name = f"{image_id_prefix}_{depth}_mean"
            try:
                image = ee.Image(f"projects/soilgrids-isric/{image_id_prefix}_mean")
                value = image.select(band_name).reduceRegion(
                    reducer=ee.Reducer.first(),
                    geometry=region,
                    scale=250
                ).getInfo()
                if value and band_name in value and value[band_name] is not None:
                    return round(float(value[band_name]), 2)
            except Exception:
                continue

        return "No data"
    except Exception as e:
        return f"Error: {str(e)}"

@app.route('/soil', methods=['GET'])
def soil_data():
    lat = request.args.get('lat', type=float)
    lon = request.args.get('lon', type=float)

    if lat is None or lon is None:
        return jsonify({'error': 'lat and lon are required'}), 400

    layers = {
        "ph": "phh2o",
        "organic_carbon": "ocd",
        "clay": "clay",
        "sand": "sand",
        "silt": "silt",
        "bulk_density": "bdod",
        "cec": "cec"
    }

    result = {
        "latitude": lat,
        "longitude": lon,
        "method": "buffered fallback from 0–5cm downward"
    }

    for key, prefix in layers.items():
        result[key] = get_soil_value(lat, lon, prefix)

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
