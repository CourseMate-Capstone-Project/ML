from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import tensorflow as tf
import pickle
import logging
import os
import requests
from flask_cors import CORS

# Flask app initialization
app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# URLs for the model and dataset
model_url = "https://storage.googleapis.com/course-recommendation-data-bucket/models/course_recommendation_model.keras"
course_data_url = "https://storage.googleapis.com/course-recommendation-data-bucket/data/cleaned_courses.csv"

# Local paths for the model
local_model_path = "./models/course_recommendation_model.keras"

# Download model if not present
os.makedirs("./models", exist_ok=True)

if not os.path.exists(local_model_path):
    logging.info(f"Downloading model from {model_url}")
    response = requests.get(model_url)
    response.raise_for_status()
    with open(local_model_path, "wb") as f:
        f.write(response.content)
    logging.info(f"Model downloaded successfully: {local_model_path}")
else:
    logging.info(f"Model already exists: {local_model_path}")

# Load the pre-trained model
model = tf.keras.models.load_model(local_model_path)

# Load the course dataset directly from the URL
course_data = pd.read_csv(course_data_url)

# Load mappings from .pkl file
mappings_url = "https://storage.googleapis.com/course-recommendation-data-bucket/mappings/mappings.pkl"
response = requests.get(mappings_url)

# Pastikan file berhasil diunduh dan dimuat
if response.status_code == 200:
    mappings = pickle.loads(response.content)

    # Mengakses elemen-elemen dari mappings
    SUBCATEGORY_MAPPING = mappings.get("SUBCATEGORY_MAPPING")
    COURSE_TYPE_MAPPING = mappings.get("COURSE_TYPE_MAPPING")
    CATEGORY_MAPPING = mappings.get("CATEGORY_MAPPING")
    DURATION_MAPPING = mappings.get("DURATION_MAPPING")

    # Pastikan data ada dan valid
    if SUBCATEGORY_MAPPING and COURSE_TYPE_MAPPING and CATEGORY_MAPPING and DURATION_MAPPING:
        logging.info("Mappings loaded successfully")
    else:
        logging.error("Some mappings are missing or invalid.")
else:
    logging.error(f"Failed to retrieve the file. Status code: {response.status_code}")
    raise Exception("Failed to load mappings.")

@app.route("/api/recommend", methods=["POST"])
def recommend_courses():
    try:
        # Get JSON input
        data = request.get_json()

        user_subcategory = data.get('subcategory')
        user_course_type = data.get('course_type')
        user_duration = data.get('duration')

        if not user_subcategory or not user_course_type or not user_duration:
            return jsonify({"error": "All fields must be filled out."}), 400

        try:
            user_duration = int(user_duration)
        except ValueError:
            return jsonify({"error": "Duration must be a number."}), 400

        # Map user inputs to encoded values
        subcategory_encoded = SUBCATEGORY_MAPPING.get(user_subcategory, None)
        course_type_encoded = COURSE_TYPE_MAPPING.get(user_course_type, None)
        duration_scaled = DURATION_MAPPING.get(user_duration, None)

        if subcategory_encoded is None or course_type_encoded is None or duration_scaled is None:
            return jsonify({"error": "Invalid inputs. Please check your entries."}), 400

        # Create feature array for prediction
        user_features = np.array([[subcategory_encoded, course_type_encoded, duration_scaled]])
        logging.debug(f"Encoded user features: {user_features}")

        # Model prediction
        predictions = model.predict(user_features)
        predicted_category_encoded = predictions.argmax(axis=-1)[0]
        predicted_category = [key for key, value in CATEGORY_MAPPING.items() if value == predicted_category_encoded][0]

        # Filter and recommend courses based on predicted category and user-subcategory input
        filtered_courses = course_data[course_data['Category'] == predicted_category]

        # Filter the courses further based on the user-subcategory input
        if user_subcategory:
            filtered_courses = filtered_courses[filtered_courses['Sub-Category'] == user_subcategory]

        # Sort and select top 10 courses
        top_courses = filtered_courses.sort_values(by='Duration').head(10)

        # Prepare recommendations
        recommended_courses = [
            {
                "title": row["Title"],
                "short_intro": row["Short Intro"][:150] + "..." if len(row["Short Intro"]) > 150 else row["Short Intro"],
                "url": row["URL"]
            }
            for _, row in top_courses.iterrows()
        ]

        return jsonify({
            "predicted_category": predicted_category,
            "recommended_courses": recommended_courses
        }), 200

    except Exception as e:
        logging.error(f"Error processing request: {e}")
        return jsonify({"error": "An error occurred. Please try again."}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
