from flask import Flask, request, jsonify
from flask_restx import Api, Resource, fields
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

# Initialize Flask-RESTX API with Swagger UI
api = Api(
    app, 
    version="1.0", 
    title="Course Recommendation API", 
    description="""API untuk memberikan rekomendasi kursus berdasarkan minat, tipe kursus, dan durasi. 
    Endpoint ini membantu pengguna menemukan kursus yang sesuai dengan preferensi mereka.""",
    doc="/api/recommend"
)

# Define the model for Swagger documentation
recommendation_model = api.model('RecommendationModel', {
    'interest': fields.String(
        required=True, 
        description='Minat kursus yang diinginkan (misalnya: Machine Learning, Data Analysis, dsb.).',
        example="Machine Learning"
    ),
    'course_type': fields.String(
        required=True, 
        description='Jenis kursus yang diinginkan (misalnya: Course, Specialization, Professional Certificate, atau Project).',
        example="Course"
    ),
    'duration': fields.String(
        required=True, 
        description='Durasi maksimum kursus dalam minggu (misalnya: 4, 8, 12, dst.).',
        example="8"
    )
})

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

# Ensure the mappings are loaded
if response.status_code == 200:
    mappings = pickle.loads(response.content)

    # Extract mappings
    INTEREST_MAPPING = mappings.get("SUBCATEGORY_MAPPING")
    COURSE_TYPE_MAPPING = mappings.get("COURSE_TYPE_MAPPING")
    CATEGORY_MAPPING = mappings.get("CATEGORY_MAPPING")
    DURATION_MAPPING = mappings.get("DURATION_MAPPING")

    if not (INTEREST_MAPPING and COURSE_TYPE_MAPPING and CATEGORY_MAPPING and DURATION_MAPPING):
        logging.error("Some mappings are missing or invalid.")
else:
    logging.error(f"Failed to retrieve the mappings. Status code: {response.status_code}")
    raise Exception("Failed to load mappings.")

@api.route('/api/recommend')
class RecommendCourses(Resource):
    @api.expect(recommendation_model, validate=False)
    @api.response(200, 'Rekomendasi berhasil ditemukan.')
    @api.response(400, 'Input tidak valid atau tidak lengkap.')
    @api.response(500, 'Terjadi kesalahan pada server.')
    def post(self):
        """
        Endpoint untuk merekomendasikan kursus berdasarkan input pengguna:
        - Interest: Minat kursus (misalnya: Machine Learning, Data Analysis, dsb.)
        - Course Type: Jenis kursus (misalnya: Course, Specialization, Professional Certificate, atau Project)
        - Duration: Durasi maksimum dalam minggu (misalnya: 4, 8, 12, dst.)
        """
        try:
            # Get JSON input
            data = request.get_json()

            # Validate input
            if not data:
                return {"error": "No input data provided."}, 400

            user_interest = data.get('interest')
            user_course_type = data.get('course_type')
            user_duration = data.get('duration')

            if not user_interest or not user_course_type or not user_duration:
                return {"error": "All fields (interest, course_type, duration) must be filled out."}, 400

            # Convert duration to integer
            try:
                user_duration = int(user_duration)
            except (ValueError, TypeError):
                return {"error": "Duration must be a valid number."}, 400

            # Map inputs to encoded values
            interest_encoded = INTEREST_MAPPING.get(user_interest, None)
            course_type_encoded = COURSE_TYPE_MAPPING.get(user_course_type, None)
            duration_scaled = DURATION_MAPPING.get(user_duration, None)

            if interest_encoded is None or course_type_encoded is None or duration_scaled is None:
                return {"error": "Invalid inputs. Please check your entries."}, 400

            # Create feature array for prediction
            user_features = np.array([[interest_encoded, course_type_encoded, duration_scaled]])
            logging.debug(f"Encoded user features: {user_features}")

            # Model prediction
            predictions = model.predict(user_features)
            predicted_category_encoded = predictions.argmax(axis=-1)[0]
            predicted_category = [key for key, value in CATEGORY_MAPPING.items() if value == predicted_category_encoded][0]

            # Filter and recommend courses
            filtered_courses = course_data[
                (course_data['Category'] == predicted_category) &
                (course_data['Course Type'] == user_course_type) & 
                (course_data['Duration'] <= user_duration)
            ]

            if user_interest:
                filtered_courses = filtered_courses[filtered_courses['Sub-Category'] == user_interest]

            top_courses = filtered_courses.sort_values(by='Duration').head(10)

            recommended_courses = [
                {
                    "title": row["Title"],
                    "short_intro": row["Short Intro"][:200] + "..." if len(row["Short Intro"]) > 200 else row["Short Intro"],
                    "url": row["URL"]
                }
                for _, row in top_courses.iterrows()
            ]

            return {
                "predicted_category": predicted_category,
                "recommended_courses": recommended_courses
            }, 200

        except Exception as e:
            logging.error(f"Error processing request: {e}")
            return {"error": "An error occurred. Please try again."}, 500

api.add_resource(RecommendCourses, '/api/recommend')

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=8080)
