from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import tensorflow as tf
import logging

# Flask app initialization
app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained model
model = tf.keras.models.load_model("./models/course_recommendation_model.keras")

# Load the course dataset
course_data = pd.read_csv("./datasets/cleaned_courses.csv")

# Mapping dictionaries
SUBCATEGORY_MAPPING = {
    "Algorithms": 0, "Animal Health": 1, "Basic Science": 2, "Biology": 3, 
    "Business Essentials": 4, "Business Strategy": 5, "Chemistry": 6, 
    "Cloud Computing": 7, "Computer Security and Networks": 8, 
    "Data Analysis": 9, "Data Management": 10, "Design and Product": 11, 
    "Electrical Engineering": 12, "Entrepreneurship": 13, 
    "Environmental Science and Sustainability": 14, "Finance": 15, 
    "Health Informatics": 16, "Healthcare Management": 17, "History": 18, 
    "Leadership and Management": 19, "Machine Learning": 20, 
    "Marketing": 21, "Mechanical Engineering": 22, 
    "Mobile and Web Development": 23, "Music and Art": 24, 
    "Networking": 25, "Nutrition": 26, "Patient Care": 27, 
    "Personal Development": 28, "Philosophy": 29, 
    "Physics and Astronomy": 30, "Probability and Statistics": 31, 
    "Psychology": 32, "Public Health": 33, "Research": 34, 
    "Research Methods": 35, "Security": 36, "Software Development": 37, 
    "Support and Operations": 38
}

COURSE_TYPE_MAPPING = {
    "Course": 0, "Professional Certificate": 1, 
    "Project": 2, "Specialization": 3
}

CATEGORY_MAPPING = {
    "Arts and Humanities": 0, "Business": 1, "Computer Science": 2, 
    "Data Science": 3, "Health": 4, "Information Technology": 5, 
    "Personal Development": 6, "Physical Science and Engineering": 7
}

DURATION_MAPPING = {
    1: -1.3773422493113383, 2: -1.2626612463950162, 3: -1.1479802434786939, 
    4: -1.3200017478531771, 5: -1.205320744936855, 6: -0.9186182376460497, 
    7: -1.090639742020533, 8: -1.4346827507694992, 9: -1.0332992405623718, 
    10: -0.9759587391042107, 11: -0.8039372347297276, 12: -0.2305322201481169, 
    13: -0.7465967332715665, 14: -0.0011702143154726126, 15: 0.1708512900590106, 
    16: 0.3428727944334938, 17: -0.28787272160627797, 18: -0.345213223064439, 
    19: 0.4002132958916549, 20: -0.1731917186899558, 21: -0.11585121723179474, 
    22: -0.5745752288970833, 23: -0.4025537245226001, 24: -0.05851071577363368, 
    25: -0.8612777361878886, 26: 0.572234800266138, 27: 0.514894298807977, 
    28: 0.6869158031824601, 29: -0.6892562318134055, 30: 0.22819179151717167, 
    31: -0.6319157303552444, 32: 0.05617028714268846, 33: 0.45755379734981594, 
    34: -0.4598942259807612, 35: -0.5172347274389222, 36: 2.2351093425528092, 
    37: -1.4920232522276604, 38: 1.4896828235967152, 39: 1.5470233250548762, 
    41: 1.1456398148477487, 47: 0.2855322929753327, 49: 0.11351078860084952, 
    52: 2.063087838178326, 53: 1.7190448294293594, 54: 0.6295753017242991, 
    57: 0.8015968060987824, 62: 1.432342322138554, 63: 2.005747336720165, 
    66: 1.2603208177640708
}

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        try:
            # Get user inputs
            user_subcategory = request.form.get('subcategory')
            user_course_type = request.form.get('course_type')
            user_duration = int(request.form.get('duration'))

            # Map user inputs to encoded values
            subcategory_encoded = SUBCATEGORY_MAPPING.get(user_subcategory, None)
            course_type_encoded = COURSE_TYPE_MAPPING.get(user_course_type, None)
            duration_scaled = DURATION_MAPPING.get(user_duration, None)

            if subcategory_encoded is None or course_type_encoded is None or duration_scaled is None:
                return render_template('index.html', error="Invalid inputs. Please check your entries.")

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

            return render_template('index.html', 
                                   predicted_category=predicted_category, 
                                   recommended_courses=recommended_courses)

        except Exception as e:
            logging.error(f"Error processing request: {e}")
            return render_template('index.html', error="An error occurred. Please try again.")
    return render_template('index.html', predicted_category=None, recommended_courses=None)

if __name__ == "__main__":
    app.run(debug=True)
