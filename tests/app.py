import numpy as np
import tensorflow as tf
import pandas as pd
import streamlit as st
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. Unduh Model TFLite dari GitHub
model_url = "https://github.com/CourseMate-Capstone-Project/ML/raw/main/models/course_recommendation_model.tflite"
model_response = requests.get(model_url)
with open("course_recommendation_model.tflite", 'wb') as f:
    f.write(model_response.content)

# 2. Load Model TFLite
interpreter = tf.lite.Interpreter(model_path="course_recommendation_model.tflite")
interpreter.allocate_tensors()

# Fungsi untuk menjalankan prediksi menggunakan TFLite
def predict_with_tflite(model, input_data):
    input_details = model.get_input_details()
    output_details = model.get_output_details()

    # Set input tensor
    input_index = input_details[0]['index']
    model.set_tensor(input_index, input_data)

    # Run inference
    model.invoke()

    # Get output tensor
    output_index = output_details[0]['index']
    output_data = model.get_tensor(output_index)
    return output_data

# 3. Unduh Dataset CSV dari GitHub
dataset_url = "https://github.com/CourseMate-Capstone-Project/ML/raw/main/datasets/cleaned_courses.csv"
dataset_response = requests.get(dataset_url)
with open("filtered_courses.csv", 'wb') as f:
    f.write(dataset_response.content)

# 4. Load Data dan Preprocessing
course = pd.read_csv("filtered_courses.csv")

# Encode 'Sub-Category', 'Category', dan 'Course-Type' menggunakan LabelEncoder
label_encoder_subcategory = LabelEncoder()
label_encoder_category = LabelEncoder()
label_encoder_course_type = LabelEncoder()

course['Sub-Category-Encoded'] = label_encoder_subcategory.fit_transform(course['Sub-Category'])
course['Category-Encoded'] = label_encoder_category.fit_transform(course['Category'])
course['Course-Type-Encoded'] = label_encoder_course_type.fit_transform(course['Course Type'])

# Standarisasi durasi kursus
scaler = StandardScaler()
course['Duration-Scaled'] = scaler.fit_transform(course[['Duration']])

# 5. Input Pengguna
st.title("Course Recommendation Dashboard")

user_subcategory = st.selectbox('Select Sub-Category:', course['Sub-Category'].unique())
user_course_type = st.selectbox('Select Course-Type:', course['Course Type'].unique())
user_duration = st.slider('Select Duration (months):', min_value=1, max_value=12, value=3)

# 6. Preprocessing Input Pengguna
user_subcategory_encoded = label_encoder_subcategory.transform([user_subcategory])[0]
user_course_type_encoded = label_encoder_course_type.transform([user_course_type])[0]
user_duration_scaled = scaler.transform([[user_duration]])[0][0]

# 7. Membuat Input untuk Prediksi TFLite
user_input = np.array([[user_subcategory_encoded, user_course_type_encoded, user_duration_scaled]], dtype=np.float32)

# 8. Prediksi Kategori menggunakan TFLite
predicted_category_encoded = predict_with_tflite(interpreter, user_input)
predicted_category = label_encoder_category.inverse_transform([predicted_category_encoded.argmax()])[0]

# 9. Tampilkan Kategori yang Diprediksi
st.subheader(f"Predicted Category: {predicted_category}")

# 10. Filter Dataset untuk Kursus yang Sesuai
recommended_courses = course[course['Category'] == predicted_category]

# 11. Tampilkan Kursus yang Direkomendasikan
if recommended_courses.empty:
    st.write("No courses found for the predicted category.")
else:
    st.subheader(f"Recommended Courses in {predicted_category} category")
    for index, course in recommended_courses.iterrows():
        title = course['Title']
        short_intro = course['Short Intro']
        url = course['URL']
        duration = course['Duration']

        # Batasi panjang short intro
        if len(short_intro) > 150:
            short_intro = short_intro[:150] + "..."

        st.write(f"**Title**: {title}")
        st.write(f"**Short Intro**: {short_intro}")
        st.write(f"**URL**: [Link]({url})")
        st.write(f"**Duration**: {duration} month(s)")
        st.markdown("---")
