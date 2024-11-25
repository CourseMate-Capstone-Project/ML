# Course Recommendation API

Welcome to the **Course Recommendation API**! This repository provides a powerful API for recommending online courses tailored to users' preferences. It leverages machine learning to help users find the most relevant courses based on their interests, course types, and preferred duration.

---

## Features

- **Personalized Recommendations**: Generate course suggestions based on category, type, and duration preferences.
- **Scalable and Flexible**: Easily extendable to include additional features or integrate with other services.
- **Machine Learning-Powered**: Powered by TensorFlow, delivering accurate and relevant recommendations.
- **User-Friendly API Documentation**: Swagger UI available at `/api/recommend` for quick and easy API interaction.

---

## Directory Structure

The repository is organized as follows:

```plaintext
├── /datasets         # Contains sample datasets for courses (if needed).
├── /icons            # Placeholder for application icons or related assets.
├── /mappings         # Encoded mappings for categorical data used in the recommendation model.
├── /models           # Pre-trained machine learning models.
├── /notebooks        # Jupyter Notebooks for data preprocessing and model training.
├── .gitignore        # Files and directories to be excluded from version control.
├── Dockerfile        # Docker configuration for containerizing the application.
├── app.py            # Main Flask application and API logic.
├── index.html        # Landing page for testing the API endpoints.
├── requirements.txt  # Dependencies required to run the application.
```

---

## API Endpoints

### **Base URL**: `/api`

### **Recommendations**
- **Endpoint**: `/api/recommend`  
- **Method**: `POST`  
- **Description**: Generate personalized course recommendations based on user preferences.  

#### Request Body
```json
{
  "interest": "Machine Learning",
  "course_type": "Course",
  "duration": "8"
}
```

| Field        | Type   | Required | Description                                                                                  |
|--------------|--------|----------|----------------------------------------------------------------------------------------------|
| interest     | string | Yes      | Desired interest related to the course (e.g., "Machine Learning", "Data Analysis").          |
| course_type  | string | Yes      | Type of course (e.g., "Course", "Specialization", "Professional Certificate", "Project").    |
| duration     | string | Yes      | Maximum duration for the course in weeks (e.g., "4", "8", "12").                             |

#### Response Example
```json
{
  "predicted_category": "Data Science",
  "recommended_courses": [
    {
      "title": "Introduction to Machine Learning",
      "short_intro": "Learn the fundamentals of machine learning and build your first predictive models...",
      "url": "https://example.com/course/ml-intro"
    },
    {
      "title": "Advanced AI Techniques",
      "short_intro": "Explore advanced AI topics like deep learning, reinforcement learning, and more...",
      "url": "https://example.com/course/ai-advanced"
    }
  ]
}
```

---

## Setup Instructions

### Prerequisites
- Python 3.8 or higher
- Docker (optional, for containerized deployment)

### Installation
1. Clone this repository:
   ```bash
   git clone (https://github.com/CourseMate-Capstone-Project/ML)](https://github.com/CourseMate-Capstone-Project/ML)
   cd ML
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Flask app:
   ```bash
   python app.py
   ```
4. Access the API at `http://localhost:8080/api/recommend`.

---

## Deployment with Docker
1. Build the Docker image:
   ```bash
   docker build -t course-recommendation-api .
   ```
2. Run the container:
   ```bash
   docker run -p 8080:8080 course-recommendation-api
   ```
3. The API will be available at `http://localhost:8080/api/recommend`.

---

## Technologies Used

- **Backend Framework**: Flask
- **Machine Learning**: TensorFlow
- **Data Processing**: Pandas, NumPy
- **Containerization**: Docker
- **API Documentation**: Swagger (via Flask-RESTX)

---

## Future Enhancements

- Integrate user authentication for personalized recommendations.
- Add support for multiple languages.
- Include additional filters like course ratings and difficulty levels.

---

## Contributing

We welcome contributions! Feel free to fork the repository, make changes, and submit a pull request. For major changes, please open an issue first to discuss your ideas.

---

## Authors
 
This API was developed by **Bagus Angkasawan Sumantri Putra** as part of the Bangkit 2024 Capstone Project **C242-PR594**, in collaboration with a multidisciplinary team from Bangkit cohorts specializing in Machine Learning (ML), Cloud Computing (CC), and Mobile Development (MD).

---

Thank you for exploring the Course Recommendation API! 🚀
