🎓 Student Grades & Academic Success Predictor
📌 Overview
The Student Success Predictor is a Streamlit-based web application designed to help educators, students, and academic advisors predict academic risk levels and final grades using machine learning models.
It uses two datasets:

Kaggle StudentsPerformance (classification: At-Risk / Average / High)

UCI Student Performance (regression: Predict final grade G3 on a 0–20 scale)

This tool provides early alerts to identify at-risk students and offers data-driven insights to improve student success.

🚀 Features
Risk Category Prediction: Classifies students into performance categories using demographic and exam data.

Final Grade Prediction: Predicts a student’s final G3 grade based on prior grades, absences, and lifestyle factors.

Interactive UI: Simple, user-friendly interface powered by Streamlit.

Pre-trained Models: Models are already trained and saved in the models/ folder for instant use.

🧠 Machine Learning Models
Two models are included:

Kaggle Classifier — Uses a pipeline with preprocessing and a classification algorithm to predict risk category.

UCI Regressor — Uses Gradient Boosting (best performer) to predict final G3 grade.

Performance Metrics (lower MAE/RMSE = better, higher R² = better):

Model	MAE	RMSE	R²
Gradient Boosting	1.02	1.69	0.855
Decision Tree	1.03	1.73	0.848
KNN	1.16	1.78	0.840
Linear Regression	1.21	1.94	0.816



💡 Future Improvements
Add personalized study recommendations.

Enable CSV bulk uploads for batch predictions.

Integrate visualization dashboards for school-wide insights.

📜 License
This project is for educational purposes only. Dataset credits:

Kaggle Students Performance Dataset

UCI Student Performance Dataset
