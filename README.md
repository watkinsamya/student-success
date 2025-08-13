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

📂 Project Structure
bash
Copy
Edit
student-success/
│
├── models/
│   ├── kaggle_classifier.joblib
│   ├── uci_regressor.joblib
│
├── streamlit_app.py        # Main app file
├── requirements.txt        # Python dependencies
├── README.md               # Project documentation
└── .gitignore              # Ignore unnecessary files
⚙️ Installation & Running
1. Clone the Repository
bash
Copy
Edit
git clone https://github.com/watkinsamya/student-success.git
cd student-success
2. Create Virtual Environment & Install Dependencies
bash
Copy
Edit
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install -r requirements.txt
3. Run the App
bash
Copy
Edit
streamlit run streamlit_app.py
4. Open in Browser
Once running, open:

arduino
Copy
Edit
http://localhost:8501
📊 How It Works
Load Pre-Trained Models — Models are loaded from the models/ folder at startup.

User Input — You enter student data into the UI.

Prediction — The model processes the input and returns either:

Risk Category (Kaggle dataset)

Final Grade G3 (UCI dataset)

Display Results — Output is shown directly in the web app.

💡 Future Improvements
Add personalized study recommendations.

Enable CSV bulk uploads for batch predictions.

Integrate visualization dashboards for school-wide insights.

📜 License
This project is for educational purposes only. Dataset credits:

Kaggle Students Performance Dataset

UCI Student Performance Dataset
