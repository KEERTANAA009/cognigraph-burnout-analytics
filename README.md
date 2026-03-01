# Early Detection of Student Burnout & Dropout Risk

## Overview

[cite_start]This project is a behavioural analytics system designed to identify academic disengagement and predict student burnout and dropout risk using early behavioural signals[cite: 4, 6, 8].

## Dataset Simulation Process

[cite_start]Because real student behavioural data containing LMS tracking, attendance, and sentiment is highly sensitive and protected by privacy laws, this project utilizes simulated data[cite: 78, 80].

- [cite_start]**Dataset type:** Synthetic [cite: 79]
- [cite_start]**Why:** No real behavioural dataset combining LMS telemetry, sentiment analysis, and attendance trends is publicly available[cite: 80].
- [cite_start]**How you generated it (rules, distributions, assumptions):** [cite: 81]
  - Data was generated using Python's `numpy` library.
  - [cite_start]_Assumptions:_ Students with high burnout exhibit lower LMS login frequencies (Poisson distribution with lower lambda), higher assignment delays (Exponential distribution), lower attendance (Normal distribution skewed lower), and negative sentiment scores[cite: 6, 15, 16, 17, 18].
  - [cite_start]_Risk Score Logic:_ A weighted combination of these features, plus random noise, generates a continuous Dropout Probability (0-100), which is then binned into Low, Medium, and High Burnout Risk Levels[cite: 10, 12, 22].
- [cite_start]**Number of records:** 5,000 simulated student profiles[cite: 82].
- [cite_start]**Feature description:** [cite: 83]
  - `lms_login_frequency`: Weekly logins to the learning portal.
  - `assignment_delay_days`: Average days late for submissions.
  - `attendance_percentage`: Overall class attendance.
  - `sentiment_score`: NLP-derived score from feedback (-1.0 to 1.0).
  - `activity_irregularity_score`: Measure of erratic platform usage (0 to 10).

## Model Selection & Evaluation

We utilized a **Random Forest Regressor** to predict the Dropout Risk Score. [cite_start]Random Forest was chosen because it naturally handles non-linear relationships in behavioural data and allows us to easily extract **Feature Importances** to identify the key behavioural triggers for each specific student[cite: 22, 108].
