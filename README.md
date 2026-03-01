# 🧠 CogniGraph: Prescriptive Intervention Dashboard

**Track:** Problem Statement 1 (Early Detection of Student Burnout & Dropout Risk)  
**Submitted by:** Keertanaa (Reg No: 22MIA1172)

## Overview
Current educational analytics suffer from the "lagging indicator" problem—universities only realize a student is burning out after they miss assignments or fail exams. Furthermore, existing AI solutions are entirely *descriptive* or *predictive*; they tell an advisor a student will fail, but offer no data-backed guidance on how to prevent it. 

**CogniGraph** pivots to a **Prescriptive Analytics** approach. It treats academic burnout not just as an individual failing, but as a socially transmissible phenomenon. By combining **Natural Language Processing (NLP)** with **Graph Theory (NetworkX)**, the system identifies the root causes of burnout via **SHAP (Explainable AI)** and provides an interactive "What-If" simulator for advisors to test interventions before deploying them.

---

## 📊 Dataset Simulation Process (Mandatory Disclosure)

Because real student behavioural data containing LMS tracking, mental health sentiment, and peer-to-peer social network interactions is highly sensitive and protected by privacy laws, this project utilizes a custom-built simulated dataset.

* **Dataset type:** Synthetic
* **Why:** No real behavioural dataset combining NLP forum sentiment, linguistic fatigue, and study-group network graphs is publicly available due to FERPA/GDPR compliance.
* **How you generated it (rules, distributions, assumptions):** * *The Social Graph:* Real-world study groups do not form randomly. We utilized a **Barabási–Albert preferential attachment model** via `NetworkX` to simulate a realistic social graph where certain "hub" students possess high social influence.
  * *Linguistic & Academic Metrics:* Generated using `numpy` probability distributions. Semantic Fatigue follows a Beta distribution, Assignment Delays follow an Exponential distribution (most submit on time, a few trail heavily), and Sentiment follows a Normal distribution.
  * *The Contagion Rule:* A student's base burnout score is dynamically modified by the `peer_contagion_risk`—calculated by averaging the baseline burnout scores of their direct connections (edges) in the social graph. Burnout is modeled as contagious.
* **Number of records:** 2,000 student profiles (`complex_student_data.csv`) and their corresponding network edges (`peer_network.edgelist`).
* **Feature description:**
  1. `semantic_fatigue_score` (0-1): NLP metric representing the simplification of vocabulary and sentence structure in forum posts.
  2. `help_seeking_drop_pct` (0-100): Percentage drop in collaborative or question-asking behaviour.
  3. `forum_sentiment` (-1.0 to 1.0): NLP-derived polarity of student communications.
  4. `lms_logins`: Weekly frequency of portal access.
  5. `assignment_delay_days`: Average days late for submissions.
  6. `peer_contagion_risk`: Network-derived exposure score based on the burnout levels of immediate study-group peers.

---

## 🛠️ Tech Stack
* **Frontend/UI:** Streamlit
* **Machine Learning:** Scikit-Learn (Random Forest Regressor)
* **Explainable AI (XAI):** SHAP (SHapley Additive exPlanations)
* **Graph Analytics:** NetworkX
* **Visualizations:** Plotly Interactive Graphs

---

## 🚀 How to Run Locally
1. Clone this repository.
2. Ensure you have Python 3.9+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Run the data generation and model training pipeline (Optional, pre-trained files are included):
python generate_data.py
python train_model.py

5. Launch the interactive Prescriptive Dashboard:
python -m streamlit run app.py

