# IPL Win Predictor 🏏

An end-to-end Machine Learning web application that predicts the real-time winning probability of the chasing team in an Indian Premier League (IPL) match. Built using Python, Scikit-Learn, and Streamlit.

---

## 📌 Project Overview
Cricket matches change momentum with every single delivery. This project implements a classification model using historical IPL match data to evaluate the dynamic live situation of a match (runs required, balls remaining, wickets in hand, run rates) and predict the final outcome.

---

## 🛠️ Tech Stack
* **Language:** Python
* **Data Manipulation:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (Logistic Regression, ColumnTransformer, OneHotEncoder, Pipelines)
* **Web Framework:** Streamlit
* **Model Persistence:** Pickle

---

## 📊 Features & Data Pipeline

### 1. Data Cleaning & Processing
* Aggregates ball-by-ball data (`deliveries.csv.zip`) to calculate the target score from the first innings.
* Merges first-innings data with overall match data (`matches.csv`).
* Synchronizes legacy team names (e.g., *Delhi Daredevils* ➡️ *Delhi Capitals*, *Deccan Chargers* ➡️ *Sunrisers Hyderabad*).
* Filters and restructures dataset to focus strictly on ongoing IPL teams and regular matches (removing Duck-Worth Lewis affected matches).

### 2. Feature Engineering
The model is trained on a secondary innings chase configuration using these engineered features:
* **`runs_left`**: The remaining runs required to reach the target.
* **`balls_left`**: Deliveries remaining out of 120 balls.
* **`wickets`**: Remaining wickets available for the chasing team.
* **`curr_runrate`**: Live Current Run Rate (CRR).
* **`req_runrate`**: Live Required Run Rate (RRR).

### 3. Machine Learning Pipeline
* Categorical features (`batting_team`, `bowling_team`, `city`) are preprocessed via a `OneHotEncoder` within a `ColumnTransformer`.
* **Model Selected:** `LogisticRegression` (with `liblinear` solver). Logistic Regression is intentionally utilized over tree-based models to ensure smooth, continuous, and highly interpretable probability shifts ball-by-ball.

---

## 🚀 How to Run the Project

### Prerequisites
Ensure you have the dataset files available in your project directory:
* `deliveries.csv.zip`
* `matches.csv`

### Installation & Setup

1. **Install required dependencies:**
   ```bash
   pip install pandas numpy scikit-learn streamlit
   ```

2. **Train and Save the Model:**
   Run your training script to clean the data, generate the model pipeline, and save it as a serialized file:
   ```bash
   python train.py
   ```
   *This creates a `pipe.pkl` file in your workspace.*

3. **Launch the Web Application:**
   Run the Streamlit application file (e.g., `app.py`):
   ```bash
   streamlit run app.py
   ```

---

## 💻 Web App Usage Guide
Once the Streamlit interface loads in your browser, enter the current match data:
1. Select the **Batting Team** (Chasing team) and **Bowling Team**.
2. Choose the host **City**.
3. Input the **Target Score** set in the first innings.
4. Input the chasing team's live **Current Score**, current **Overs** completed, and **Wickets** lost.
5. Click **Predict Probability** to see the live calculated winning percentage for both sides.
