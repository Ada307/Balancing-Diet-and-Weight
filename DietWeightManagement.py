import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, mean_squared_error, r2_score
import os
import joblib

# Step 0: Verify file existence and working directory
file_path = "C:\\Users\\adaj0\\OneDrive\\Desktop\\Balancing Diet & Weight (Responses).xlsx"

if not os.path.exists(file_path):
    raise FileNotFoundError(
        f"The file '{file_path}' was not found in the current working directory ({os.getcwd()}). "
        "Please ensure the file is in the correct directory or specify the full path. "
        "If running in Google Colab, upload the file using: from google.colab import files; uploaded = files.upload()"
    )

# Step 1: Load the dataset
try:
    data = pd.read_excel(file_path)
except Exception as e:
    raise Exception(f"Error loading the Excel file: {str(e)}. Please check the file format and path.")

# Step 2: Data Cleaning
# Drop irrelevant columns
data = data.drop(columns=["Timestamp", "Name (optional)"], errors="ignore")

# Handle missing values
data = data.dropna(subset=["Current Weight (in kg)", "Height (in cm)", "What is your current goal?"])  # Critical columns
for col in data.columns:
    if data[col].dtype == "object":
        data[col] = data[col].fillna(data[col].mode()[0])  # Fill categorical with mode
    else:
        data[col] = data[col].fillna(data[col].median())  # Fill numerical with median

# Correct erroneous entries
data["How many hours of sleep do you get daily?"] = data["How many hours of sleep do you get daily?"].replace(
    ["45816", "45818", "45753"], "6-8"
)
data["How many times per week do you exercise?"] = data["How many times per week do you exercise?"].replace(
    ["45659", "45720", "45783", "45660"], "3-4"
)
data["Daily sedentary hours (sitting/desk time in hours)"] = data["Daily sedentary hours (sitting/desk time in hours)"].replace(
    ["45784", "45721", "45847"], "5-7"
)

# Standardize case in 'Have you seen any significant weight changes in the last 6–12 months?'
data["Have you seen any significant weight changes in the last 6–12 months?"] = data[
    "Have you seen any significant weight changes in the last 6–12 months?"
].str.title()  # Convert to title case (e.g., 'yes' -> 'Yes', 'no' -> 'No')

# Convert ranges to numerical values
def convert_sleep_hours(sleep):
    if sleep == "2-4":
        return 3
    elif sleep == "4-6":
        return 5
    elif sleep == "6-8":
        return 7
    elif sleep == "8+":
        return 9
    return float(sleep) if isinstance(sleep, str) and sleep.isdigit() else 7

def convert_sedentary_hours(hours):
    if hours == "1-3":
        return 2
    elif hours == "3-5":
        return 4
    elif hours == "5-7":
        return 6
    elif hours == "7-9":
        return 8
    elif hours == "9+":
        return 10
    return float(hours) if isinstance(hours, str) and hours.isdigit() else 6

def convert_steps(steps):
    if steps == "<1000":
        return 500
    elif steps == "1000-3000":
        return 2000
    elif steps == "3000-5000":
        return 4000
    elif steps == "5000+":
        return 6000
    return float(steps) if isinstance(steps, str) and steps.isdigit() else 4000

def convert_meals_per_day(meals):
    if meals == "2":
        return 2
    elif meals == "3":
        return 3
    elif meals == "4":
        return 4
    elif meals == "4+":
        return 4.5
    return float(meals) if isinstance(meals, str) and meals.isdigit() else 3

def convert_water_intake(water):
    if water == "<1":
        return 0.5
    elif water == "1–2":
        return 1.5
    elif water == "2–3":
        return 2.5
    elif water == "3–4":
        return 3.5
    elif water == ">4":
        return 4.5
    elif water == "Not sure":
        return 2  # Default to average intake
    return float(water) if isinstance(water, str) and water.replace(".", "").isdigit() else 2

data["Sleep Hours"] = data["How many hours of sleep do you get daily?"].apply(convert_sleep_hours)
data["Sedentary Hours"] = data["Daily sedentary hours (sitting/desk time in hours)"].apply(convert_sedentary_hours)
data["Daily Steps"] = data["Average daily steps (if known)"].apply(convert_steps)
data["Meals per Day"] = data["How many meals do you have per day?"].apply(convert_meals_per_day)
data["Water Intake (liters)"] = data["Daily water intake (in liters)"].apply(convert_water_intake)

# Step 3: Meal Standardization and Categorization
# Synonym dictionary for meal terms
meal_synonyms = {
    "daal": "dal", "chhapati": "roti", "chapati": "roti", "paratha": "roti", "parantha": "roti",
    "naan": "roti", "fulka": "roti", "phulka": "roti", "sbji": "sabzi", "sbzi": "sabzi",
    "sabji": "sabzi", "chaval": "chawal", "daliyaa": "daliya", "khichdi": "daliya"
}

# Standardize meal descriptions
def standardize_meal(meal):
    if pd.isna(meal) or meal.lower() in ["none", "na", "skipped"]:
        return "Skipped"
    meal = meal.lower()
    for synonym, standard in meal_synonyms.items():
        meal = meal.replace(synonym, standard)
    if meal in ["homemade", "indian meal", "meal", "variety", "variety meal", "depends on mood"]:
        return "Balanced"  # Default for vague entries
    return meal

data["Typical breakfast"] = data["Typical breakfast"].apply(standardize_meal)
data["Typical lunch"] = data["Typical lunch"].apply(standardize_meal)
data["Typical dinner"] = data["Typical dinner"].apply(standardize_meal)

# Updated meal categorization
def categorize_meal(meal):
    if meal == "Skipped":
        return "Skipped"
    meal = meal.lower()
    if any(x in meal for x in ["roti", "rice", "chawal", "bread", "poha", "cornflakes", "jalebi", "oats", "toast", "upma", "idli", "sambhar", "puri"]):
        return "Carb-heavy"
    elif any(x in meal for x in ["egg", "chicken", "paneer", "protein", "fish", "salad", "grilled"]):
        return "Protein-rich"
    elif any(x in meal for x in ["dal", "lentils", "sabzi", "vegetable", "daliya", "khichdi", "fruits", "curry", "stew", "soup", "quinoa"]):
        return "Balanced"
    return "Other"

data["Breakfast Type"] = data["Typical breakfast"].apply(categorize_meal)
data["Lunch Type"] = data["Typical lunch"].apply(categorize_meal)
data["Dinner Type"] = data["Typical dinner"].apply(categorize_meal)

# Step 4: Feature Engineering
# Calculate BMI
data["BMI"] = data["Current Weight (in kg)"] / ((data["Height (in cm)"] / 100) ** 2)

# Encode exercise intensity
def encode_exercise_intensity(exercise):
    low = ["Walking", "Yoga", "Home Workouts"]
    moderate = ["Jogging", "Running", "Cycling"]
    high = ["Gym (Strength/Cardio)", "Swimming"]
    exercises = exercise.split(", ") if isinstance(exercise, str) else [exercise]
    intensity = 0
    for ex in exercises:
        if ex in low:
            intensity = max(intensity, 1)
        elif ex in moderate:
            intensity = max(intensity, 2)
        elif ex in high:
            intensity = max(intensity, 3)
    return intensity

data["Exercise Intensity"] = data["Type of exercise you usually do (choose all that apply)"].apply(encode_exercise_intensity)

# Create dedicated LabelEncoders
le_success = LabelEncoder()
success_col = "Have you seen any significant weight changes in the last 6–12 months?"
data[success_col] = le_success.fit_transform(data[success_col].astype(str))
#print(f"Encoding for '{success_col}': {dict(zip(le_success.classes_, range(len(le_success.classes_))))}")

le_goal = LabelEncoder()
goal_col = "What is your current goal?"
data[goal_col] = le_goal.fit_transform(data[goal_col].astype(str))
#print(f"Encoding for '{goal_col}': {dict(zip(le_goal.classes_, range(len(le_goal.classes_))))}")

# Validate expected values in 'What is your current goal?'
if "Weight Loss" not in le_goal.classes_ or "Weight Gain" not in le_goal.classes_:
    raise ValueError(f"Expected values 'Weight Loss' and 'Weight Gain' not found in '{goal_col}'. Unique values: {le_goal.classes_}")

# Encode categorical variables
le = LabelEncoder()
categorical_cols = [
    "Gender", "Current Academic or Work Status", "Do you have any of these medical conditions?",
    "Are you on any long-term medication?", "Stress Level", "Do you consume alcohol or smoke?",
    "Do you snack between meals?", "Type of diet", "How often do you eat processed/junk food?",
    "What is your current goal?", "Have you seen any significant weight changes in the last 6–12 months?",
    "Breakfast Type", "Lunch Type", "Dinner Type"
]

# Validate that all categorical columns exist
for col in categorical_cols:
    if col not in data.columns:
        raise ValueError(f"Column '{col}' not found in dataset. Available columns: {data.columns.tolist()}")

for col in categorical_cols:
    data[col] = le.fit_transform(data[col].astype(str))

# Convert exercise frequency to numerical
def convert_exercise_freq(freq):
    if freq == "0":
        return 0
    elif freq == "1-2":
        return 1.5
    elif freq == "3-4":
        return 3.5
    elif freq == "5-6":
        return 5.5
    elif freq == "6+":
        return 7
    return float(freq) if isinstance(freq, str) and freq.isdigit() else 3.5

data["Exercise Frequency"] = data["How many times per week do you exercise?"].apply(convert_exercise_freq)

# Step 5: Clustering for Pattern Identification
features_for_clustering = [
    "BMI", "Exercise Frequency", "Exercise Intensity", "Daily Steps", "Sedentary Hours",
    "Sleep Hours", "Meals per Day", "Breakfast Type", "Lunch Type", "Dinner Type",
    "Water Intake (liters)", "How often do you eat processed/junk food?"
]
X_cluster = data[features_for_clustering]
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)

kmeans = KMeans(n_clusters=4, random_state=42)
data["Cluster"] = kmeans.fit_predict(X_cluster_scaled)

# Step 6: Prepare Data for Prediction
# Target for classification: Weight change success
yes_encoded = le_success.transform(["Yes"])[0]
maybe_encoded = le_success.transform(["Maybe"])[0]
success_col = "Have you seen any significant weight changes in the last 6–12 months?"
if success_col not in data.columns:
    raise ValueError(f"Column '{success_col}' not found in dataset. Available columns: {data.columns.tolist()}")
data["Success"] = data[success_col].apply(
    lambda x: 1 if x == yes_encoded or x == maybe_encoded else 0
)

# Target for regression: Estimated days to goal
def estimate_days_to_goal(row):
    goal = row["What is your current goal?"]
    weight = row["Current Weight (in kg)"]
    weight_loss_encoded = le_goal.transform(["Weight Loss"])[0]
    weight_gain_encoded = le_goal.transform(["Weight Gain"])[0]
    if goal == weight_loss_encoded:
        return (weight * 0.05) / 0.1 * 7  # Assume 5% weight loss at 0.1 kg/week
    elif goal == weight_gain_encoded:
        return (weight * 0.05) / 0.05 * 7  # Assume 5% weight gain at 0.05 kg/week
    else:
        return 30  # Default for maintenance or general fitness
data["Days to Goal"] = data.apply(estimate_days_to_goal, axis=1)

# Features for prediction
features = [
    "Age", "Gender", "Height (in cm)", "Current Weight (in kg)", "BMI", "Exercise Frequency",
    "Exercise Intensity", "Daily Steps", "Sedentary Hours", "Sleep Hours",
    "Meals per Day", "Do you snack between meals?", "Type of diet",
    "Breakfast Type", "Lunch Type", "Dinner Type", "Water Intake (liters)",
    "How often do you eat processed/junk food?", "Stress Level", "Do you consume alcohol or smoke?",
    "Do you have any of these medical conditions?", "Are you on any long-term medication?"
]
X = data[features]
y_success = data["Success"]
y_days = data["Days to Goal"]

# Split data
X_train, X_test, y_success_train, y_success_test = train_test_split(
    X, y_success, test_size=0.2, random_state=42
)
X_train, X_test, y_days_train, y_days_test = train_test_split(
    X, y_days, test_size=0.2, random_state=42
)

# Scale features for prediction
scaler_predict = StandardScaler()
X_train_scaled = scaler_predict.fit_transform(X_train)
X_test_scaled = scaler_predict.transform(X_test)

# Step 7: Train Models
# Classification model for success prediction
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_scaled, y_success_train)

# Regression model for days to goal
reg = RandomForestRegressor(n_estimators=100, random_state=42)
reg.fit(X_train_scaled, y_days_train)

# Step 8: Evaluate Models
# Classification evaluation
y_success_pred = clf.predict(X_test_scaled)
print("Classification Report for Success Prediction:")
print(classification_report(y_success_test, y_success_pred))

# Regression evaluation
y_days_pred = reg.predict(X_test_scaled)
print("Regression Metrics for Days to Goal Prediction:")
print(f"Mean Squared Error: {mean_squared_error(y_days_test, y_days_pred):.2f}")
print(f"R² Score: {r2_score(y_days_test, y_days_pred):.2f}")
# Evaluate regression model on training data
y_days_train_pred = reg.predict(X_train_scaled)
print("\nRegression Metrics for Days to Goal Prediction (Training Set):")
print(f"Mean Squared Error (Train): {mean_squared_error(y_days_train, y_days_train_pred):.2f}")
print(f"R² Score (Train): {r2_score(y_days_train, y_days_train_pred):.2f})")

# Step 9: Recommendation System
def recommend_diet_exercise(row, cluster_centers, scaler_cluster):
    # Ensure the input has the correct features
    cluster_input = row[features_for_clustering].values.reshape(1, -1)
    cluster_input_df = pd.DataFrame(cluster_input, columns=features_for_clustering)
    #print(f"Input shape to scaler: {cluster_input_df.shape}")
    print(f"\nExpected features for clustering: {features_for_clustering}")
    if cluster_input_df.shape[1] != len(features_for_clustering):
        raise ValueError(
            f"Input has {cluster_input_df.shape[1]} features, but expected {len(features_for_clustering)}. "
            f"Features provided: {cluster_input_df.columns.tolist()}"
        )
    cluster = kmeans.predict(scaler_cluster.transform(cluster_input_df))[0]
    similar_profiles = data[data["Cluster"] == cluster]
    successful_profiles = similar_profiles[similar_profiles["Success"] == 1]

    # Diet recommendation
    common_breakfast = successful_profiles["Breakfast Type"].mode()[0] if not successful_profiles.empty else data["Breakfast Type"].mode()[0]
    common_lunch = successful_profiles["Lunch Type"].mode()[0] if not successful_profiles.empty else data["Lunch Type"].mode()[0]
    common_dinner = successful_profiles["Dinner Type"].mode()[0] if not successful_profiles.empty else data["Dinner Type"].mode()[0]
    diet_rec = f"Breakfast: {le.inverse_transform([common_breakfast])[0]}, Lunch: {le.inverse_transform([common_lunch])[0]}, Dinner: {le.inverse_transform([common_dinner])[0]}"

    # Exercise recommendation
    common_exercise = successful_profiles["Type of exercise you usually do (choose all that apply)"].mode()[0] if not successful_profiles.empty else data["Type of exercise you usually do (choose all that apply)"].mode()[0]
    common_freq = successful_profiles["Exercise Frequency"].mean() if not successful_profiles.empty else data["Exercise Frequency"].mean()
    exercise_rec = f"Exercise: {common_exercise}, Frequency: {common_freq:.1f} times/week"

    return diet_rec, exercise_rec

# Example: Predict and recommend for a test instance
test_instance = X_test.iloc[0]
success_prob = clf.predict_proba([X_test_scaled[0]])[0][1]
days_pred = reg.predict([X_test_scaled[0]])[0]
diet_rec, exercise_rec = recommend_diet_exercise(test_instance, kmeans.cluster_centers_, scaler_cluster)

print(f"\nExample Prediction:")
print(f"Probability of Success: {success_prob:.2f}")
print(f"Estimated Days to Goal: {days_pred:.0f}")
print(f"Diet Recommendation: {diet_rec}")
print(f"Exercise Recommendation: {exercise_rec}")

# Save the model (optional)
joblib.dump(clf, "success_classifier.pkl")
joblib.dump(reg, "days_regressor.pkl")
joblib.dump(scaler_cluster, "scaler_cluster.pkl")
joblib.dump(scaler_predict, "scaler_predict.pkl")
joblib.dump(kmeans, "kmeans.pkl")

print("\n--- Predict for a New User Based on Your Dataset ---")

# Collect height and weight first to compute BMI
height_cm = float(input("Enter your height in cm: "))
weight_kg = float(input("Enter your current weight in kg: "))
height_m = height_cm / 100
bmi = weight_kg / (height_m ** 2)

# User input collection
user_input = {
    "Age": int(input("Enter your age: ")),
    "Gender": int(input("Gender (0 = Male, 1 = Female): ")),
    "Height (in cm)": height_cm,
    "Current Weight (in kg)": weight_kg,
    "BMI": round(bmi, 2),  # Calculated
    "Exercise Frequency": int(input("How many times do you exercise per week?: ")),
    "Exercise Intensity": int(input("Exercise intensity (1 = Low, 2 = Moderate, 3 = High): ")),
    "Daily Steps": int(input("Average daily steps (enter 0 if unknown): ")),
    "Sedentary Hours": float(input("Sedentary hours per day: ")),
    "Sleep Hours": float(input("Average hours of sleep per day: ")),
    "Meals per Day": int(input("How many meals do you have per day?: ")),
    "Do you snack between meals?": int(input("Do you snack between meals? (0 = No, 1 = Yes): ")),
    "Type of diet": int(input("Type of diet (0 = Veg, 1 = Non-Veg, 2 = Vegan, etc.): ")),
    "Breakfast Type": int(input("Breakfast type (0 = Skipped, 1 = Light, 2 = Heavy): ")),
    "Lunch Type": int(input("Lunch type (0 = Skipped, 1 = Light, 2 = Heavy): ")),
    "Dinner Type": int(input("Dinner type (0 = Skipped, 1 = Light, 2 = Heavy): ")),
    "Water Intake (liters)": float(input("Daily water intake in liters: ")),
    "How often do you eat processed/junk food?": int(input("Junk food frequency (0 = Never, 1 = Sometimes, 2 = Often): ")),
    "Stress Level": int(input("Stress level (1 = Low, 2 = Moderate, 3 = High): ")),
    "Do you consume alcohol or smoke?": int(input("Do you smoke or drink alcohol? (0 = No, 1 = Yes): ")),
    "Do you have any of these medical conditions?": int(input("Any medical conditions? (0 = No, 1 = Yes): ")),
    "Are you on any long-term medication?": int(input("Long-term medication? (0 = No, 1 = Yes): "))
}

# Define the required features for clustering
clustering_features = [
    "BMI", "Exercise Frequency", "Exercise Intensity", "Daily Steps", "Sedentary Hours", 
    "Sleep Hours", "Meals per Day", "Breakfast Type", "Lunch Type", "Dinner Type", 
    "Water Intake (liters)", "How often do you eat processed/junk food?"
]

# Filter the user input to only include the necessary features for clustering
user_input_for_clustering = {key: user_input[key] for key in clustering_features}

# Convert to DataFrame for prediction
user_df = pd.DataFrame([user_input], columns=features)

# Step 2: Scale the user input for prediction
user_df_scaled = scaler_predict.transform(user_df)

# Step 3: Predict success probability and days to goal
success_prob = clf.predict_proba(user_df_scaled)[0][1]  # Probability of class 1 (success)
days_pred = reg.predict(user_df_scaled)[0]

# --- Diet & Exercise Recommendation Based on Cluster ---
# Now scale only the necessary features for clustering
user_cluster_df = pd.DataFrame([user_input_for_clustering], columns=clustering_features)
user_cluster_scaled = scaler_cluster.transform(user_cluster_df)

# Generate diet and exercise recommendation
diet_rec, exercise_rec = recommend_diet_exercise(user_cluster_df.iloc[0], kmeans.cluster_centers_, scaler_cluster)

# --- Final User Predictions ---
print("\n--- Personalized Prediction Results ---")
print(f"Probability of Success: {success_prob:.2f}")
print(f"Estimated Days to Goal: {days_pred:.0f}")
print(f"Diet Recommendation: {diet_rec}")
print(f"Exercise Recommendation: {exercise_rec}")
