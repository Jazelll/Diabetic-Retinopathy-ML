import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from joblib import dump

# Step 1: Load the dataset
data = pd.read_csv('diabetic_retinopathy_data.csv')

# Step 2: Data Preprocessing
# Convert categorical variables to numerical format
label_encoder = LabelEncoder()

# Convert binary categorical variables (0: No, 1: Yes)
binary_columns = ['Gender','Age', 'Floaters', 'Blurred_Vision', 'Fluctuating_Vision',
                  'Impaired_Color_Vision', 'Empty_Areas', 'Vision_Loss', 'Diabetic_Retinopathy']

# Step 3: Split Data
y = data['Diabetic_Retinopathy']
X = data.drop('Diabetic_Retinopathy', axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 4: Model Selection
model = RandomForestClassifier(random_state=0)

# Step 5: Train
model.fit(X_train, y_train)

# Step 6: Evaluation
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 7: Save the Model
model_file = 'DR_prediction_model.joblib'
dump(model, model_file)
print("Model saved as", model_file)

# Step 8: Prediction (example)
new_data = pd.DataFrame([[0, 33, 0, 0, 0, 0, 0, 0]], columns=X.columns)
prediction = model.predict(new_data)

# Map predicted values to labels
predicted_label = "With Diabetic Retinopathy" if prediction[0] == 1 else "Without Diabetic Retinopathy"
print("Prediction for new data:", predicted_label)