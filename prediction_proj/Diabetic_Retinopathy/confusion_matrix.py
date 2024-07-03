import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
diabetes_data = pd.read_csv('diabetic_retinopathy_data.csv')

# Assuming 'Diabetic_Retinopathy' is the target column and others are features
x = diabetes_data.drop(columns=['Diabetic_Retinopathy'])
y = diabetes_data['Diabetic_Retinopathy']

# Step 2: Handle Imbalanced Data using SMOTE
smote = SMOTE(random_state=0)
x_resampled, y_resampled = smote.fit_resample(x, y)

# Step 3: Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x_resampled, y_resampled, test_size=0.2, random_state=0)

# Step 4: Train the model (Example: Random Forest classifier)
clf = RandomForestClassifier(random_state=0)
clf.fit(x_train, y_train)

# Step 5: Evaluate the model
y_pred = clf.predict(x_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Step 6: Print Confusion Matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))
tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

# Step 7: Print TP, TN
print(f'True Positives: {tp}')
print(f'True Negatives: {tn}')

# Step 8: Print Classification Report
print('Classification Report:')
print(classification_report(y_test, y_pred))

# Visualize Confusion Matrix
plt.figure(figsize=(12, 10))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', annot_kws={'size': 16})

plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - Diabetic Retinopathy')
plt.show()

# Step 9: Save the trained model
model_file = 'DR_prediction_model.joblib'
joblib.dump(clf, model_file)
