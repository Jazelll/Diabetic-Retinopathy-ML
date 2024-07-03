import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import load
import pyttsx3

# Load the saved model
model_file = 'DR_prediction_model.joblib'
model = load(model_file)

# Define the order of features
features_order = ['Gender','Age', 'Floaters', 'Blurred_Vision', 'Fluctuating_Vision',
                  'Impaired_Color_Vision', 'Empty_Areas', 'Vision_Loss']

# Mapping for categorical variables
label_encoder = LabelEncoder()
binary_categorical_columns = ['Floaters', 'Blurred_Vision', 'Fluctuating_Vision',
                                'Impaired_Color_Vision', 'Empty_Areas', 'Vision_Loss']
def predict_diabetes():
    # Collect user input
    user_data = dict()

    print("==================")
    print("Diabetic Retinopathy Prediction")
    print("Type the number if the symptom is visible:")
    print(" 0 ---> No")
    print(" 1 ---> Yes")
    print("==================")

    # Gender
    gender_input = input("Enter the person's gender (1 for Male, 0 for Female): ")
    user_data['Gender'] = [gender_input]

    # Age
    age_input = input("Enter the person's age: ")
    user_data['Age'] = [age_input]

    for column in binary_categorical_columns:
        user_input = input(f"Enter value for {column.replace('_', ' ')}: ")
        user_data[column] = [user_input]

    # Convert user input to DataFrame
    user_df = pd.DataFrame(user_data)

    # Reorder columns to match the order used during model training
    user_df = user_df[features_order]

    # Convert binary categorical variables to numerical format
    for column in binary_categorical_columns:
        user_df[column] = user_df[column].astype(int)

    # Predict using the model
    prediction = model.predict(user_df)

    # Interpret prediction
    if prediction[0] == 1:
        prediction_text = "Based on the input data, the model predicts that the person may have a diabetic retinopathy."
        print("Prediction: With Diabetic Retinopathy")
    else:
        prediction_text = "Based on the input data, the model predicts that the person may not have a diabetic retinopathy."
        print("Prediction: Without Diabetic Retinopathy")

    # Initialize text-to-speech engine
    engine = pyttsx3.init()
    engine.say(prediction_text)
    engine.runAndWait()

# Run the prediction function
predict_diabetes()