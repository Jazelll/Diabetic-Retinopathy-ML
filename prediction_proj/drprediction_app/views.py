from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from django.template import loader
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from joblib import load
from .forms import PredictForm


def index(request):
    return render(request, 'layout/base.html')

def home(request):
    return render(request, 'home.html')

def evaluate(request):
    return render(request, 'evaluate.html')


#GET PREDICTION RESULT
# Load the saved model
model_file = 'Diabetic_Retinopathy/DR_prediction_model.joblib'
model = load(model_file)

# Define the order of features
features_order = ['Gender','Age', 'Floaters', 'Blurred_Vision', 'Fluctuating_Vision',
                  'Impaired_Color_Vision', 'Empty_Areas', 'Vision_Loss']

# Mapping for categorical variables
label_encoder = LabelEncoder()
binary_categorical_columns = ['Floaters', 'Blurred_Vision', 'Fluctuating_Vision',
                                'Impaired_Color_Vision', 'Empty_Areas', 'Vision_Loss']

def get_result(request):


    if request.method == 'POST':
        user_data = {}

        # Gender
        gender_input = request.POST.get('Gender')
        user_data['Gender'] = [gender_input]

        # Age
        age_input = request.POST.get('Age')
        user_data['Age'] = [age_input]

        for column in binary_categorical_columns:
            user_input = request.POST.get(column)
            user_data[column] = [user_input]

        # Convert user input to DataFrame
        user_df = pd.DataFrame(user_data)

        # Reorder columns to match the order used during model training
        user_df = user_df[features_order]

        # Convert binary categorical variables to numerical format
        for column in binary_categorical_columns:
            user_df[column] = user_df[column]

        # Predict using the model
        prediction = model.predict(user_df)

        # Interpret prediction
        if prediction[0] == 1:
            prediction_text = "Based on the input data, the model predicts that the person may have a diabetic retinopathy."
            prediction_result = "Prediction: With Diabetic Retinopathy"
        else:
            prediction_text = "Based on the input data, the model predicts that the person may not have a diabetic retinopathy."
            prediction_result = "Prediction: Without Diabetic Retinopathy"

        context = {'prediction': prediction_result, 'prediction_text': prediction_text}
        print(context["prediction"])
        return render(request, 'result.html', {"result": context})

    return render(request, 'result.html')