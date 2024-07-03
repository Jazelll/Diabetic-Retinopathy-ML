from django import forms

class PredictForm(forms.Form):
    # floater = forms.RadioSelect(choices=['yes', 'no'], label="", max_length=1)
    subject = forms.CharField(max_length=100)