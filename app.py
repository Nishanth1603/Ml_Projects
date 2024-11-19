from flask import Flask, request, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Route for home page
@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    try:
        if request.method == 'GET':
            return render_template('home.html')
        
        # Capture form data
        gender = request.form.get('gender')
        ethnicity = request.form.get('ethnicity')
        parental_level_of_education = request.form.get('parental_level_of_education')
        lunch = request.form.get('lunch')
        test_preparation_course = request.form.get('test_preparation_course')
        
        # Validate and convert scores
        try:
            reading_score = float(request.form.get('reading_score'))
            writing_score = float(request.form.get('writing_score'))
        except ValueError:
            return render_template('home.html', error="Reading or Writing score must be a valid number.")
        
        # Create CustomData instance
        data = CustomData(
            gender=gender,
            race_ethnicity=ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        
        # Get data as DataFrame
        pred_df = data.get_data_as_data_frame()
        print(f"Data for prediction:\n{pred_df}")
        print("Before Prediction")

        # Make prediction using the pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        print("After Prediction")
        
        # Return the result to the template
        return render_template('home.html', results=results[0])

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        return render_template('home.html', error="An error occurred during prediction. Please try again.")

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True)
