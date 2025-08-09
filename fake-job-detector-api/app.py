# Step 7: Flask API for Fake Job Detector
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re
import scipy.sparse as sp
import numpy as np
from datetime import datetime
import os

# Initialize Flask app
app = Flask(__name__)
CORS(app, origins=[
    "http://localhost:3000",  # Local development
    "https://your-actual-frontend-url.onrender.com",  # Replace with your actual frontend URL
])

# Global variables for model components
model_pipeline = None
svm_model = None
tfidf = None
scaler = None
numerical_features = None

# Load model on startup
def load_model():
    """Load the trained model pipeline"""
    global model_pipeline, svm_model, tfidf, scaler, numerical_features
    
    try:
        # Load the model pipeline
        model_path = 'fake_job_detector_v1.pkl'
        model_pipeline = joblib.load(model_path)
        
        # Extract components
        svm_model = model_pipeline['svm_model']
        tfidf = model_pipeline['tfidf_vectorizer']
        scaler = model_pipeline['scaler']
        numerical_features = model_pipeline['numerical_features']
        
        print("✅ Model loaded successfully!")
        print(f"📊 Model trained on: {model_pipeline['training_date']}")
        print(f"🎯 Model accuracy: {model_pipeline['performance_metrics']['accuracy']:.2%}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        return False

# Text cleaning function (same as training)
def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == '':
        return ''
    
    # Convert to string and lowercase
    text = str(text).lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
    # Remove email addresses
    text = re.sub(r'\S+@\S+', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', ' ', text)
    
    return text.strip()

def create_features(title, description, company_profile='', requirements='', benefits=''):
    """Create features for prediction (same process as training)"""
    
    # Clean text inputs
    title = clean_text(title)
    description = clean_text(description)
    company_profile = clean_text(company_profile)
    requirements = clean_text(requirements)
    benefits = clean_text(benefits)
    
    # Combine text features (same weighting as training)
    combined_text = f"{title} {title} {title} {description} {description} {requirements} {company_profile} {benefits}".strip()
    
    # Create numerical features
    features_dict = {
        'title_length': len(title),
        'description_length': len(description),
        'requirements_length': len(requirements),
        'combined_length': len(combined_text),
        'title_word_count': len(title.split()) if title else 0,
        'description_word_count': len(description.split()) if description else 0,
        'combined_word_count': len(combined_text.split()) if combined_text else 0,
        'has_company_profile': 1 if company_profile else 0,
        'has_requirements': 1 if requirements else 0,
        'has_benefits': 1 if benefits else 0,
        'exclamation_count': combined_text.count('!'),
        'capital_ratio': sum(1 for c in combined_text if c.isupper()) / len(combined_text) if len(combined_text) > 0 else 0
    }
    
    return combined_text, features_dict

@app.route('/')
def home():
    """API home endpoint"""
    return jsonify({
        'message': 'Fake Job Detector API',
        'version': model_pipeline['model_version'] if model_pipeline else 'Model not loaded',
        'status': 'ready' if model_pipeline else 'error',
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'model_info': '/model-info (GET)'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model_pipeline else 'unhealthy',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_pipeline is not None
    })

@app.route('/model-info')
def model_info():
    """Get model information"""
    if not model_pipeline:
        return jsonify({'error': 'Model not loaded'}), 500
    
    return jsonify({
        'model_version': model_pipeline['model_version'],
        'training_date': model_pipeline['training_date'],
        'performance_metrics': model_pipeline['performance_metrics'],
        'training_info': model_pipeline['training_info']
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if model is loaded
        if not model_pipeline:
            return jsonify({'error': 'Model not loaded'}), 500
        
        # Get JSON data
        data = request.get_json()
        
        # Validate required fields
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        if 'title' not in data or 'description' not in data:
            return jsonify({'error': 'Title and description are required'}), 400
        
        # Extract job posting data
        title = data.get('title', '')
        description = data.get('description', '')
        company_profile = data.get('company_profile', '')
        requirements = data.get('requirements', '')
        benefits = data.get('benefits', '')
        
        # Validate inputs
        if not title.strip() or not description.strip():
            return jsonify({'error': 'Title and description cannot be empty'}), 400
        
        # Create features
        combined_text, features_dict = create_features(
            title, description, company_profile, requirements, benefits
        )
        
        # Transform text features
        X_text = tfidf.transform([combined_text])
        
        # Transform numerical features
        X_numerical = np.array([features_dict[col] for col in numerical_features]).reshape(1, -1)
        X_num_scaled = scaler.transform(X_numerical)
        
        # Combine features
        X_combined = sp.hstack([X_text, sp.csr_matrix(X_num_scaled)])
        
        # Make predictions
        prediction = svm_model.predict(X_combined)[0]
        probability = svm_model.predict_proba(X_combined)[0, 1]
        
        # Determine risk level and confidence
        if probability < 0.25:
            risk_level = "LOW"
            confidence = "High confidence - Likely legitimate"
            color = "green"
        elif probability < 0.6:
            risk_level = "MEDIUM"
            confidence = "Moderate confidence - Review carefully"
            color = "yellow"
        else:
            risk_level = "HIGH"
            confidence = "High confidence - Likely fraudulent"
            color = "red"
        
        # Response
        response = {
            'prediction': {
                'is_fake': bool(prediction),
                'label': 'FAKE' if prediction == 1 else 'REAL',
                'fake_probability': float(probability),
                'real_probability': float(1 - probability)
            },
            'risk_assessment': {
                'level': risk_level,
                'confidence': confidence,
                'color': color
            },
            'input_analysis': {
                'title_length': features_dict['title_length'],
                'description_length': features_dict['description_length'],
                'has_company_info': features_dict['has_company_profile'] == 1,
                'has_requirements': features_dict['has_requirements'] == 1,
                'word_count': features_dict['combined_word_count']
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    print("🚀 Starting Fake Job Detector API...")
    
    # Load model on startup
    if load_model():
        print("🌐 Starting Flask server...")
        # Use PORT from environment for Render
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    else:
        print("❌ Failed to load model. Please check the model file exists.")