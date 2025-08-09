# # Step 7: Flask API for Fake Job Detector
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# import joblib
# import pandas as pd
# import re
# import scipy.sparse as sp
# import numpy as np
# from datetime import datetime
# import os
# import logging

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)
# # Initialize Flask app
# app = Flask(__name__)
# CORS(app, origins=[
#     "http://localhost:3000",  # Local development
#     "https://fake-job-posting-detector.onrender.com",  # Replace with your actual frontend URL
# ])

# # Global variables for model components
# model_pipeline = None
# svm_model = None
# tfidf = None
# scaler = None
# numerical_features = None

# # Load model on startup
# def load_model():
#     """Load the trained model pipeline"""
#     global model_pipeline, svm_model, tfidf, scaler, numerical_features
     
    
#     # try:
#     #     # Load the model pipeline
#     #     model_path = 'fake_job_detector_v1.pkl'
#     #     model_pipeline = joblib.load(model_path)
        
#     #     # Extract components
#     #     svm_model = model_pipeline['svm_model']
#     #     tfidf = model_pipeline['tfidf_vectorizer']
#     #     scaler = model_pipeline['scaler']
#     #     numerical_features = model_pipeline['numerical_features']
        
#     #     print("✅ Model loaded successfully!")
#     #     print(f"📊 Model trained on: {model_pipeline['training_date']}")
#     #     print(f"🎯 Model accuracy: {model_pipeline['performance_metrics']['accuracy']:.2%}")
        
#     #     return True
        
#     # except Exception as e:
#     #     print(f"❌ Error loading model: {str(e)}")
#     #     return False

#     try:
#         # Check if model file exists
#         model_path = 'fake_job_detector_v1.pkl'
#         if not os.path.exists(model_path):
#             logger.error(f"Model file {model_path} not found in directory: {os.getcwd()}")
#             logger.info(f"Files in current directory: {os.listdir('.')}")
#             return False
        
#         # Check file size
#         file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
#         logger.info(f"Model file size: {file_size:.2f} MB")
        
#         # Load the model pipeline
#         logger.info("Loading model pipeline...")
#         model_pipeline = joblib.load(model_path)
        
#         # Extract components
#         svm_model = model_pipeline['svm_model']
#         tfidf = model_pipeline['tfidf_vectorizer']
#         scaler = model_pipeline['scaler']
#         numerical_features = model_pipeline['numerical_features']
        
#         logger.info("✅ Model loaded successfully!")
#         logger.info(f"📊 Model trained on: {model_pipeline['training_date']}")
#         logger.info(f"🎯 Model accuracy: {model_pipeline['performance_metrics']['accuracy']:.2%}")
        
#         return True
        
#     except Exception as e:
#         logger.error(f"❌ Error loading model: {str(e)}")
#         import traceback
#         logger.error(traceback.format_exc())
#         return False

# # Text cleaning function (same as training)
# def clean_text(text):
#     """Clean and preprocess text data"""
#     if pd.isna(text) or text == '':
#         return ''
    
#     # Convert to string and lowercase
#     text = str(text).lower()
    
#     # Remove HTML tags
#     text = re.sub(r'<[^>]+>', ' ', text)
    
#     # Remove URLs
#     text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    
#     # Remove email addresses
#     text = re.sub(r'\S+@\S+', ' ', text)
    
#     # Remove extra whitespace
#     text = re.sub(r'\s+', ' ', text)
    
#     # Remove special characters but keep basic punctuation
#     text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', ' ', text)
    
#     return text.strip()

# def create_features(title, description, company_profile='', requirements='', benefits=''):
#     """Create features for prediction (same process as training)"""
    
#     # Clean text inputs
#     title = clean_text(title)
#     description = clean_text(description)
#     company_profile = clean_text(company_profile)
#     requirements = clean_text(requirements)
#     benefits = clean_text(benefits)
    
#     # Combine text features (same weighting as training)
#     combined_text = f"{title} {title} {title} {description} {description} {requirements} {company_profile} {benefits}".strip()
    
#     # Create numerical features
#     features_dict = {
#         'title_length': len(title),
#         'description_length': len(description),
#         'requirements_length': len(requirements),
#         'combined_length': len(combined_text),
#         'title_word_count': len(title.split()) if title else 0,
#         'description_word_count': len(description.split()) if description else 0,
#         'combined_word_count': len(combined_text.split()) if combined_text else 0,
#         'has_company_profile': 1 if company_profile else 0,
#         'has_requirements': 1 if requirements else 0,
#         'has_benefits': 1 if benefits else 0,
#         'exclamation_count': combined_text.count('!'),
#         'capital_ratio': sum(1 for c in combined_text if c.isupper()) / len(combined_text) if len(combined_text) > 0 else 0
#     }
    
#     return combined_text, features_dict

# @app.route('/')
# def home():
#     """API home endpoint"""
#     return jsonify({
#         'message': 'Fake Job Detector API',
#         'version': model_pipeline['model_version'] if model_pipeline else 'Model not loaded',
#         'status': 'ready' if model_pipeline else 'error',
#         'endpoints': {
#             'predict': '/predict (POST)',
#             'health': '/health (GET)',
#             'model_info': '/model-info (GET)'
#         }
#     })

# @app.route('/health')
# def health():
#     """Health check endpoint"""
#     return jsonify({
#         'status': 'healthy' if model_pipeline else 'unhealthy',
#         'timestamp': datetime.now().isoformat(),
#         'model_loaded': model_pipeline is not None
#     })

# @app.route('/model-info')
# def model_info():
#     """Get model information"""
#     if not model_pipeline:
#         return jsonify({'error': 'Model not loaded'}), 500
    
#     return jsonify({
#         'model_version': model_pipeline['model_version'],
#         'training_date': model_pipeline['training_date'],
#         'performance_metrics': model_pipeline['performance_metrics'],
#         'training_info': model_pipeline['training_info']
#     })

# @app.route('/predict', methods=['POST'])
# def predict():
#     """Main prediction endpoint"""
#     try:
#         # Check if model is loaded
#         if not model_pipeline:
#             return jsonify({'error': 'Model not loaded'}), 500
        
#         # Get JSON data
#         data = request.get_json()
        
#         # Validate required fields
#         if not data:
#             return jsonify({'error': 'No JSON data provided'}), 400
        
#         if 'title' not in data or 'description' not in data:
#             return jsonify({'error': 'Title and description are required'}), 400
        
#         # Extract job posting data
#         title = data.get('title', '')
#         description = data.get('description', '')
#         company_profile = data.get('company_profile', '')
#         requirements = data.get('requirements', '')
#         benefits = data.get('benefits', '')
        
#         # Validate inputs
#         if not title.strip() or not description.strip():
#             return jsonify({'error': 'Title and description cannot be empty'}), 400
        
#         # Create features
#         combined_text, features_dict = create_features(
#             title, description, company_profile, requirements, benefits
#         )
        
#         # Transform text features
#         X_text = tfidf.transform([combined_text])
        
#         # Transform numerical features
#         X_numerical = np.array([features_dict[col] for col in numerical_features]).reshape(1, -1)
#         X_num_scaled = scaler.transform(X_numerical)
        
#         # Combine features
#         X_combined = sp.hstack([X_text, sp.csr_matrix(X_num_scaled)])
        
#         # Make predictions
#         prediction = svm_model.predict(X_combined)[0]
#         probability = svm_model.predict_proba(X_combined)[0, 1]
        
#         # Determine risk level and confidence
#         if probability < 0.25:
#             risk_level = "LOW"
#             confidence = "High confidence - Likely legitimate"
#             color = "green"
#         elif probability < 0.6:
#             risk_level = "MEDIUM"
#             confidence = "Moderate confidence - Review carefully"
#             color = "yellow"
#         else:
#             risk_level = "HIGH"
#             confidence = "High confidence - Likely fraudulent"
#             color = "red"
        
#         # Response
#         response = {
#             'prediction': {
#                 'is_fake': bool(prediction),
#                 'label': 'FAKE' if prediction == 1 else 'REAL',
#                 'fake_probability': float(probability),
#                 'real_probability': float(1 - probability)
#             },
#             'risk_assessment': {
#                 'level': risk_level,
#                 'confidence': confidence,
#                 'color': color
#             },
#             'input_analysis': {
#                 'title_length': features_dict['title_length'],
#                 'description_length': features_dict['description_length'],
#                 'has_company_info': features_dict['has_company_profile'] == 1,
#                 'has_requirements': features_dict['has_requirements'] == 1,
#                 'word_count': features_dict['combined_word_count']
#             },
#             'timestamp': datetime.now().isoformat()
#         }
        
#         return jsonify(response)
        
#     except Exception as e:
#         return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# # Error handlers
# @app.errorhandler(404)
# def not_found(error):
#     return jsonify({'error': 'Endpoint not found'}), 404

# @app.errorhandler(500)
# def internal_error(error):
#     return jsonify({'error': 'Internal server error'}), 500

# if __name__ == '__main__':
#     print("🚀 Starting Fake Job Detector API...")
    
#     # Load model on startup
#     if load_model():
#         print("🌐 Starting Flask server...")
#         # Use PORT from environment for Render
#         port = int(os.environ.get('PORT', 5000))
#         app.run(debug=False, host='0.0.0.0', port=port)
#     else:
#         print("❌ Failed to load model. Please check the model file exists.")

































# Diagnostic Flask API for Fake Job Detector
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import re
import scipy.sparse as sp
import numpy as np
from datetime import datetime
import os
import logging
import sys
import traceback

# Configure logging to show more details
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)

# Configure CORS
CORS(app, origins=[
    "http://localhost:3000",
    "https://fake-job-posting-detector.onrender.com",
    "https://*.onrender.com"
])

# Global variables for model components
model_pipeline = None
svm_model = None
tfidf = None
scaler = None
numerical_features = None

# Load model on startup
def load_model():
    """Load the trained model pipeline with detailed logging"""
    global model_pipeline, svm_model, tfidf, scaler, numerical_features
    
    try:
        # Log current working directory and files
        current_dir = os.getcwd()
        logger.info(f"Current working directory: {current_dir}")
        logger.info(f"Files in current directory: {os.listdir('.')}")
        
        # Check for common model file names
        possible_model_files = [
            'fake_job_detector_v1.pkl',
            'model.pkl',
            'fake_job_detector.pkl'
        ]
        
        model_path = None
        for filename in possible_model_files:
            if os.path.exists(filename):
                model_path = filename
                logger.info(f"Found model file: {filename}")
                break
        
        if not model_path:
            logger.error(f"No model file found. Looking for: {possible_model_files}")
            # List all .pkl files
            pkl_files = [f for f in os.listdir('.') if f.endswith('.pkl')]
            logger.info(f"Available .pkl files: {pkl_files}")
            return False
        
        # Check file size
        file_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
        logger.info(f"Model file size: {file_size:.2f} MB")
        
        # Load the model pipeline
        logger.info("Loading model pipeline...")
        model_pipeline = joblib.load(model_path)
        logger.info("Model pipeline loaded successfully!")
        
        # Check model structure
        logger.info(f"Model pipeline keys: {list(model_pipeline.keys()) if isinstance(model_pipeline, dict) else 'Not a dictionary'}")
        
        # Try to extract components
        if isinstance(model_pipeline, dict):
            svm_model = model_pipeline.get('svm_model')
            tfidf = model_pipeline.get('tfidf_vectorizer')
            scaler = model_pipeline.get('scaler')
            numerical_features = model_pipeline.get('numerical_features')
            
            logger.info(f"SVM model loaded: {svm_model is not None}")
            logger.info(f"TF-IDF vectorizer loaded: {tfidf is not None}")
            logger.info(f"Scaler loaded: {scaler is not None}")
            logger.info(f"Numerical features loaded: {numerical_features is not None}")
            
            if all([svm_model, tfidf, scaler, numerical_features]):
                logger.info("✅ All model components loaded successfully!")
                if 'performance_metrics' in model_pipeline:
                    logger.info(f"🎯 Model accuracy: {model_pipeline['performance_metrics'].get('accuracy', 'Unknown')}")
                return True
            else:
                logger.error("❌ Some model components are missing")
                return False
        else:
            logger.error("❌ Model pipeline is not a dictionary")
            return False
        
    except Exception as e:
        logger.error(f"❌ Error loading model: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return False

# Text cleaning function
def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text) or text == '':
        return ''
    
    text = str(text).lower()
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-zA-Z0-9\s\.\,\!\?]', ' ', text)
    return text.strip()

def create_features(title, description, company_profile='', requirements='', benefits=''):
    """Create features for prediction"""
    title = clean_text(title)
    description = clean_text(description)
    company_profile = clean_text(company_profile)
    requirements = clean_text(requirements)
    benefits = clean_text(benefits)
    
    combined_text = f"{title} {title} {title} {description} {description} {requirements} {company_profile} {benefits}".strip()
    
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
    """API home endpoint with diagnostics"""
    return jsonify({
        'message': 'Fake Job Detector API - Diagnostic Version',
        'version': model_pipeline.get('model_version', 'Unknown') if model_pipeline else 'Model not loaded',
        'status': 'ready' if model_pipeline else 'error - model not loaded',
        'model_loaded': model_pipeline is not None,
        'current_directory': os.getcwd(),
        'available_files': os.listdir('.'),
        'pkl_files': [f for f in os.listdir('.') if f.endswith('.pkl')],
        'endpoints': {
            'predict': '/predict (POST)',
            'health': '/health (GET)',
            'debug': '/debug (GET)',
            'model_info': '/model-info (GET)'
        }
    })

@app.route('/debug')
def debug():
    """Debug endpoint to check file system and model status"""
    try:
        debug_info = {
            'working_directory': os.getcwd(),
            'directory_contents': os.listdir('.'),
            'pkl_files': [f for f in os.listdir('.') if f.endswith('.pkl')],
            'model_loaded': model_pipeline is not None,
            'python_version': sys.version,
            'environment_vars': {
                'PORT': os.environ.get('PORT', 'Not set'),
                'PYTHON_VERSION': os.environ.get('PYTHON_VERSION', 'Not set')
            }
        }
        
        if model_pipeline:
            debug_info['model_components'] = {
                'svm_model': svm_model is not None,
                'tfidf_vectorizer': tfidf is not None,
                'scaler': scaler is not None,
                'numerical_features': numerical_features is not None
            }
            if isinstance(model_pipeline, dict):
                debug_info['model_keys'] = list(model_pipeline.keys())
        
        return jsonify(debug_info)
    
    except Exception as e:
        return jsonify({
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy' if model_pipeline else 'unhealthy - model not loaded',
        'timestamp': datetime.now().isoformat(),
        'model_loaded': model_pipeline is not None
    })

@app.route('/model-info')
def model_info():
    """Get model information"""
    if not model_pipeline:
        return jsonify({'error': 'Model not loaded'}), 500
    
    if isinstance(model_pipeline, dict):
        return jsonify({
            'model_version': model_pipeline.get('model_version', 'Unknown'),
            'training_date': model_pipeline.get('training_date', 'Unknown'),
            'performance_metrics': model_pipeline.get('performance_metrics', {}),
            'training_info': model_pipeline.get('training_info', {})
        })
    else:
        return jsonify({'error': 'Model format not recognized'}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint with detailed error handling"""
    try:
        logger.info("Prediction request received")
        
        # Check if model is loaded
        if not model_pipeline:
            logger.error("Model not loaded - cannot make predictions")
            return jsonify({'error': 'Model not loaded'}), 500
        
        if not all([svm_model, tfidf, scaler, numerical_features]):
            logger.error("Model components missing")
            return jsonify({'error': 'Model components not properly loaded'}), 500
        
        # Get JSON data
        data = request.get_json()
        logger.info(f"Received data: {data}")
        
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
        
        logger.info("Creating features...")
        # Create features
        combined_text, features_dict = create_features(
            title, description, company_profile, requirements, benefits
        )
        
        logger.info("Transforming text features...")
        # Transform text features
        X_text = tfidf.transform([combined_text])
        
        logger.info("Transforming numerical features...")
        # Transform numerical features
        X_numerical = np.array([features_dict[col] for col in numerical_features]).reshape(1, -1)
        X_num_scaled = scaler.transform(X_numerical)
        
        logger.info("Combining features...")
        # Combine features
        X_combined = sp.hstack([X_text, sp.csr_matrix(X_num_scaled)])
        
        logger.info("Making prediction...")
        # Make predictions
        prediction = svm_model.predict(X_combined)[0]
        probability = svm_model.predict_proba(X_combined)[0, 1]
        
        logger.info(f"Prediction: {prediction}, Probability: {probability}")
        
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
        
        logger.info("Prediction successful")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

# Initialize model when the module is imported (for gunicorn)
logger.info("🚀 Initializing Fake Job Detector API...")
load_model()

if __name__ == '__main__':
    logger.info("🚀 Starting Fake Job Detector API...")
    
    # Load model on startup
    if not model_pipeline:
        load_model()
    
    logger.info("🌐 Starting Flask server...")
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)