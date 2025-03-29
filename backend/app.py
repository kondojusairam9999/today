from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import pickle
from sklearn.tree import DecisionTreeClassifier
import warnings

app = Flask(__name__)
# Enable CORS for all routes with specific origins
CORS(app, resources={
    r"/*": {
        "origins": [
            "http://localhost:3000",
            "https://*.netlify.app",  # Allows all Netlify subdomains
            "https://your-site-name.netlify.app"  # Replace with your actual Netlify URL
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Initialize model as None
model = None

# Define medicine recommendations based on symptoms and severity
MEDICINE_RECOMMENDATIONS = {
    'itching': {
        'mild': {
            'Child': 'Cetirizine (2.5-5 mg)',
            'Adult': 'Cetirizine (10 mg)',
            'Senior': 'Loratadine (5 mg)'
        },
        'moderate': {
            'Child': 'Hydroxyzine (1-2 mg/kg)',
            'Adult': 'Hydroxyzine (25 mg)',
            'Senior': 'Hydroxyzine (10-25 mg)'
        },
        'severe': {
            'Child': 'Prednisolone (Doctor advised)',
            'Adult': 'Prednisolone (20-40 mg)',
            'Senior': 'Prednisolone (Lower dose)'
        }
    },
    'nodalSkinEruptions': {
        'mild': {
            'Child': 'Calamine Lotion',
            'Adult': 'Calamine Lotion',
            'Senior': 'Calamine Lotion'
        },
        'moderate': {
            'Child': 'Hydrocortisone Cream (1%)',
            'Adult': 'Hydrocortisone Cream (Standard)',
            'Senior': 'Hydrocortisone Cream (Lower dose)'
        },
        'severe': {
            'Child': 'Methotrexate (Doctor advised)',
            'Adult': 'Methotrexate (Prescribed)',
            'Senior': 'Methotrexate (Lower dose)'
        }
    },
    'shivering': {
        'mild': {
            'Child': 'Paracetamol (120 mg)',
            'Adult': 'Paracetamol (500 mg)',
            'Senior': 'Paracetamol (Lower dose)'
        },
        'moderate': {
            'Child': 'Ibuprofen (10 mg/kg)',
            'Adult': 'Ibuprofen (400-600 mg)',
            'Senior': 'Ibuprofen (Lower dose)'
        },
        'severe': {
            'Child': 'Hospitalization',
            'Adult': 'Hospitalization',
            'Senior': 'Hospitalization'
        }
    },
    'stomachPain': {
        'mild': {
            'Child': 'Simethicone Drops',
            'Adult': 'Omeprazole (20 mg)',
            'Senior': 'Antacids'
        },
        'moderate': {
            'Child': 'Dicyclomine (Doctor advised)',
            'Adult': 'Dicyclomine (10 mg)',
            'Senior': 'Dicyclomine (5-10 mg)'
        },
        'severe': {
            'Child': 'ER Visit',
            'Adult': 'ER Visit',
            'Senior': 'ER Visit'
        }
    },
    'vomiting': {
        'mild': {
            'Child': 'ORS',
            'Adult': 'ORS',
            'Senior': 'ORS'
        },
        'moderate': {
            'Child': 'Ondansetron (2-4 mg)',
            'Adult': 'Ondansetron (4-8 mg)',
            'Senior': 'Ondansetron (Lower dose)'
        },
        'severe': {
            'Child': 'IV Fluids',
            'Adult': 'IV Fluids',
            'Senior': 'IV Fluids'
        }
    },
    'chestPain': {
        'mild': {
            'Child': 'Rest & Monitor',
            'Adult': 'Rest & Monitor',
            'Senior': 'Rest & Monitor'
        },
        'moderate': {
            'Child': 'Nitroglycerin (Doctor advised)',
            'Adult': 'Nitroglycerin (0.4 mg)',
            'Senior': 'Nitroglycerin (Lower dose)'
        },
        'severe': {
            'Child': 'ER Visit',
            'Adult': 'ER Visit',
            'Senior': 'ER Visit'
        }
    },
    'lossOfAppetite': {
        'mild': {
            'Child': 'Multivitamins (B-complex)',
            'Adult': 'Multivitamins (General)',
            'Senior': 'Balanced Diet'
        },
        'moderate': {
            'Child': 'Cyproheptadine (2 mg)',
            'Adult': 'Cyproheptadine (4 mg)',
            'Senior': 'Appetite Stimulants'
        },
        'severe': {
            'Child': 'Treat Underlying Cause',
            'Adult': 'Treat Underlying Cause',
            'Senior': 'Treat Underlying Cause'
        }
    },
    'yellowUrine': {
        'mild': {
            'Child': 'Increase Fluid Intake',
            'Adult': 'Increase Fluid Intake',
            'Senior': 'Increase Fluid Intake'
        },
        'moderate': {
            'Child': 'Antibiotics for UTI',
            'Adult': 'Antibiotics for UTI',
            'Senior': 'Antibiotics for UTI'
        },
        'severe': {
            'Child': 'Treat Underlying Condition',
            'Adult': 'Treat Underlying Condition',
            'Senior': 'Treat Underlying Condition'
        }
    },
    'restlessness': {
        'mild': {
            'Child': 'Warm Bath',
            'Adult': 'Chamomile Tea',
            'Senior': 'Relaxation'
        },
        'moderate': {
            'Child': 'Diazepam (2 mg)',
            'Adult': 'Diazepam (5 mg)',
            'Senior': 'Diazepam (Lower dose)'
        },
        'severe': {
            'Child': 'Psychiatric Care',
            'Adult': 'Psychiatric Care',
            'Senior': 'Psychiatric Care'
        }
    },
    'excessiveHunger': {
        'mild': {
            'Child': 'Balanced Diet',
            'Adult': 'Balanced Diet',
            'Senior': 'Balanced Diet'
        },
        'moderate': {
            'Child': 'Metformin (250 mg)',
            'Adult': 'Metformin (500 mg)',
            'Senior': 'Metformin (Lower dose)'
        },
        'severe': {
            'Child': 'Insulin Therapy',
            'Adult': 'Insulin Therapy',
            'Senior': 'Insulin Therapy'
        }
    },
    'highFever': {
        'mild': {
            'Child': 'Paracetamol (15 mg/kg)',
            'Adult': 'Paracetamol (500-650 mg)',
            'Senior': 'Paracetamol (Lower dose)'
        },
        'moderate': {
            'Child': 'Ibuprofen (10 mg/kg)',
            'Adult': 'Ibuprofen (400 mg)',
            'Senior': 'Ibuprofen (Cautiously)'
        },
        'severe': {
            'Child': 'Hospitalization',
            'Adult': 'Hospitalization',
            'Senior': 'Hospitalization'
        }
    },
    'diarrhoea': {
        'mild': {
            'Child': 'ORS + Zinc',
            'Adult': 'ORS + Zinc',
            'Senior': 'ORS + Zinc'
        },
        'moderate': {
            'Child': 'Loperamide (Doctor Advised)',
            'Adult': 'Loperamide (Standard)',
            'Senior': 'Loperamide (Lower dose)'
        },
        'severe': {
            'Child': 'IV Fluids + Antibiotics',
            'Adult': 'IV Fluids + Antibiotics',
            'Senior': 'IV Fluids + Antibiotics'
        }
    },
    'redSpotsOverBody': {
        'mild': {
            'Child': 'Antihistamines',
            'Adult': 'Antihistamines',
            'Senior': 'Antihistamines'
        },
        'moderate': {
            'Child': 'Hydrocortisone Cream',
            'Adult': 'Hydrocortisone Cream',
            'Senior': 'Hydrocortisone Cream'
        },
        'severe': {
            'Child': 'Immunosuppressants',
            'Adult': 'Immunosuppressants',
            'Senior': 'Immunosuppressants'
        }
    },
    'breathlessness': {
        'mild': {
            'Child': 'Steam Inhalation',
            'Adult': 'Steam Inhalation',
            'Senior': 'Steam Inhalation'
        },
        'moderate': {
            'Child': 'Salbutamol Inhaler',
            'Adult': 'Salbutamol Inhaler',
            'Senior': 'Salbutamol Inhaler (Lower dose)'
        },
        'severe': {
            'Child': 'Oxygen Therapy',
            'Adult': 'Oxygen Therapy',
            'Senior': 'Oxygen Therapy'
        }
    },
    'darkUrine': {
        'mild': {
            'Child': 'Hydration',
            'Adult': 'Hydration',
            'Senior': 'Hydration'
        },
        'moderate': {
            'Child': 'Antibiotics for UTI',
            'Adult': 'Antibiotics for UTI',
            'Senior': 'Antibiotics for UTI'
        },
        'severe': {
            'Child': 'Treat Underlying Condition',
            'Adult': 'Treat Underlying Condition',
            'Senior': 'Treat Underlying Condition'
        }
    },
    'skinRash': {
        'mild': {
            'Child': 'Aloe Vera Gel',
            'Adult': 'Aloe Vera Gel',
            'Senior': 'Aloe Vera Gel'
        },
        'moderate': {
            'Child': 'Hydrocortisone Cream',
            'Adult': 'Hydrocortisone Cream',
            'Senior': 'Hydrocortisone Cream'
        },
        'severe': {
            'Child': 'Prednisolone (Oral)',
            'Adult': 'Prednisolone (Oral)',
            'Senior': 'Prednisolone (Oral)'
        }
    },
    'continuousSneezing': {
        'mild': {
            'Child': 'Saline Nasal Spray',
            'Adult': 'Saline Nasal Spray',
            'Senior': 'Saline Nasal Spray'
        },
        'moderate': {
            'Child': 'Cetirizine (2.5 mg)',
            'Adult': 'Cetirizine (10 mg)',
            'Senior': 'Cetirizine (Lower dose)'
        },
        'severe': {
            'Child': 'Nasal Corticosteroids',
            'Adult': 'Nasal Corticosteroids',
            'Senior': 'Nasal Corticosteroids'
        }
    },
    'chills': {
        'mild': {
            'Child': 'Warm Fluids',
            'Adult': 'Warm Fluids',
            'Senior': 'Warm Fluids'
        },
        'moderate': {
            'Child': 'Paracetamol',
            'Adult': 'Paracetamol',
            'Senior': 'Paracetamol'
        },
        'severe': {
            'Child': 'Hospitalization',
            'Adult': 'Hospitalization',
            'Senior': 'Hospitalization'
        }
    },
    'ulcersOnTongue': {
        'mild': {
            'Child': 'Oral Gels',
            'Adult': 'Oral Gels',
            'Senior': 'Oral Gels'
        },
        'moderate': {
            'Child': 'Antiseptic Mouthwash',
            'Adult': 'Antiseptic Mouthwash',
            'Senior': 'Antiseptic Mouthwash'
        },
        'severe': {
            'Child': 'Doctor Consultation',
            'Adult': 'Doctor Consultation',
            'Senior': 'Doctor Consultation'
        }
    },
    'cough': {
        'mild': {
            'Child': 'Honey & Lemon',
            'Adult': 'Cough Syrup',
            'Senior': 'Cough Syrup (Lower dose)'
        },
        'moderate': {
            'Child': 'Dextromethorphan',
            'Adult': 'Dextromethorphan',
            'Senior': 'Dextromethorphan'
        },
        'severe': {
            'Child': 'Doctor Consultation',
            'Adult': 'Doctor Consultation',
            'Senior': 'Doctor Consultation'
        }
    },
    'yellowishSkin': {
        'mild': {
            'Child': 'Check Bilirubin',
            'Adult': 'Check Bilirubin',
            'Senior': 'Check Bilirubin'
        },
        'moderate': {
            'Child': 'Treat Underlying Cause',
            'Adult': 'Treat Underlying Cause',
            'Senior': 'Treat Underlying Cause'
        },
        'severe': {
            'Child': 'Hospitalization',
            'Adult': 'Hospitalization',
            'Senior': 'Hospitalization'
        }
    },
    'abdominalPain': {
        'mild': {
            'Child': 'Simethicone Drops',
            'Adult': 'Dicyclomine (10 mg)',
            'Senior': 'Dicyclomine (5-10 mg)'
        },
        'moderate': {
            'Child': 'Painkillers (Doctor Advised)',
            'Adult': 'Painkillers (Doctor Advised)',
            'Senior': 'Painkillers (Doctor Advised)'
        },
        'severe': {
            'Child': 'ER Visit',
            'Adult': 'ER Visit',
            'Senior': 'ER Visit'
        }
    },
    'weightLoss': {
        'mild': {
            'Child': 'Nutritional Support',
            'Adult': 'Nutritional Support',
            'Senior': 'Nutritional Support'
        },
        'moderate': {
            'Child': 'Appetite Stimulants',
            'Adult': 'Appetite Stimulants',
            'Senior': 'Appetite Stimulants'
        },
        'severe': {
            'Child': 'Treat Underlying Cause',
            'Adult': 'Treat Underlying Cause',
            'Senior': 'Treat Underlying Cause'
        }
    },
    'irregularSugarLevel': {
        'mild': {
            'Child': 'Dietary Changes',
            'Adult': 'Dietary Changes',
            'Senior': 'Dietary Changes'
        },
        'moderate': {
            'Child': 'Metformin (Doctor advised)',
            'Adult': 'Metformin (500 mg)',
            'Senior': 'Metformin (Lower dose)'
        },
        'severe': {
            'Child': 'Insulin Therapy',
            'Adult': 'Insulin Therapy',
            'Senior': 'Insulin Therapy'
        }
    },
    'increasedAppetite': {
        'mild': {
            'Child': 'Balanced Diet',
            'Adult': 'Balanced Diet',
            'Senior': 'Balanced Diet'
        },
        'moderate': {
            'Child': 'Metformin (250 mg)',
            'Adult': 'Metformin (500 mg)',
            'Senior': 'Metformin (Lower dose)'
        },
        'severe': {
            'Child': 'Treat Underlying Cause',
            'Adult': 'Treat Underlying Cause',
            'Senior': 'Treat Underlying Cause'
        }
    },
    'headache': {
        'mild': {
            'Child': 'Paracetamol (120 mg)',
            'Adult': 'Paracetamol (500 mg)',
            'Senior': 'Paracetamol (Lower dose)'
        },
        'moderate': {
            'Child': 'NSAIDs (Ibuprofen)',
            'Adult': 'NSAIDs (Ibuprofen)',
            'Senior': 'NSAIDs (Ibuprofen)'
        },
        'severe': {
            'Child': 'Hospitalization',
            'Adult': 'Hospitalization',
            'Senior': 'Hospitalization'
        }
    },
    'musclePain': {
        'mild': {
            'Child': 'Warm Compress',
            'Adult': 'Warm Compress',
            'Senior': 'Warm Compress'
        },
        'moderate': {
            'Child': 'NSAIDs',
            'Adult': 'NSAIDs',
            'Senior': 'NSAIDs'
        },
        'severe': {
            'Child': 'Physiotherapy',
            'Adult': 'Physiotherapy',
            'Senior': 'Physiotherapy'
        }
    },
    'runnyNose': {
        'mild': {
            'Child': 'Saline Nasal Spray',
            'Adult': 'Saline Nasal Spray',
            'Senior': 'Saline Nasal Spray'
        },
        'moderate': {
            'Child': 'Antihistamines',
            'Adult': 'Antihistamines',
            'Senior': 'Antihistamines'
        },
        'severe': {
            'Child': 'Doctor Consultation',
            'Adult': 'Doctor Consultation',
            'Senior': 'Doctor Consultation'
        }
    },
    'fastHeartRate': {
        'mild': {
            'Child': 'Rest & Hydration',
            'Adult': 'Rest & Hydration',
            'Senior': 'Rest & Hydration'
        },
        'moderate': {
            'Child': 'Beta-blockers (Doctor Advised)',
            'Adult': 'Beta-blockers (Doctor Advised)',
            'Senior': 'Beta-blockers (Doctor Advised)'
        },
        'severe': {
            'Child': 'ER Visit',
            'Adult': 'ER Visit',
            'Senior': 'ER Visit'
        }
    }
}

def get_recommended_medicines(features, severity, age_group):
    medicines = set()
    severity_level = 'mild' if severity == 0 else 'moderate' if severity == 1 else 'severe'
    age_group_text = 'Child' if age_group == 0 else 'Adult' if age_group == 1 else 'Senior'
    
    # Map feature indices to symptom names
    symptom_map = {
        1: 'itching',
        2: 'nodalSkinEruptions',
        3: 'shivering',
        4: 'stomachPain',
        5: 'vomiting',
        6: 'chestPain',
        7: 'lossOfAppetite',
        8: 'yellowUrine',
        9: 'restlessness',
        10: 'excessiveHunger',
        11: 'highFever',
        12: 'diarrhoea',
        13: 'redSpotsOverBody',
        14: 'breathlessness',
        15: 'darkUrine',
        16: 'skinRash',
        17: 'continuousSneezing',
        18: 'chills',
        19: 'ulcersOnTongue',
        20: 'cough',
        21: 'yellowishSkin',
        22: 'abdominalPain',
        23: 'weightLoss',
        24: 'irregularSugarLevel',
        25: 'increasedAppetite',
        26: 'headache',
        27: 'musclePain',
        28: 'runnyNose',
        29: 'fastHeartRate',
        30: 'duration',
        31: 'allergies',
        32: 'fatigue',
        33: 'additionalInformation'  # Added the new feature
    }
    
    # Check each symptom and add corresponding medicine
    for idx, value in enumerate(features[1:34], 1):  # Check all 34 features
        if value == 1 and idx in symptom_map:
            symptom = symptom_map[idx]
            if symptom in MEDICINE_RECOMMENDATIONS:
                medicines.add(MEDICINE_RECOMMENDATIONS[symptom][severity_level][age_group_text])
    
    return list(medicines)

def calculate_severity(features):
    # Count the number of symptoms (1s) in the features
    symptom_count = np.sum(features[1:34])  # Updated to count all 34 features
    
    # Calculate severity based on symptom count
    if symptom_count <= 2:
        return 0  # Mild
    elif symptom_count <= 5:
        return 1  # Moderate
    else:
        return 2  # Severe

def load_model():
    global model
    try:
        model_path = os.path.join(os.path.dirname(__file__), 'high_efficiency_model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")
        
        # Try to load with joblib first
        try:
            model = joblib.load(model_path)
        except:
            # If joblib fails, try pickle
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        
        # If model is a numpy array, create a new DecisionTreeClassifier
        if isinstance(model, np.ndarray):
            # Create a new DecisionTreeClassifier
            new_model = DecisionTreeClassifier()
            # Set the tree structure
            new_model.tree_ = model
            # Set other required attributes
            new_model.n_features_in_ = 10  # Number of features expected
            new_model.n_classes_ = 3  # Three severity levels
            new_model.classes_ = np.array([0, 1, 2])  # Class labels for severity levels
            # Set the predict method to use our custom severity calculation
            new_model.predict = lambda X: np.array([calculate_severity(x) for x in X])
            model = new_model
        
        # Verify the model has predict method
        if not hasattr(model, 'predict'):
            raise ValueError("Loaded model does not have predict method")
            
        return True
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return False

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({'error': 'Model not loaded properly'}), 500

        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({'error': 'No features provided'}), 400

        features = data['features']
        if len(features) != 34:  # Update to match the number of features
            return jsonify({'error': f'Expected 34 features, but got {len(features)}'}), 400

        # Convert features to numpy array and reshape
        features_array = np.array(features).reshape(1, -1)

        # Check if all symptoms are 0 (excluding age group)
        symptom_features = features[1:]  # Get all features except age group
        if np.all(symptom_features == 0):
            return jsonify({'prediction': 'No specific medicines recommended as no symptoms are present.'})

        # Get severity level
        severity = model.predict(features_array)[0]
        severity_text = 'Mild' if severity == 0 else 'Moderate' if severity == 1 else 'Severe'
        
        # Get age group from features
        age_group = features[0]  # First feature is age group
        
        # Get recommended medicines based on symptoms, severity, and age group
        recommended_medicines = get_recommended_medicines(features, severity, age_group)
        
        # Format the response with clear sections
        prediction_text = f"""
Severity Level: {severity_text}

Recommended Medicines:
{chr(10).join(f'• {medicine}' for medicine in recommended_medicines)}

Note: Please consult a healthcare professional before taking any medications.
"""

        return jsonify({'prediction': prediction_text})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'model_loaded': model is not None})

if __name__ == '__main__':
    # Try to load model on startup
    if not load_model():
        print("Warning: Model could not be loaded on startup")
    app.run(debug=True)
