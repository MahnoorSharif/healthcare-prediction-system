from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MDL_DIR = os.path.join(BASE_DIR, 'models')
DATA_DIR = os.path.join(BASE_DIR, 'datasets')

app = FastAPI(
    title='Healthcare Risk API',
    description='Backend service for Diabetes and Heart Disease prediction',
    version='1.0.0'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        'http://localhost:5173',
        'http://127.0.0.1:5173',
        'http://localhost:5174',
        'http://127.0.0.1:5174',
        'http://localhost:3000',
        'http://127.0.0.1:3000',
    ],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)

class PredictRequest(BaseModel):
    disease: str
    features: dict[str, float]

class PredictResponse(BaseModel):
    disease: str
    models: list[dict]
    best_model: dict


def clean_model_input(X):
    return np.nan_to_num(np.asarray(X, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)


def patch_loaded_model(model):
    if hasattr(model, 'predict_proba') and not hasattr(model, 'multi_class'):
        try:
            model.multi_class = 'ovr'
        except Exception:
            pass
    return model


def load_assets():
    assets = {}
    assets['scaler_d'] = joblib.load(os.path.join(MDL_DIR, 'scaler_diabetes.pkl'))
    assets['lr_d'] = patch_loaded_model(joblib.load(os.path.join(MDL_DIR, 'logistic_regression_diabetes.pkl')))
    assets['knn_d'] = patch_loaded_model(joblib.load(os.path.join(MDL_DIR, 'knn_diabetes.pkl')))
    assets['dt_d'] = patch_loaded_model(joblib.load(os.path.join(MDL_DIR, 'decision_tree_diabetes.pkl')))
    assets['scaler_h'] = joblib.load(os.path.join(MDL_DIR, 'scaler_heart.pkl'))
    assets['lr_h'] = patch_loaded_model(joblib.load(os.path.join(MDL_DIR, 'logistic_regression_heart.pkl')))
    assets['knn_h'] = patch_loaded_model(joblib.load(os.path.join(MDL_DIR, 'knn_heart.pkl')))
    assets['dt_h'] = patch_loaded_model(joblib.load(os.path.join(MDL_DIR, 'decision_tree_heart.pkl')))
    assets['heart_cols'] = joblib.load(os.path.join(MDL_DIR, 'heart_columns.pkl'))
    with open(os.path.join(MDL_DIR, 'best_models.json'), 'r') as f:
        assets['best'] = json.load(f)
    return assets

assets = load_assets()


@app.get('/')
def health_check():
    return {
        'status': 'ok',
        'service': 'Healthcare Risk API',
        'models_available': ['Diabetes', 'Heart Disease']
    }


@app.get('/visuals/{filename}')
def get_visual(filename: str):
    safe_filename = os.path.basename(filename)
    file_path = os.path.join(MDL_DIR, safe_filename)

    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail='Visual not found')

    return FileResponse(file_path, media_type='image/png')


@app.post('/predict', response_model=PredictResponse)
def predict(request: PredictRequest):
    disease = request.disease.strip().lower()
    features = request.features

    if disease not in {'diabetes', 'heart disease', 'heart'}:
        raise HTTPException(status_code=400, detail='Disease must be Diabetes or Heart Disease')

    if disease.startswith('diabetes'):
        required = ['pregnancies', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'dpf', 'age']
        missing = [f for f in required if f not in features]
        if missing:
            raise HTTPException(status_code=400, detail=f'Missing features: {missing}')

        raw_input = [[
            features['pregnancies'],
            features['glucose'],
            features['bp'],
            features['skin'],
            features['insulin'],
            features['bmi'],
            features['dpf'],
            features['age'],
        ]]
        x = clean_model_input(assets['scaler_d'].transform(raw_input))
        model_set = {
            'Logistic Regression': assets['lr_d'],
            'KNN': assets['knn_d'],
            'Decision Tree': assets['dt_d'],
        }
        label_map = {0: 'No Diabetes', 1: 'Diabetes'}
        best_name = assets['best']['diabetes']
        explanation_map = {
            0: 'Low risk prediction based on diabetes model signals.',
            1: 'High risk prediction based on diabetes model signals.',
        }

    else:
        required = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        missing = [f for f in required if f not in features]
        if missing:
            raise HTTPException(status_code=400, detail=f'Missing features: {missing}')

        base_input = {
            'age': features['age'],
            'sex': features['sex'],
            'fbs': features['fbs'],
            'trestbps': features['trestbps'],
            'chol': features['chol'],
            'thalach': features['thalach'],
            'exang': features['exang'],
            'oldpeak': features['oldpeak'],
        }
        for col_name, val, values in [
            ('cp', features['cp'], [0, 1, 2, 3]),
            ('restecg', features['restecg'], [0, 1, 2]),
            ('slope', features['slope'], [0, 1, 2]),
            ('ca', features['ca'], [0, 1, 2, 3, 4]),
            ('thal', features['thal'], [0, 1, 2, 3]),
        ]:
            for option in values:
                base_input[f'{col_name}_{option}'] = 1 if val == option else 0

        heart_cols = assets['heart_cols']
        for col in heart_cols:
            if col not in base_input:
                base_input[col] = 0

        input_df = pd.DataFrame([base_input])
        x = clean_model_input(assets['scaler_h'].transform(input_df[heart_cols]))
        model_set = {
            'Logistic Regression': assets['lr_h'],
            'KNN': assets['knn_h'],
            'Decision Tree': assets['dt_h'],
        }
        label_map = {0: 'No Disease', 1: 'Heart Disease'}
        best_name = assets['best']['heart']
        explanation_map = {
            0: 'Low risk prediction based on heart disease model signals.',
            1: 'High risk prediction based on heart disease model signals.',
        }

    models = []
    best_model = None

    for model_name, model in model_set.items():
        prediction = int(model.predict(x)[0])
        proba = float(model.predict_proba(x)[0][1] * 100)
        models.append({
            'name': model_name,
            'label': label_map[prediction],
            'prediction': prediction,
            'probability': proba,
        })

        if model_name == best_name:
            best_model = {
                'name': model_name,
                'label': label_map[prediction],
                'prediction': prediction,
                'probability': proba,
                'explanation': explanation_map[prediction],
            }

    if best_model is None:
        raise HTTPException(status_code=500, detail='Best model metadata is missing')

    return {
        'disease': 'Diabetes' if disease.startswith('diabetes') else 'Heart Disease',
        'models': models,
        'best_model': best_model,
    }
