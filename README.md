# GlucoSense - Diabetes Prediction with ML and RAG
A comprehensive diabetes prediction system that combines ensemble machine learning with Retrieval-Augmented Generation (RAG) for accurate risk assessment and clinically-grounded explanations.

## 🚀 Key Features

- Stacked Ensemble Model: Combines AdaBoost, XGBoost, and CatBoost for superior predictive performance
- FAISS Similarity Search: Retrieves clinically similar patient cases for evidence-based explanations
- Google Gemini Integration: Generates personalized, human-readable clinical explanations
- SMOTE Balancing: Handles class imbalance for robust model training
- Feature Importance Analysis: Identifies key clinical drivers of diabetes risk
- Clinical Safety: Includes disclaimers and avoids medical treatment recommendations
  
## 📊 Model Performance

- **Accuracy**: 95%
- **ROC-AUC**: 0.989
- **Precision**: 95%
- **Recall**: 95%

## 🛠️ Installation

Prerequisites
- Python 3.8+
- Google Gemini API key
  
**Clone the repository**:
 ```bash
   git clone https://github.com/akshata-13/GlucoSense.git
   cd GlucoSense
 ```

## Install dependencies:

```bash
pip install -r requirements.txt
```

## Set up environment variables:

Create .env file in project root

Add your Gemini API key:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```
Get your free API key from: [Google AI Studio](https://aistudio.google.com/app/api-keys)

## Run the system:
```bash
python glucosense.py
```

## 📈 Technical Details

## Models Used
- AdaBoost: Sequential ensemble with 100 estimators
- XGBoost: Gradient boosting with 200 estimators, 0.05 learning rate
- CatBoost: Meta-learner with efficient categorical handling
- Stacked Ensemble: Combines all models with 5-fold cross-validation

## Feature Engineering
- Top Features: Glucose, BMI, Age, Blood Pressure, Family History
- Preprocessing: StandardScaler, SMOTE, Label Encoding
- Selection: XGBoost-based feature importance thresholding

## RAG Implementation
- Embeddings: SentenceTransformer (all-MiniLM-L6-v2)
- Similarity: FAISS L2 distance with neighbor filtering
- Retrieval: Top 50 neighbors filtered by predicted labels

## 📋 Dataset
The system uses the Mendeley Diabetes_Final_Data_V2 dataset: https://data.mendeley.com/datasets/7m7555vgrn/1

## ⚠️ MEDICAL DISCLAIMER: 
This system is designed for educational and research purposes only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult qualified healthcare providers for medical decisions.

## 📄 License
This project is licensed under the MIT License.

## Contributions are welcome! Please feel free to submit a Pull Request.
