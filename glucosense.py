# diabetes_predictor.py
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import AdaBoostClassifier, StackingClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report, roc_auc_score
from imblearn.over_sampling import SMOTE
import torch
import os
import textwrap
from dotenv import load_dotenv
import google.generativeai as genai
import matplotlib.pyplot as plt
import seaborn as sns

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

class DiabetesPredictor:
    def __init__(self, data_path="Diabetes_Final_Data_V2.csv"):
        self.model_features = [
            'gender', 'age', 'pulse_rate', 'systolic_bp', 'diastolic_bp',
            'glucose', 'height', 'weight', 'bmi',
            'family_diabetes', 'hypertensive', 'family_hypertension',
            'cardiovascular_disease', 'stroke'
        ]
        
        self.data_path = data_path
        self.is_trained = False
        
    def load_and_train(self):
        """Load data and train the model"""
        print("📁 Loading data...")
        
        # Load CSV file
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Preprocessing
        df['diabetic'] = df['diabetic'].map({'Yes': 1, 'No': 0})
        df['gender'] = LabelEncoder().fit_transform(df['gender'])
        
        X = df[self.model_features]
        y = df['diabetic']

        # Preprocessing
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Handle class imbalance
        print("🔄 Handling class imbalance with SMOTE...")
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
        )

        # Feature selection
        print("🔍 Selecting important features...")
        xgb_fs = XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric='logloss')
        xgb_fs.fit(X_train, y_train)

        self.selector = SelectFromModel(xgb_fs, threshold="median", prefit=True)
        X_train_sel = self.selector.transform(X_train)
        X_test_sel = self.selector.transform(X_test)

        # Model training
        print("🤖 Training ensemble model...")
        ada = AdaBoostClassifier(n_estimators=100, random_state=42)
        xgb = XGBClassifier(n_estimators=200, learning_rate=0.05, subsample=0.9,
                           random_state=42, use_label_encoder=False, eval_metric='logloss')
        cat_meta = CatBoostClassifier(verbose=0, random_seed=42)

        self.stacked_model = StackingClassifier(
            estimators=[('ada', ada), ('xgb', xgb)],
            final_estimator=cat_meta,
            passthrough=True,
            cv=5
        )

        self.stacked_model.fit(X_train_sel, y_train)

        # Evaluation
        print("📊 Evaluating model...")
        y_pred = self.stacked_model.predict(X_test_sel)
        y_proba = self.stacked_model.predict_proba(X_test_sel)[:, 1]

        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))
        
        self.df = df
        self.is_trained = True
        print("✅ Model training completed!")
        
        # Store feature importance model for later plotting
        self.xgb_fs = xgb_fs
        
        return self
    
    def setup_faiss(self):
        """Setup FAISS for similarity search"""
        if not self.is_trained:
            raise ValueError("Train the model first using load_and_train()")
            
        print("🔧 Setting up FAISS similarity search...")
        
        # Prepare text embeddings
        self.df['text'] = self.df[self.model_features].apply(self.row_to_text, axis=1)
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        embeddings = self.embed_model.encode(
            self.df['text'].tolist(),
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')

        # Build FAISS index
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)
        print(f"✅ FAISS index built: {self.index.ntotal} patients indexed.")
        
        return self
    
    def row_to_text(self, row):
        """Convert patient data to text description"""
        return (
            f"Gender: {row['gender']}, Age: {row['age']}, "
            f"Pulse: {row['pulse_rate']}, BP: {row['systolic_bp']}/{row['diastolic_bp']}, "
            f"Glucose: {row['glucose']}, BMI: {row['bmi']:.1f}, "
            f"Family Diabetes: {row['family_diabetes']}, Hypertension: {row['hypertensive']}"
        )
    
    def predict_and_explain(self, patient_dict, k=5):
        """Predict for a single patient and find similar cases"""
        if not self.is_trained:
            raise ValueError("Train the model first using load_and_train()")
            
        # Prepare input
        input_df = pd.DataFrame([patient_dict])
        if 'gender' in input_df.columns and input_df.loc[0, 'gender'] in ('Male','Female'):
            input_df['gender'] = input_df['gender'].map({'Male':1,'Female':0})

        # Check for missing features
        missing = [c for c in self.model_features if c not in input_df.columns]
        if missing:
            raise ValueError(f"Input is missing required features: {missing}")

        # Prepare numeric input
        X_input = input_df[self.model_features].astype(float)
        X_scaled_input = self.scaler.transform(X_input)
        X_sel_input = self.selector.transform(X_scaled_input)

        # Prediction
        prob = float(self.stacked_model.predict_proba(X_sel_input)[0, 1])
        label = int(prob > 0.5)

        # Similarity search
        patient_row_series = input_df.iloc[0]
        input_text = self.row_to_text(patient_row_series)
        input_vec = self.embed_model.encode([input_text], convert_to_numpy=True, normalize_embeddings=True).astype('float32')

        # Query FAISS
        N_search = 50
        distances, indices = self.index.search(input_vec, N_search)
        nn_indices = indices[0].tolist()

        # Get neighbors
        neighbours = self.df.iloc[nn_indices].copy().reset_index(drop=True)

        # Predict for neighbors
        neigh_X = neighbours[self.model_features].astype(float)
        neigh_X_scaled = self.scaler.transform(neigh_X)
        neigh_X_sel = self.selector.transform(neigh_X_scaled)
        neigh_probs = self.stacked_model.predict_proba(neigh_X_sel)[:, 1]
        neigh_pred_labels = (neigh_probs > 0.5).astype(int)

        neighbours = neighbours.assign(
            neigh_pred_prob=neigh_probs,
            neigh_pred_label=neigh_pred_labels
        )

        # Filter neighbors by predicted label
        filtered = neighbours[neighbours['neigh_pred_label'] == label]
        if filtered.shape[0] >= k:
            similar_df_out = filtered.head(k).copy()
        else:
            gt_filtered = neighbours[neighbours['diabetic'] == label]
            if gt_filtered.shape[0] >= k:
                similar_df_out = gt_filtered.head(k).copy()
            else:
                similar_df_out = neighbours.head(k).copy()

        display_cols = ['age', 'glucose', 'bmi', 'systolic_bp', 'diastolic_bp', 'diabetic',
                       'neigh_pred_prob', 'neigh_pred_label']

        print(f"\n🧠 Prediction Probability: {prob:.4f}")
        print(f"📊 Classified as: {'Diabetic' if label == 1 else 'Non-Diabetic'}")
        print(f"🔍 Top {k} Similar Patients:\n")
        print(similar_df_out[display_cols].reset_index(drop=True))

        return {"prob": prob, "label": label, "similar": similar_df_out[display_cols].reset_index(drop=True)}
    
    def setup_gemini(self):
        """Setup Gemini API with proper error handling"""
        load_dotenv()
        GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
        
        if not GOOGLE_API_KEY:
            raise RuntimeError("""
❌ GOOGLE_API_KEY not found in .env file!

Please create a .env file with your Gemini API key:
1. Get API key from: https://aistudio.google.com/app/apikey
2. Create .env file in same directory
3. Add: GOOGLE_API_KEY=your_actual_key_here
""")
        
        try:
            genai.configure(api_key=GOOGLE_API_KEY)
            # Test the configuration
            models = genai.list_models()
            print("✅ Gemini API configured successfully")
            print(f"📋 Available models: {[model.name for model in models if 'gemini' in model.name]}")
            return True
        except Exception as e:
            print(f"❌ Gemini configuration failed: {e}")
            return False
    
    def predict_with_gemini(self, patient_dict, k=5, gemini_model="gemini-1.5-flash"):
        """Predict with Gemini explanation"""
        if not self.is_trained:
            raise ValueError("Train the model first using load_and_train()")
            
        # Setup Gemini
        if not hasattr(self, 'gemini_configured') or not self.gemini_configured:
            self.gemini_configured = self.setup_gemini()
            
        if not self.gemini_configured:
            print("❌ Skipping Gemini explanation due to configuration issues")
            return None
        
        # Get prediction
        out = self.predict_and_explain(patient_dict, k=k)
        prob = out["prob"]
        label = out["label"]
        similar_df = out["similar"]

        # Format patient text
        patient_df = pd.DataFrame([patient_dict])
        if isinstance(patient_df.loc[0, 'gender'], str):
            patient_df['gender'] = patient_df['gender'].map({'Male':1,'Female':0})
        patient_text = self.row_to_text(patient_df.iloc[0])

        pred_label_str = "Diabetic" if label == 1 else "Non-Diabetic"
        
        # Call Gemini
        llm_text = self.call_gemini_plain(patient_text, pred_label_str, prob, similar_df, k=k, model=gemini_model)

        print("\n" + "="*60)
        print("🎯 GEMINI EXPLANATION & CARE PLAN")
        print("="*60)
        if llm_text:
            print(llm_text)
        else:
            print("❌ Gemini returned no text. Possible issues:")
            print("   - API key invalid or expired")
            print("   - Model name incorrect")
            print("   - API quota exceeded")
            print("   - Network connectivity issues")
        return llm_text
    
    def call_gemini_plain(self, patient_text, pred_label, prob, similar_df, k=5, model="gemini-1.5-flash"):
        """Call Gemini API for explanation with better error handling"""
        def format_similar_text(similar_df, cols=None, max_k=5):
            if cols is None:
                cols = ['age','glucose','bmi','systolic_bp','diastolic_bp','diabetic']
            lines = []
            for i, row in enumerate(similar_df.head(max_k).itertuples(), start=1):
                vals = []
                for c in cols:
                    if c in similar_df.columns:
                        vals.append(f"{c}: {getattr(row, c)}")
                lines.append(f"{i}) " + ", ".join(vals))
            return "\n".join(lines)

        PROMPT_TEMPLATE = textwrap.dedent("""
        You are a concise, careful clinical assistant. Use the provided data and similar cases as grounding.
        Do NOT invent facts or specific treatment doses. If uncertain, recommend clinician follow-up.

        Patient summary:
        {patient_text}

        Model prediction:
        Label: {pred_label}    Probability: {prob:.2f}

        Top {k} similar cases (grounding):
        {similar_summaries}

        Task:
        1) In 3-5 short sentences, explain why the ML model predicted {pred_label}. Mention the top 2-3 drivers (e.g., glucose, BMI, family history, BP).
        2) Provide a concise, practical care plan with headings:
           - Immediate actions (48-72 hours)
           - Recommended diagnostic tests (what to order and why)
           - Lifestyle recommendations (diet, exercise, monitoring)
           - Red flags that should prompt urgent care
        3) Keep the language clear and non-alarming. Give plain readable text (no JSON).
        """).strip()

        similar_summaries = format_similar_text(similar_df, max_k=k)
        prompt = PROMPT_TEMPLATE.format(
            patient_text=patient_text,
            pred_label=pred_label,
            prob=prob,
            k=k,
            similar_summaries=similar_summaries
        )

        try:
            # Use the correct model name format
            if "gemini" in model:
                model_name = model
            else:
                model_name = f"models/{model}"
                
            gemini_model_instance = genai.GenerativeModel(model_name)
            response = gemini_model_instance.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"❌ Gemini API call failed: {e}")
            print(f"💡 Trying alternative model...")
            
            # Try with default model
            try:
                gemini_model_instance = genai.GenerativeModel("gemini-pro")
                response = gemini_model_instance.generate_content(prompt)
                return response.text.strip()
            except Exception as e2:
                print(f"❌ Fallback also failed: {e2}")
                return None
    
    def plot_feature_importance(self):
        """Plot feature importance with fixed palette warning"""
        if not hasattr(self, 'xgb_fs'):
            print("Feature importance not available. Train model first.")
            return
            
        print("📈 Plotting feature importance...")
        feature_importances = self.xgb_fs.get_booster().get_score(importance_type='gain')
        
        # Convert to pandas Series
        importance_series = pd.Series(feature_importances)
        
        # Map feature names
        importance_series.index = [self.model_features[int(i.replace('f', ''))] for i in importance_series.index]
        
        # Sort features by importance
        sorted_importance = importance_series.sort_values(ascending=False)

        # Plotting with fixed palette warning
        plt.figure(figsize=(10, 6))
        # Fix the palette warning by using hue parameter correctly
        sns.barplot(x=sorted_importance.values, y=sorted_importance.index, 
                   hue=sorted_importance.index, legend=False, palette='viridis')
        plt.title('Feature Importance (Gain) from XGBoost')
        plt.xlabel('Importance (Gain)')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.show()

def main():
    """Main function to demonstrate the workflow"""
    print("🩺 DIABETES PREDICTION SYSTEM")
    print("=" * 50)
    
    try:
        # Initialize and train the model
        predictor = DiabetesPredictor("Diabetes_Final_Data_V2.csv")
        predictor.load_and_train()
        predictor.setup_faiss()
        
        # Test patient 1
        print("\n" + "="*50)
        print("TEST PATIENT 1")
        print("="*50)
        
        patient1 = {
            'gender': 0,  # Female
            'age': 52,
            'pulse_rate': 82,
            'systolic_bp': 135,
            'diastolic_bp': 90,
            'glucose': 155,
            'height': 1.60,
            'weight': 72,
            'bmi': 28.6,
            'family_diabetes': 1,
            'hypertensive': 1,
            'family_hypertension': 1,
            'cardiovascular_disease': 0,
            'stroke': 0
        }
        
        predictor.predict_with_gemini(patient1, k=5)
        
        # Test patient 2
        print("\n" + "="*50)
        print("TEST PATIENT 2")
        print("="*50)
        
        patient2 = {
            'gender': 1,  # Male
            'age': 58,
            'pulse_rate': 88,
            'systolic_bp': 150,
            'diastolic_bp': 95,
            'glucose': 210,
            'height': 1.62,
            'weight': 94,
            'bmi': 32.9,
            'family_diabetes': 1,
            'hypertensive': 1,
            'family_hypertension': 1,
            'cardiovascular_disease': 0,
            'stroke': 0
        }
        
        predictor.predict_with_gemini(patient2, k=5)
        
        # Plot feature importance
        print("\n" + "="*50)
        print("FEATURE IMPORTANCE")
        print("="*50)
        predictor.plot_feature_importance()
        
    except Exception as e:
        print(f"❌ Error: {e}")
        print("\n💡 Troubleshooting tips:")
        print("1. Make sure your CSV file is in the same directory")
        print("2. Create .env file with: GOOGLE_API_KEY=your_actual_key")
        print("3. Get API key from: https://aistudio.google.com/app/apikey")
        print("4. Install dependencies: pip install -r requirements.txt")

if __name__ == "__main__":
    main()