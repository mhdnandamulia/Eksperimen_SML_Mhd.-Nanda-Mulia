
"""
AUTOMATED PREPROCESSING SCRIPT FOR HEART DISEASE DATASET
Script ini melakukan preprocessing otomatis berdasarkan hasil EDA
Level: Skilled (3 pts)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_heart_data():
    """Load dataset heart disease."""
    df = pd.read_csv('heart.csv')
    print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    return df

def preprocess_heart_data(df, test_size=0.2, random_state=42):
    """
    Main preprocessing function untuk dataset heart disease.
    Mengembalikan data yang siap untuk training model.
    """
    print("=" * 60)
    print("🚀 STARTING AUTOMATED PREPROCESSING")
    print("=" * 60)
    
    # ========== STEP 1: DATA CLEANING ==========
    print("\n1. 🧹 DATA CLEANING")
    print("-" * 30)
    
    # Backup data asli
    df_original = df.copy()
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates()
    removed_count = initial_count - len(df)
    if removed_count > 0:
        print(f"   Removed {removed_count} duplicates ({removed_count/initial_count*100:.1f}%)")
    else:
        print("   No duplicates found")
    
    # Check missing values
    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"   Found {missing} missing values")
        # Simple imputation for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    else:
        print("   No missing values found")
    
    # ========== STEP 2: DEFINE FEATURE TYPES ==========
    print("\n2. 🏷️ DEFINING FEATURE TYPES")
    print("-" * 30)
    
    # Define columns by type
    TARGET_COL = 'target'
    
    # Categorical features (dengan strategi berbeda)
    cat_onehot = ['cp', 'thal', 'slope']  # > 2 categories
    cat_label = ['sex', 'fbs', 'exang', 'restecg']  # binary atau ordinal
    
    # Numerical features
    numerical = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    
    # Special feature: 'ca' (bisa dianggap categorical atau numerical)
    # Dataset heart: ca adalah nilai 0-3
    df['ca'] = df['ca'].astype(int)  # Ensure integer type
    
    print(f"   Target column: {TARGET_COL}")
    print(f"   Numerical features ({len(numerical)}): {', '.join(numerical)}")
    print(f"   Categorical (OneHot) ({len(cat_onehot)}): {', '.join(cat_onehot)}")
    print(f"   Categorical (Label) ({len(cat_label)}): {', '.join(cat_label)}")
    print(f"   Special feature: ca (treated as categorical)")
    
    # ========== STEP 3: OUTLIER HANDLING ==========
    print("\n3. 📊 OUTLIER HANDLING (Robust Scaling)")
    print("-" * 30)
    
    # Gunakan RobustScaler untuk numerical features (lebih robust terhadap outlier)
    robust_scaler = RobustScaler()
    df[numerical] = robust_scaler.fit_transform(df[numerical])
    print("   Applied RobustScaler to numerical features")
    
    # ========== STEP 4: FEATURE ENCODING ==========
    print("\n4. 🔠 FEATURE ENCODING")
    print("-" * 30)
    
    # Label Encoding untuk categorical binary/ordinal
    label_encoders = {}
    for col in cat_label:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
        print(f"   Label encoded: {col}")
    
    # Untuk 'ca' (meskipun numerical, treat as categorical karena limited values 0-3)
    le_ca = LabelEncoder()
    df['ca'] = le_ca.fit_transform(df['ca'])
    label_encoders['ca'] = le_ca
    print(f"   Label encoded: ca")
    
    # OneHot Encoding untuk categorical dengan >2 categories
    df = pd.get_dummies(df, columns=cat_onehot, prefix=cat_onehot, drop_first=True)
    print(f"   OneHot encoded: {', '.join(cat_onehot)}")
    
    # ========== STEP 5: DATA SPLITTING ==========
    print("\n5. ✂️ DATA SPLITTING")
    print("-" * 30)
    
    # Pisahkan features dan target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    
    # Stratified split (penting untuk balanced dataset)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Pertahankan distribusi target
    )
    
    print(f"   Training set: {X_train.shape[0]} samples")
    print(f"   Testing set : {X_test.shape[0]} samples")
    print(f"   Features    : {X_train.shape[1]} features")
    
    # ========== STEP 6: FINAL SCALING ==========
    print("\n6. ⚖️ FINAL FEATURE SCALING (StandardScaler)")
    print("-" * 30)
    
    # StandardScaler untuk semua features (setelah encoding)
    final_scaler = StandardScaler()
    
    # Fit hanya pada training data
    X_train_scaled = final_scaler.fit_transform(X_train)
    X_test_scaled = final_scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("   Applied StandardScaler to all features")
    
    # ========== STEP 7: SAVE PROCESSED DATA ==========
    print("\n7. 💾 SAVING PROCESSED DATA")
    print("-" * 30)
    
    # Save processed data
    X_train.to_csv('X_train_processed.csv', index=False)
    X_test.to_csv('X_test_processed.csv', index=False)
    
    y_train_df = pd.DataFrame(y_train, columns=[TARGET_COL])
    y_test_df = pd.DataFrame(y_test, columns=[TARGET_COL])
    y_train_df.to_csv('y_train_processed.csv', index=False)
    y_test_df.to_csv('y_test_processed.csv', index=False)
    
    # Save full dataset (with split indicator)
    train_full = pd.concat([X_train, y_train_df], axis=1)
    test_full = pd.concat([X_test, y_test_df], axis=1)
    train_full['data_split'] = 'train'
    test_full['data_split'] = 'test'
    
    full_processed = pd.concat([train_full, test_full], axis=0)
    full_processed.to_csv('heart_disease_fully_processed.csv', index=False)
    
    print("   Saved files:")
    print("   • X_train_processed.csv")
    print("   • X_test_processed.csv")
    print("   • y_train_processed.csv")
    print("   • y_test_processed.csv")
    print("   • heart_disease_fully_processed.csv")
    
    # ========== STEP 8: SUMMARY ==========
    print("\n" + "=" * 60)
    print("📊 PREPROCESSING SUMMARY")
    print("=" * 60)
    
    print(f"\n📈 DATA BALANCE (Training Set):")
    train_class_dist = y_train.value_counts()
    for cls, count in train_class_dist.items():
        pct = count / len(y_train) * 100
        cls_name = "No Disease" if cls == 0 else "Has Disease"
        print(f"   {cls_name}: {count} samples ({pct:.1f}%)")
    
    print(f"\n🔢 FEATURE COUNT:")
    print(f"   Original features: {df_original.shape[1] - 1}")
    print(f"   After preprocessing: {X_train.shape[1]}")
    
    print(f"\n📁 OUTPUT FILES CREATED:")
    print("   1. X_train_processed.csv - Training features")
    print("   2. X_test_processed.csv  - Testing features")
    print("   3. y_train_processed.csv - Training labels")
    print("   4. y_test_processed.csv  - Testing labels")
    print("   5. heart_disease_fully_processed.csv - Full dataset with split flag")
    
    print("\n" + "=" * 60)
    print("✅ PREPROCESSING COMPLETE - DATA READY FOR MODEL TRAINING")
    print("=" * 60)
    
    return X_train, X_test, y_train, y_test, final_scaler, label_encoders

def main():
    """Main function to run the preprocessing pipeline."""
    try:
        print("🔧 HEART DISEASE PREPROCESSING PIPELINE")
        print("Level: Skilled (3 pts)")
        print("Dataset: Heart Disease UCI")
        
        # Load data
        df = load_heart_data()
        
        # Run preprocessing
        results = preprocess_heart_data(df)
        
        print("\n🎉 Pipeline executed successfully!")
        print("\n📝 Usage in modeling:")
        print("   from automate_heart_preprocessing import preprocess_heart_data")
        print("   X_train, X_test, y_train, y_test, scaler, encoders = preprocess_heart_data(df)")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Error in preprocessing pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # Run the pipeline
    main()
