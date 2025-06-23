import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import yaml 
import pickle
import json
import dagshub
import os
import warnings
import argparse
warnings.filterwarnings('ignore')

def setup_mlflow_dagshub():
    """Setup MLflow dengan DagsHub integration dan autolog support"""
    # Setup DagsHub (sesuaikan dengan username dan repo Anda)
    dagshub.init(repo_owner='silmiaathqia', repo_name='Worker-Productivity-MLflow', mlflow=True)
    
    # Set MLflow tracking URI - gunakan DagsHub untuk online tracking
    # Hapus override ke localhost agar bisa sync ke DagsHub
    # mlflow.set_tracking_uri("http://127.0.0.1:5000")  # COMMENTED OUT
    
    # Set experiment name yang konsisten dengan tuning
    experiment_name = "Worker_Productivity_Classification_Sklearn"
    mlflow.set_experiment(experiment_name)
    
    print("MLflow dan DagsHub berhasil dikonfigurasi!")
    print(f"Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"Experiment: {experiment_name}")
    
    return experiment_name

def load_processed_data(data_path="processed_data"):
    """Load preprocessed data from kriteria 1"""
    try:
        # Load data
        train_data = pd.read_csv(f'{data_path}/data_train.csv')
        val_data = pd.read_csv(f'{data_path}/data_validation.csv')
        test_data = pd.read_csv(f'{data_path}/data_test.csv')
        
        # Separate features and target
        X_train = train_data.drop('productivity_label_encoded', axis=1)
        y_train = train_data['productivity_label_encoded']
        
        X_val = val_data.drop('productivity_label_encoded', axis=1)
        y_val = val_data['productivity_label_encoded']
        
        X_test = test_data.drop('productivity_label_encoded', axis=1)
        y_test = test_data['productivity_label_encoded']
        
        print(f"Data berhasil dimuat!")
        print(f"Shape - Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
        print(f"Kelas target: {sorted(y_train.unique())}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def create_mlp_model():
    """Create MLP model using sklearn for worker productivity classification"""
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers with 128, 64, 32 neurons
        activation='relu',
        solver='adam',
        alpha=0.001,  # L2 regularization
        batch_size='auto',
        learning_rate='constant',
        learning_rate_init=0.001,
        max_iter=500,
        shuffle=True,
        random_state=42,
        tol=1e-4,
        validation_fraction=0.1,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        n_iter_no_change=10,  # Early stopping
        early_stopping=True,
        verbose=True
    )
    
    return model

def plot_confusion_matrix(y_true, y_pred, class_names=['High', 'Low', 'Medium']):
    """Plot confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - Basic Model')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix_basic.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'confusion_matrix_basic.png'

def plot_metrics_comparison(metrics_dict):
    """Plot metrics comparison"""
    metrics = list(metrics_dict.keys())
    values = list(metrics_dict.values())
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(metrics, values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    plt.title('Model Performance Metrics - Basic Model')
    plt.ylabel('Score')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('metrics_comparison_basic.png', dpi=300, bbox_inches='tight')
    plt.close()
    return 'metrics_comparison_basic.png'

def create_environment_file():
    """Create conda environment file"""
    conda_env = {
        "channels": ["conda-forge"],
        "dependencies": [
            "python=3.12.7",  # Updated to match your actual Python version
            "pip",
            {
                "pip": [
                    "mlflow==2.19.0",
                    "scikit-learn==1.5.2",
                    "pandas==2.3.0",
                    "numpy==1.26.4",
                    "matplotlib==3.10.3",
                    "seaborn==0.13.2",
                    "dagshub==0.5.10",
                    "PyYAML==6.0.2"
                ]
            }
        ],
        "name": "mlflow-env"
    }
    
    with open('conda.yaml', 'w') as f:
        import yaml
        yaml.dump(conda_env, f)
    
    return 'conda.yaml'

def cleanup_temporary_files():
    """
    Clean up temporary files created during training
    Fungsi ini untuk:
    1. Menghemat storage space
    2. Membersihkan workspace dari file temporary
    3. Hanya menyimpan file yang diperlukan untuk deployment
    """
    temp_files = [
        'confusion_matrix_basic.png',
        'metrics_comparison_basic.png'
    ]
    
    cleaned_files = []
    for file in temp_files:
        if os.path.exists(file):
            try:
                os.remove(file)
                cleaned_files.append(file)
            except Exception as e:
                print(f"Could not remove {file}: {e}")
    
    if cleaned_files:
        print(f"\nCleaned up temporary files: {cleaned_files}")
        print("Note: Files are already saved to MLflow, safe to remove locally.")
    else:
        print("\nNo temporary files to clean up.")

def create_deployment_info():
    """Create deployment information file"""
    deployment_info = {
        "model_name": "Basic MLP Worker Productivity Classifier",
        "model_type": "sklearn.neural_network.MLPClassifier",
        "version": "1.0",
        "created_date": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "features": {
            "architecture": "(128, 64, 32)",
            "activation": "relu",
            "solver": "adam",
            "regularization": "L2 (alpha=0.001)",
            "early_stopping": True
        },
        "deployment": {
            "required_files": [
                "model (from MLflow)",
                "scaler_basic.pkl",
                "conda.yaml"
            ],
            "mlflow_model_name": "WorkerProductivityMLP_Basic",
            "prediction_classes": ["High", "Low", "Medium"]
        },
        "performance_note": "Check MLflow experiment for detailed metrics"
    }
    
    with open('deployment_info_basic.json', 'w') as f:
        json.dump(deployment_info, f, indent=2)
    
    return 'deployment_info_basic.json'

def train_basic_model(data_path="processed_data", experiment_name="Worker_Productivity_Classification_Sklearn"):
    """Train basic MLP model dengan MLflow autolog"""
    
    # Setup MLflow dan DagsHub
    setup_mlflow_dagshub()
    mlflow.set_experiment(experiment_name)
    
    # Load data
    data = load_processed_data(data_path)
    if data is None:
        return
    
    X_train, X_val, X_test, y_train, y_val, y_test = data
    
    # Combine train and validation for sklearn MLPClassifier (it handles validation internally)
    X_train_full = pd.concat([X_train, X_val])
    y_train_full = pd.concat([y_train, y_val])
    
    # Scale the features (important for neural networks)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_full)
    X_test_scaled = scaler.transform(X_test)
    
    print("Data scaling completed!")
    print(f"Training set shape: {X_train_scaled.shape}")
    print(f"Test set shape: {X_test_scaled.shape}")
    
    # Create DagsHub info file
    dagshub_info = {
        "project_name": "Worker Productivity Classification",
        "model_type": "Basic MLP (Sklearn)",
        "dagshub_url": "https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow",
        "dataset_info": {
            "total_samples": len(X_train_full) + len(X_test),
            "features": X_train_full.shape[1],
            "classes": ["High", "Low", "Medium"]
        }
    }
    
    with open('dagshub_info.json', 'w') as f:
        json.dump(dagshub_info, f, indent=2)

    # Run experiment dengan MLflow autolog
    with mlflow.start_run(run_name="Basic_MLP_Autolog"):
        print("\n" + "="*60)
        print("STARTING BASIC MLP TRAINING WITH AUTOLOG")
        print("="*60)
        
        # ENABLE autolog untuk automatic logging
        mlflow.sklearn.autolog()
        
        # Create and train model
        model = create_mlp_model()
        
        print("\nModel Configuration:")
        print(f"Hidden layers: {model.hidden_layer_sizes}")
        print(f"Activation: {model.activation}")
        print(f"Solver: {model.solver}")
        print(f"Alpha (L2 reg): {model.alpha}")
        print(f"Max iterations: {model.max_iter}")
        
        # Train model - autolog akan mencatat parameter dan metrics secara otomatis
        print("\nStarting training with autolog enabled...")
        model.fit(X_train_scaled, y_train_full)
        
        print(f"\nTraining completed!")
        print(f"Number of iterations: {model.n_iter_}")
        print(f"Final loss: {model.loss_:.6f}")
        
        # Make predictions
        y_pred = model.predict(X_test_scaled)
        y_pred_proba = model.predict_proba(X_test_scaled)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print(f"\nTest Results:")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test F1-Score: {f1:.4f}")
        
        # Classification report
        print("\nDetailed Classification Report:")
        class_names = ['High', 'Low', 'Medium']
        class_report = classification_report(y_test, y_pred, target_names=class_names)
        print(class_report)
        
        # Save classification report
        with open('classification_report_basic.txt', 'w') as f:
            f.write("Basic MLP Model - Classification Report\n")
            f.write("=" * 45 + "\n")
            f.write(class_report)
        
        # Create visualizations
        cm_file = plot_confusion_matrix(y_test, y_pred, class_names)
        metrics_file = plot_metrics_comparison({
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1
        })
        
        # Create environment file
        env_file = create_environment_file()
        
        # Create deployment info
        deployment_file = create_deployment_info()
        
        # Create model summary
        with open('model_summary_basic.txt', 'w') as f:
            f.write("Basic MLP Model Summary\n")
            f.write("=" * 30 + "\n")
            f.write(f"Architecture: {model.hidden_layer_sizes}\n")
            f.write(f"Activation: {model.activation}\n")
            f.write(f"Solver: {model.solver}\n")
            f.write(f"L2 Regularization: {model.alpha}\n")
            f.write(f"Training iterations: {model.n_iter_}\n")
            f.write(f"Final loss: {model.loss_:.6f}\n")
            f.write(f"Convergence: {'Yes' if model.n_iter_ < model.max_iter else 'No'}\n")
            f.write("\nPerformance Metrics:\n")
            f.write(f"Test Accuracy: {accuracy:.4f}\n")
            f.write(f"Test Precision: {precision:.4f}\n")
            f.write(f"Test Recall: {recall:.4f}\n")
            f.write(f"Test F1-Score: {f1:.4f}\n")
        
        # Save scaler
        with open('scaler_basic.pkl', 'wb') as f:
            pickle.dump(scaler, f)
        
        # Log additional artifacts to MLflow (minimal, focus on autolog)
        mlflow.log_artifact('scaler_basic.pkl')
        mlflow.log_artifact(deployment_file)
        
        # Log tags untuk identifikasi
        mlflow.set_tags({
            "model_type": "basic_mlp",
            "framework": "sklearn",
            "version": "1.0",
            "dataset": "worker_productivity",
            "author": "silmiathqia",
            "experiment_group": "baseline",
            "logging_type": "autolog_focused"
        })
        
        print("\n" + "="*60)
        print("AUTOLOG ARTIFACTS (AUTOMATIC):")
        print("="*60)
        print("✓ Model: Logged automatically by autolog")
        print("✓ Parameters: Logged automatically by autolog")
        print("✓ Metrics: Logged automatically by autolog")
        print("✓ Training curves: Logged automatically by autolog")
        
        print("\n" + "="*60)
        print("MANUAL ARTIFACTS (ESSENTIAL ONLY):")
        print("="*60)
        print("✓ Scaler: scaler_basic.pkl (deployment essential)")
        print("✓ Deployment Info: deployment_info_basic.json")
        print("✓ Local files: Environment, reports, visualizations")
        
        print("\n" + "="*60)
        print("AUTOLOG SUMMARY:")
        print("="*60)
        print("✓ mlflow.sklearn.autolog() ENABLED")
        print("✓ Parameters logged automatically to DagsHub")
        print("✓ Metrics logged automatically to DagsHub")
        print("✓ Model logged automatically to DagsHub")
        print("✓ Manual logging minimized (focus on autolog)")
        print("✓ Tracking URI: DagsHub online")
        
        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETED!")
        print("="*60)
        print(f"Check MLflow UI at: {mlflow.get_tracking_uri()}")
        print("DagsHub integration active for online sync")
        
         # Return metrics for summary
        metrics = {
            'test_accuracy': accuracy,
            'test_precision_weighted': precision,
            'test_recall_weighted': recall,
            'test_f1_score_weighted': f1,
            'n_iterations': model.n_iter_,
            'convergence_achieved': model.n_iter_ < model.max_iter
        }
		
		# Simpan model lokal untuk inference/exporter
		with open('model_basic.pkl', 'wb') as f:
			pickle.dump(model, f)
		print("✓ Model saved locally as model_basic.pkl")

		# Registrasi model ke MLflow Registry (agar bisa serve via models:/...)
		mlflow.sklearn.log_model(
			sk_model=model,
			artifact_path="model",
			registered_model_name="WorkerProductivityMLP_Basic"
		)
		print("✓ Model registered to MLflow Registry as 'WorkerProductivityMLP_Basic'")

        return model, scaler, metrics

def make_prediction(model, scaler, sample_data):
    """Make prediction on new data"""
    # Scale the input data
    sample_scaled = scaler.transform(sample_data)
    
    # Make prediction
    prediction = model.predict(sample_scaled)
    prediction_proba = model.predict_proba(sample_scaled)
    
    class_names = ['High', 'Low', 'Medium']
    
    print(f"Prediction: {class_names[prediction[0]]}")
    print("Prediction probabilities:")
    for i, prob in enumerate(prediction_proba[0]):
        print(f"  {class_names[i]}: {prob:.4f}")
    
    return prediction, prediction_proba

def main():
    """Main function untuk MLproject entry point"""
    parser = argparse.ArgumentParser(description='Worker Productivity Classification Training')
    parser.add_argument('--data_path', type=str, default='processed_data', 
                       help='Path to processed data directory')
    parser.add_argument('--experiment_name', type=str, 
                       default='Worker_Productivity_Classification_Sklearn',
                       help='MLflow experiment name')
    parser.add_argument('--cleanup', action='store_true', 
                       help='Clean up temporary files after training')
    
    args = parser.parse_args()
    
    print("KLASIFIKASI PRODUKTIVITAS PEKERJA - AUTOLOG FOCUSED MODEL")
    print("=" * 75)
    print("Features:")
    print("  ✓ MLflow Autolog ENABLED (Primary Focus)")
    print("  ✓ Automatic Parameter & Metrics Logging")
    print("  ✓ DagsHub Integration ACTIVE")
    print("  ✓ Minimal Manual Logging")
    print("  ✓ Online Tracking & Sync")
    print("  ✓ Essential Deployment Artifacts Only")
    print("=" * 75)
    
    # Train model
    results = train_basic_model(args.data_path, args.experiment_name)
    
    if results:
        model, scaler, metrics = results
        
        print(f"\n" + "="*60)
        print("AUTOLOG TRAINING SUMMARY:")
        print("="*60)
        print(f"Final Accuracy: {metrics['test_accuracy']:.4f}")
        print(f"Final F1-Score: {metrics['test_f1_score_weighted']:.4f}")
        print(f"Training iterations: {metrics['n_iterations']}")
        print(f"Convergence: {'Yes' if metrics['convergence_achieved'] else 'No'}")
        print(f"Autolog: ✓ ENABLED & FOCUSED")
        
        # Cleanup if requested
        if args.cleanup:
            cleanup_temporary_files()
            print("✓ Cleanup completed.")
        else:
            print("✓ Files kept locally.")
        
        print(f"\n" + "="*60)
        print("NEXT STEPS:")
        print("="*60)
        print("1. Check DagsHub at: https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow")
        print("2. Review automatically logged parameters and metrics")
        print("3. Use deployment_info_basic.json for deployment guidance")
        print("4. Autolog handled most artifacts automatically")
        
        print("\n✅ Autolog-focused MLP model training completed successfully!")
        print("✅ MLflow autolog ENABLED - minimal manual logging applied!")
        print("✅ DagsHub integration ACTIVE - results will sync online!")
        
    else:
        print("\n❌ Training failed. Please check the error messages above.")
        print("Common issues:")
        print("- Data files not found in 'processed_data/' directory")
        print("- DagsHub authentication issues")
        print("- Missing dependencies")
        
    print(f"\n{'='*75}")
    print("PROGRAM COMPLETED")
    print(f"{'='*75}")

if __name__ == "__main__":
    main()
