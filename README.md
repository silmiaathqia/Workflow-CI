# Worker Productivity MLflow CI/CD Pipeline

[![MLflow CI/CD with Docker Hub](https://github.com/silmiaathqia/Workflow-CI/actions/workflows/ci-mlflow-docker.yml/badge.svg)](https://github.com/silmiaathqia/Workflow-CI/actions/workflows/ci-mlflow-docker.yml)
[![Python](https://img.shields.io/badge/python-3.12.7-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.19.0-orange.svg)](https://mlflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://hub.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## 🎯 Overview

Automated CI/CD pipeline for Worker Productivity Classification using **MLflow Project** with **Docker Hub** integration. This pipeline automatically trains, validates, and deploys a Neural Network model for predicting worker productivity levels (High, Medium, Low).

### Key Features
- ✅ **Automated ML Pipeline** with MLflow Project
- 🐳 **Docker containerization** for deployment
- 📊 **DagsHub integration** for experiment tracking
- 🔄 **GitHub Actions CI/CD**
- 📈 **Automated model versioning**
- 🏷️ **Automated releases** with artifacts

## 🏗️ Project Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci-mlflow-docker.yml          # Main CI/CD workflow
├── MLProject/
│   ├── modelling.py                      # Main training script
│   ├── conda.yaml                        # Environment dependencies
│   ├── MLproject                         # MLflow project configuration
│   └── processed_data/
│       ├── data_train.csv               # Training dataset
│       ├── data_validation.csv          # Validation dataset
│       ├── data_test.csv                # Test dataset
│       ├── label_encoder.pkl            # Label encoder
│       ├── feature_scaler.pkl           # Feature scaler
│       ├── label_mapping.csv            # Label mappings
│       ├── feature_names.csv            # Feature names
│       └── preprocessing_summary.json   # Preprocessing info
├── README.md
├── requirements.txt                     # Python dependencies
└── conda.yaml                          # Main conda environment
```

## 🚀 Quick Start

### Prerequisites
- Python 3.12.7
- MLflow 2.19.0
- Docker (for containerization)
- DagsHub account (for experiment tracking)

### Local Development
```bash
# Clone repository
git clone https://github.com/silmiaathqia/Workflow-CI.git
cd Workflow-CI

# Create conda environment
conda env create -f conda.yaml
conda activate mlflow-env

# Run training locally
cd MLProject
python modelling.py --data_path processed_data --experiment_name "Worker_Productivity_Local"
```

### Docker Usage
```bash
# Pull latest image
docker pull your-dockerhub-username/worker-productivity-mlp

# Run container
docker run -p 8080:8080 your-dockerhub-username/worker-productivity-mlp
```

## 🔧 Model Architecture

**Multi-Layer Perceptron (MLP) Classifier**
- **Hidden Layers**: (128, 64, 32) neurons
- **Activation**: ReLU
- **Solver**: Adam optimizer
- **Regularization**: L2 (α=0.001)
- **Early Stopping**: Enabled
- **Classes**: High, Medium, Low productivity

## 📊 Pipeline Workflow

1. **Data Validation** - Verify all required CSV files exist
2. **Environment Setup** - Create conda environment with dependencies
3. **Model Training** - Train MLP model with MLflow tracking
4. **Evaluation** - Generate metrics, confusion matrix, and reports
5. **Docker Build** - Create containerized application
6. **Deployment** - Push to Docker Hub
7. **Release** - Create GitHub release with artifacts

## 🎛️ Configuration

### Required GitHub Secrets
```
MLFLOW_TRACKING_URI     # DagsHub MLflow URI
DAGSHUB_USERNAME        # DagsHub username
DAGSHUB_USER_TOKEN      # DagsHub access token
DOCKER_USERNAME         # Docker Hub username
DOCKER_PASSWORD         # Docker Hub password/token
DOCKER_REPO            # Docker repository name
```

### MLflow Project Parameters
```yaml
data_path: "processed_data"              # Path to training data
experiment_name: "Worker_Productivity_Classification_Sklearn"
```

## 📈 Model Performance

The pipeline automatically tracks:
- **Accuracy**: Overall classification accuracy
- **Precision/Recall/F1**: Per-class and weighted metrics
- **Confusion Matrix**: Visual classification results
- **Training Loss**: Model convergence monitoring

## 🐳 Docker Deployment

The pipeline creates a production-ready Docker image with:
- Python 3.12.7 slim base
- All ML dependencies pre-installed
- Non-root user for security
- Port 8080 exposed
- Automatic model serving capability

## 📋 Generated Artifacts

Each pipeline run produces:
- `*.pkl` - Trained models and scalers
- `*.txt` - Classification reports and summaries
- `*.json` - Configuration and deployment info
- `*.png` - Visualization plots
- `Dockerfile` - Container configuration

## 🔗 Integration Links

- **MLflow Tracking**: [DagsHub Project](https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow)
- **Docker Hub**: [Container Repository](https://hub.docker.com/r/your-username/worker-productivity-mlp)
- **CI/CD Pipeline**: [GitHub Actions](https://github.com/silmiaathqia/Workflow-CI/actions)

## 🛠️ Troubleshooting

### Common Issues
1. **Data Files Missing**: Ensure all CSV files are in `processed_data/`
2. **MLflow Connection**: Check DagsHub credentials in secrets
3. **Docker Build Failed**: Verify Docker Hub credentials
4. **Model Training Error**: Check data format and feature consistency

### Debug Commands
```bash
# Check repository structure
find . -name "*.csv" -o -name "*.py" -o -name "*.yml"

# Validate MLflow connection
python -c "import mlflow; print(mlflow.get_tracking_uri())"

# Test Docker image locally
docker build -t test-image .
docker run --rm test-image python modelling.py --help
```

## 📝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MLflow** for experiment tracking and model management
- **DagsHub** for collaborative ML platform
- **scikit-learn** for machine learning algorithms
- **GitHub Actions** for CI/CD automation

---

**Note**: This project demonstrates advanced MLOps practices with complete automation from training to deployment using MLflow Projects and Docker Hub integration.
