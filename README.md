# Worker Productivity MLflow CI/CD Pipeline

[![MLflow CI/CD with Docker Hub](https://github.com/YOUR_USERNAME/Workflow-CI/actions/workflows/ci-mlflow-docker.yml/badge.svg)](https://github.com/YOUR_USERNAME/Workflow-CI/actions/workflows/ci-mlflow-docker.yml)

## 🎯 Overview

Automated CI/CD pipeline for Worker Productivity Classification using MLflow Project with Docker Hub integration.

## 🏗️ Project Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci-mlflow-docker.yml
├── MLProject/
│   ├── modelling.py
│   ├── conda.yaml
│   ├── MLProject
│   └── processed_data/
│       ├── data_train.csv
│       ├── data_validation.csv
│       ├── data_test.csv
│       ├── label_encoder.pkl
│       ├── feature_scaler.pkl
│       ├── label_mapping.csv
│       ├── feature_names.csv
│       └── preprocessing_summary.json
├── README.md
└── requirements.txt
```

## 🚀 Features

- ✅ **MLflow Project Integration**: Automated model training using MLflow Projects
- ✅ **DagsHub Integration**: Experiment tracking and model versioning
- ✅ **Docker Hub**: Automated Docker image building and pushing using `mlflow build-docker`
- ✅ **GitHub Actions**: Complete CI/CD pipeline with artifact management
- ✅ **Artifact Storage**: Automatic upload to GitHub Releases
- ✅ **Manual Trigger**: Workflow can be triggered manually via GitHub Actions

## 🔧 Setup Instructions

### 1. Repository Setup

1. Create a new public repository on GitHub named `Workflow-CI`
2. Clone the repository and add the project structure above
3. Push the initial code to the main branch

### 2. GitHub Secrets Configuration

Add the following secrets in your GitHub repository settings (`Settings > Secrets and variables > Actions`):

```bash
# DagsHub/MLflow Configuration
MLFLOW_TRACKING_URI=https://dagshub.com/YOUR_USERNAME/Worker-Productivity-MLflow.mlflow
DAGSHUB_USERNAME=your_dagshub_username
DAGSHUB_USER_TOKEN=your_dagshub_token

# Docker Hub Configuration
DOCKER_USERNAME=your_dockerhub_username
DOCKER_PASSWORD=your_dockerhub_password
DOCKER_REPO=your_dockerhub_username/worker-productivity-mlp
```

### 3. DagsHub Setup

1. Create a DagsHub account and repository
2. Generate a personal access token
3. Configure the repository for MLflow tracking

### 4. Docker Hub Setup

1. Create a Docker Hub account
2. Create a new repository (e.g., `worker-productivity-mlp`)
3. Generate an access token for GitHub Actions

## 🎯 Workflow Triggers

The CI/CD pipeline runs on:

- **Push** to `main` or `master` branches
- **Pull Request** to `main` or `master` branches  
- **Manual trigger** via GitHub Actions interface

## 🔄 Pipeline Steps

### 1. Environment Setup
- Checkout repository
- Setup Python 3.12.7
- Setup Conda environment
- Install MLflow and dependencies

### 2. MLflow Training
- Configure DagsHub authentication
- Run MLflow project: `mlflow run . --env-manager=conda`
- Log model, metrics, and artifacts

### 3. Docker Image Creation
- Setup Docker Buildx
- Login to Docker Hub
- Build Docker image using `mlflow models build-docker`
- Push image to Docker Hub

### 4. Artifact Management
- Upload artifacts to GitHub Actions
- Create GitHub Release (on main branch)
- Include all model files, reports, and configurations

## 🐳 Docker Usage

After successful pipeline execution, the Docker image will be available:

```bash
# Pull the image
docker pull YOUR_USERNAME/worker-productivity-mlp

# Run the container
docker run -p 8080:8080 YOUR_USERNAME/worker-productivity-mlp

# Or run with MLServer (if enabled)
docker run -p 8080:8080 -p 8081:8081 YOUR_USERNAME/worker-productivity-mlp
```

## 📊 MLflow Integration

### Model Registry
- Model name: `WorkerProductivityMLP_Basic`
- Automatic versioning on each training run
- Complete experiment tracking

### Artifacts Logged
- ✅ Trained model with signature
- ✅ Feature scaler (`scaler_basic.pkl`)
- ✅ Training reports and summaries
- ✅ Confusion matrix and metrics visualizations
- ✅ Environment configuration (`conda.yaml`)
- ✅ Deployment information

## 🎯 Advanced Features (Level 4)

This implementation achieves **Advanced (4 pts)** by including:

1. ✅ **MLflow Project structure** with proper `MLProject` file
2. ✅ **Complete CI/CD workflow** with GitHub Actions
3. ✅ **Artifact storage** in GitHub Releases
4. ✅ **Docker Hub integration** using `mlflow build-docker` function
5. ✅ **Automated Docker image building and pushing**

## 🔧 Troubleshooting

### Common Issues

1. **MLflow Authentication**
   - Ensure DagsHub tokens are correctly set in secrets
   - Check MLflow tracking URI format

2. **Docker Build Failures**
   - Pipeline includes fallback Docker build mechanism
   - Check Docker Hub credentials and repository permissions

3. **Model Registration**
   - Pipeline waits for model registration with retry logic
   - Includes alternative approaches if primary method fails

### Debug Steps

1. Check GitHub Actions logs for detailed error messages
2. Verify all secrets are properly configured
3. Ensure DagsHub repository is accessible
4. Check Docker Hub repository permissions

## 📈 Performance Metrics

The pipeline tracks comprehensive metrics:

- **Model Performance**: Accuracy, Precision, Recall, F1-Score
- **Training Metrics**: Loss, iterations, convergence status
- **Per-class Metrics**: Individual class performance
- **Confusion Matrix**: Detailed classification results

## 🔗 Links

- **Docker Hub**: https://hub.docker.com/r/silmiathqia/worker-productivity-mlp
- **DagsHub**: https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow
- **MLflow UI**: Access via DagsHub MLflow interface

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run the pipeline to ensure everything works
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**Note**: This project demonstrates advanced MLOps practices with complete automation from training to deployment using MLflow Projects and Docker Hub integration.
