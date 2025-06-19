# Worker Productivity MLflow CI/CD Pipeline

[![MLflow CI/CD](https://github.com/silmiaathqia/Workflow-CI/actions/workflows/ci-mlflow.yml/badge.svg)](https://github.com/silmiaathqia/Workflow-CI/actions/workflows/ci-mlflow.yml) [![Python](https://img.shields.io/badge/python-3.12.7-blue.svg)](https://www.python.org/downloads/) [![MLflow](https://img.shields.io/badge/MLflow-2.19.0-orange.svg)](https://mlflow.org/) [![Docker](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Pipeline CI/CD otomatis untuk klasifikasi produktivitas pekerja menggunakan **MLflow Project** dan **GitHub Actions**. System ini secara otomatis melatih, memvalidasi, dan menyebarkan model Neural Network untuk memprediksi tingkat produktivitas pekerja (High, Medium, Low) dengan integrasi penuh ke **DagsHub** dan **Docker Hub**.

## Project Structure

```
Workflow-CI/
├── .github/
│   └── workflows/
│       └── ci-mlflow.yml          # Main CI/CD workflow
├── MLProject/
│   ├── modelling.py               # Main training script
│   ├── conda.yaml                 # Environment dependencies
│   ├── MLproject                  # MLflow project configuration
│   └── processed_data/
│       ├── data_train.csv         # Training dataset
│       ├── data_validation.csv    # Validation dataset
│       └── data_test.csv          # Test dataset
├── README.md
└── conda.yaml                    # Main conda environment
```

## Model Architecture

**Basic MLP Classifier (Scikit-learn)**
- Hidden layers: (128, 64, 32)
- Activation: ReLU
- Solver: Adam optimizer
- L2 regularization: alpha=0.001
- Early stopping: enabled
- Target classes: High, Low, Medium
- Framework: scikit-learn 1.5.2

## Quick Start

### Prerequisites

- Python 3.12.7
- Conda environment manager
- GitHub account dengan Actions enabled
- DagsHub account untuk MLflow tracking
- Docker Hub account untuk deployment

### Local Development

1. **Clone repository**
   ```bash
   git clone https://github.com/silmiaathqia/Workflow-CI.git
   cd Workflow-CI
   ```

2. **Setup environment**
   ```bash
   conda env create -f conda.yaml
   conda activate mlflow-env
   ```

3. **Prepare data**
   Pastikan file data berada di `MLProject/processed_data/`:
   - `data_train.csv`
   - `data_validation.csv` 
   - `data_test.csv`

4. **Run training locally**
   ```bash
   cd MLProject
   python modelling.py --data_path processed_data --experiment_name Worker_Productivity_Local
   ```

### MLflow Project Execution

```bash
cd MLProject
mlflow run . \
  --env-manager=conda \
  --experiment-name="Worker_Productivity_Classification_Sklearn" \
  -P data_path="processed_data" \
  -P experiment_name="Worker_Productivity_Classification_Sklearn"
```

## CI/CD Pipeline Features

### Automated Workflow Triggers
- Push ke branch `main` atau `master`
- Pull request ke branch utama
- Manual dispatch melalui GitHub Actions

### Pipeline Steps

1. **Environment Setup**
   - Setup Conda dengan Python 3.12.7
   - Install dependencies dari `conda.yaml`
   - Configure MLflow dan DagsHub authentication

2. **Data Validation**
   - Verify struktur data dan required files
   - Validate data integrity
   - Check feature compatibility

3. **Model Training**
   - Automatic MLflow experiment creation
   - Neural network training dengan early stopping
   - Comprehensive metrics logging
   - Model registration ke MLflow Model Registry

4. **Artifact Generation**
   - Model files (`.pkl`)
   - Performance reports (`.txt`)
   - Visualizations (`.png`)
   - Deployment configurations (`.json`, `.yaml`)

5. **Docker Deployment**
   - Automatic Docker image build dengan MLflow
   - Push ke Docker Hub
   - Tagged dengan timestamp dan latest

6. **Release Management**
   - Automatic GitHub release creation
   - Artifact packaging dan upload
   - Deployment documentation

## Configuration

### GitHub Secrets Required

```yaml
# MLflow & DagsHub
MLFLOW_TRACKING_URI: "https://dagshub.com/username/repo.mlflow"
DAGSHUB_USERNAME: "your-dagshub-username"
DAGSHUB_USER_TOKEN: "your-dagshub-token"

# Docker Hub
DOCKER_USERNAME: "your-docker-username"
DOCKER_PASSWORD: "your-docker-password"
DOCKER_REPO: "your-docker-username/worker-productivity-mlp"
```

### Environment Variables

```bash
export MLFLOW_TRACKING_URI="https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow.mlflow"
export MLFLOW_TRACKING_USERNAME="your-username"
export MLFLOW_TRACKING_PASSWORD="your-token"
```

## Dependencies

### Core Dependencies
- **Python**: 3.12.7
- **MLflow**: 2.19.0 (experiment tracking & model registry)
- **scikit-learn**: 1.5.2 (machine learning)
- **pandas**: 2.3.0 (data manipulation)
- **numpy**: 1.26.4 (numerical computing)

### Visualization & Reporting
- **matplotlib**: 3.10.3
- **seaborn**: 0.13.2

### Integration & Deployment
- **dagshub**: 0.5.10 (MLflow hosting)
- **cloudpickle**: 3.1.1 (model serialization)
- **PyYAML**: 6.0.2 (configuration)
- **psutil**: 7.0.0 (system monitoring)

## Model Performance Metrics

Pipeline secara otomatis menghitung dan log:
- **Accuracy**: Overall prediction accuracy
- **Precision**: Per-class dan weighted average
- **Recall**: Per-class dan weighted average
- **F1-Score**: Harmonic mean precision dan recall
- **Confusion Matrix**: Detailed classification results

## Generated Artifacts

### Training Artifacts
- `scaler_basic.pkl` - Feature scaler untuk preprocessing
- `classification_report_basic.txt` - Detailed performance report
- `model_summary_basic.txt` - Model architecture summary
- `confusion_matrix_basic.png` - Visual confusion matrix
- `metrics_comparison_basic.png` - Performance metrics visualization

### Deployment Artifacts
- `deployment_info_basic.json` - Deployment configuration
- `dagshub_info.json` - DagsHub project information
- `conda.yaml` - Environment specification
- `docker_info.json` - Docker build information

## Docker Deployment

### Pull dan Run Container

```bash
# Pull latest image
docker pull your-docker-username/worker-productivity-mlp:latest

# Run container
docker run -p 8080:8080 your-docker-username/worker-productivity-mlp:latest
```

### Health Check

```bash
curl http://localhost:8080/health
```

## MLflow Integration

### Tracking Dashboard
- **URL**: https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow
- **Experiments**: Worker_Productivity_Classification_Sklearn
- **Models**: WorkerProductivityMLP_Basic

### Model Registry
- Model versioning otomatis
- Staging dan production stages
- Model lineage tracking
- Performance comparison

## Monitoring & Logging

### CI/CD Pipeline Monitoring
- GitHub Actions workflow status
- Build logs dan error reporting
- Artifact generation tracking
- Deployment status verification

### Model Performance Monitoring
- MLflow experiment tracking
- Metrics comparison across runs
- Model performance degradation detection
- Data drift monitoring (coming soon)

## Advanced Features

### MLflow Project Structure
- Reproducible experiments dengan `MLproject` file
- Environment isolation dengan Conda
- Parameter configuration flexibility
- Entry points untuk different training modes

### Docker Integration
- MLflow model serving dengan MLServer
- Automatic image building dari model registry
- Multi-stage deployment (dev, staging, prod)
- Container health monitoring

### GitHub Actions Advanced
- Conditional deployment berdasarkan performance
- Automatic rollback pada failure
- Multi-environment deployment
- Integration testing automation

## Troubleshooting

### Common Issues

**Environment Setup Errors**
```bash
# Clear conda cache
conda clean --all

# Recreate environment
conda env remove -n mlflow-env
conda env create -f conda.yaml
```

**MLflow Connection Issues**
```bash
# Check authentication
export MLFLOW_TRACKING_URI="your-tracking-uri"
mlflow experiments list
```

**Docker Build Failures**
```bash
# Check Docker daemon
docker --version
docker info

# Manual image build
cd MLProject
docker build -t worker-productivity-mlp .
```

### Debug Mode

Enable debug output dalam workflow:
```yaml
- name: Debug Information
  run: |
    echo "Repository structure:"
    find . -type f -name "*.py" | head -20
    ls -la MLProject/
```

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add improvement'`)
4. Push ke branch (`git push origin feature/improvement`)
5. Create Pull Request

### Development Guidelines
- Follow PEP 8 untuk Python code
- Add tests untuk new features
- Update documentation untuk changes
- Ensure CI/CD pipeline passes

## License

Project ini dilisensikan di bawah MIT License - lihat file [LICENSE](LICENSE) untuk detail.

## Links & Resources

- **GitHub Repository**: https://github.com/silmiaathqia/Workflow-CI
- **DagsHub MLflow**: https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow
- **Docker Hub**: https://hub.docker.com/r/your-username/worker-productivity-mlp
- **MLflow Documentation**: https://mlflow.org/docs/latest/
- **GitHub Actions**: https://docs.github.com/en/actions

## Support

Untuk pertanyaan atau bantuan:
- Create GitHub Issue
- Check MLflow logs di DagsHub
- Review CI/CD workflow logs
- Consult troubleshooting section

---

*Automated ML Pipeline with MLflow, Docker, and GitHub Actions*
