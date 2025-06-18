# Worker Productivity MLflow CI/CD Pipeline

[![MLflow CI/CD](https://github.com/silmiaathqia/Workflow-CI/actions/workflows/ci-mlflow.yml/badge.svg)](https://github.com/silmiaathqia/Workflow-CI/actions/workflows/ci-mlflow.yml) [![Python](https://img.shields.io/badge/python-3.12.7-blue.svg)](https://www.python.org/downloads/) [![MLflow](https://img.shields.io/badge/MLflow-2.19.0-orange.svg)](https://mlflow.org/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

## Overview

Pipeline CI/CD otomatis untuk klasifikasi produktivitas pekerja menggunakan **MLflow Project**. Pipeline ini secara otomatis melatih, memvalidasi, dan menyebarkan model Neural Network untuk memprediksi tingkat produktivitas pekerja (High, Medium, Low).

### Fitur Utama

- Automated ML Pipeline dengan MLflow Project
- Integrasi DagsHub untuk experiment tracking
- GitHub Actions CI/CD
- Automated model versioning  
- Automated releases dengan artifacts

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
│       ├── data_test.csv          # Test dataset
│       ├── label_encoder.pkl      # Label encoder
│       ├── feature_scaler.pkl     # Feature scaler
│       ├── label_mapping.csv      # Label mappings
│       ├── feature_names.csv      # Feature names
│       └── preprocessing_summary.json # Preprocessing info
├── README.md
├── requirements.txt               # Python dependencies
└── conda.yaml                    # Main conda environment
```

## Model Architecture

**Basic MLP Classifier (Scikit-learn)**
- Hidden layers: (128, 64, 32)
- Activation: ReLU
- Solver: Adam
- L2 regularization: alpha=0.001
- Early stopping: enabled
- Target classes: High, Low, Medium

## Data Requirements

### Format Data

Data harus dalam format CSV dengan kolom target `productivity_label_encoded` dan fitur-fitur numerik yang telah diproses.

#### File yang Diperlukan:
- `data_train.csv` - Data training
- `data_validation.csv` - Data validasi
- `data_test.csv` - Data testing

#### Format Target:
- 0: High Productivity
- 1: Low Productivity  
- 2: Medium Productivity

## Setup dan Installation

### Prerequisites

- Python 3.12.7
- Conda atau Miniconda
- Git
- Akun GitHub dengan Actions enabled
- Akun DagsHub untuk MLflow tracking

### Environment Setup

1. Clone repository:
```bash
git clone https://github.com/silmiaathqia/Workflow-CI.git
cd Workflow-CI
```

2. Buat conda environment:
```bash
conda env create -f conda.yaml
conda activate mlflow-env
```

3. Install dependencies tambahan:
```bash
pip install -r requirements.txt
```

### GitHub Secrets Configuration

Untuk menjalankan CI/CD pipeline, tambahkan secrets berikut di GitHub repository:

```
MLFLOW_TRACKING_URI      # https://dagshub.com/username/repo.mlflow
DAGSHUB_USERNAME         # Username DagsHub anda
DAGSHUB_USER_TOKEN       # Token akses DagsHub
```

#### Cara mendapatkan DagsHub Token:
1. Login ke DagsHub.com
2. Pergi ke Settings > Access Tokens
3. Generate new token dengan scope MLflow
4. Copy token ke GitHub Secrets

## Usage

### Local Training

Jalankan model training secara lokal:

```bash
cd MLProject
python modelling.py --data_path processed_data --experiment_name "Worker_Productivity_Local"
```

### MLflow Project

Jalankan sebagai MLflow Project:

```bash
mlflow run . -P data_path="processed_data" -P experiment_name="Worker_Productivity_Classification"
```

### Automated CI/CD

Pipeline akan otomatis berjalan saat:
- Push ke branch `main` atau `master`
- Pull request ke branch `main` atau `master`
- Manual trigger melalui GitHub Actions

## Pipeline Workflow

### Tahapan CI/CD:

1. **Repository Checkout** - Download source code
2. **Environment Setup** - Setup Conda dan dependencies
3. **Data Validation** - Validasi keberadaan dan format data
4. **Model Training** - Training model dengan MLflow tracking
5. **Artifact Generation** - Generate model artifacts dan reports
6. **Model Registration** - Register model di DagsHub
7. **Release Creation** - Create GitHub release dengan artifacts

### Artifacts yang Dihasilkan:

- `scaler_basic.pkl` - Feature scaler
- `classification_report_basic.txt` - Laporan klasifikasi
- `model_summary_basic.txt` - Summary model
- `deployment_info_basic.json` - Info deployment
- `confusion_matrix_basic.png` - Confusion matrix plot
- `metrics_comparison_basic.png` - Perbandingan metrics

## Monitoring dan Tracking

### MLflow Tracking

Semua experiment dapat dimonitor melalui:
- **DagsHub MLflow UI**: `https://dagshub.com/username/Worker-Productivity-MLflow`
- **Local MLflow**: `mlflow ui` (jika running local)

### Metrics yang Ditrack:

- Accuracy
- Precision (weighted)
- Recall (weighted)  
- F1-Score (weighted)
- Per-class precision, recall, F1
- Training loss
- Number of iterations
- Convergence status

## Model Performance

Model performance dapat dilihat melalui:

1. **MLflow Experiments** - Detailed metrics dan parameters
2. **GitHub Releases** - Automated reports dalam setiap release
3. **Classification Report** - Per-class performance metrics
4. **Confusion Matrix** - Visual representation performance

## Troubleshooting

### Common Issues:

**Data Loading Error**
```
Error loading data: FileNotFoundError
```
**Solution**: Pastikan file data tersedia di folder `processed_data/`

**MLflow Connection Error**
```
MLflow setup error: Authentication failed
```
**Solution**: Periksa GitHub Secrets dan DagsHub token

**Model Training Failure**
```
Training failed: Memory error
```
**Solution**: Reduce model complexity atau gunakan batch training

**Pipeline Failure**
```
Workflow failed at step X
```
**Solution**: Check GitHub Actions logs untuk detail error

### Debug Mode

Untuk debugging lokal, gunakan verbose mode:

```bash
python modelling.py --data_path processed_data --experiment_name "Debug_Session" --cleanup
```

## Contributing

1. Fork repository
2. Create feature branch: `git checkout -b feature/nama-fitur`
3. Commit changes: `git commit -am 'Add fitur baru'`
4. Push branch: `git push origin feature/nama-fitur`
5. Submit Pull Request

### Development Guidelines:

- Follow PEP 8 untuk Python code style
- Tambahkan tests untuk fitur baru
- Update documentation jika diperlukan
- Pastikan CI/CD pipeline berjalan sukses

## License

Project ini menggunakan MIT License. Lihat file [LICENSE](LICENSE) untuk detail.

## Links

- **Repository**: https://github.com/silmiaathqia/Workflow-CI
- **DagsHub Project**: https://dagshub.com/silmiaathqia/Worker-Productivity-MLflow
- **MLflow Documentation**: https://mlflow.org/docs/
- **GitHub Actions**: https://docs.github.com/en/actions

## Support

Jika mengalami masalah atau memiliki pertanyaan:

1. Check **Issues** section di GitHub repository
2. Review **MLflow logs** di DagsHub
3. Check **GitHub Actions logs** untuk CI/CD issues
4. Create new issue dengan detail error dan environment info
