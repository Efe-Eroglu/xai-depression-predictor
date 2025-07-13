<div align="center">
  
# 🧠 Depression Risk Analysis Platform

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Active-brightgreen.svg)]()

  <img src="https://img.shields.io/badge/AI-Powered-orange?style=for-the-badge&logo=openai" alt="AI Powered">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-yellow?style=for-the-badge&logo=scikit-learn" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Explainable%20AI-SHAP-purple?style=for-the-badge" alt="Explainable AI">
</div>

---

## 📋 Table of Contents / İçindekiler

- [English](#english)
  - [Overview](#overview)
  - [Features](#features)
  - [Technology Stack](#technology-stack)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model Performance](#model-performance)
  - [Screenshots](#screenshots)
  - [Architecture](#architecture)
  - [Contributing](#contributing)
  - [License](#license)

- [Türkçe](#türkçe)
  - [Genel Bakış](#genel-bakış)
  - [Özellikler](#özellikler)
  - [Teknoloji Altyapısı](#teknoloji-altyapısı)
  - [Kurulum](#kurulum)
  - [Kullanım](#kullanım)
  - [Model Performansı](#model-performansı)
  - [Ekran Görüntüleri](#ekran-görüntüleri)
  - [Mimari](#mimari)
  - [Katkıda Bulunma](#katkıda-bulunma)
  - [Lisans](#lisans)

---

# English

## Overview

The Depression Risk Analysis Platform is an advanced AI-powered web application that leverages machine learning algorithms to assess depression risk based on various lifestyle, academic, and health factors. The platform provides personalized risk assessments with explainable AI using SHAP (SHapley Additive exPlanations) methodology.

### 🎯 Key Objectives
- **Early Detection**: Identify potential depression risk factors early
- **Personalized Assessment**: Provide individualized risk analysis
- **Explainable AI**: Transparent decision-making process
- **User-Friendly Interface**: Modern, responsive web design
- **Professional Recommendations**: Actionable health advice

## Features

### 🧠 Core Features
- **Multi-Model Support**: Choose from 4 different ML algorithms
  - Logistic Regression
  - Random Forest
  - XGBoost
  - Support Vector Machine (SVM)
- **Real-time Analysis**: Instant risk assessment
- **SHAP Explanations**: Understand prediction factors
- **Risk Level Classification**: High/Medium/Low risk categorization

### 🎨 User Interface
- **Modern Design**: Glassmorphism and gradient effects
- **Responsive Layout**: Works on all devices
- **Interactive Elements**: Smooth animations and transitions
- **Professional Styling**: Clean, medical-grade appearance

### 📊 Analytics & Insights
- **Model Performance Metrics**: Accuracy, confidence scores
- **Feature Importance**: SHAP waterfall charts
- **Personalized Recommendations**: Health improvement suggestions
- **Statistical Overview**: Comprehensive risk analysis

## Technology Stack

### 🛠️ Backend Technologies
```python
# Core Framework
Streamlit == 1.28+          # Web application framework
Python == 3.8+              # Programming language

# Machine Learning
scikit-learn == 1.3+        # ML algorithms and preprocessing
XGBoost == 1.7+             # Gradient boosting
SHAP == 0.43+               # Explainable AI
joblib == 1.3+              # Model serialization

# Data Processing
pandas == 2.0+              # Data manipulation
numpy == 1.24+              # Numerical computing

# Visualization
matplotlib == 3.7+          # Plotting library
plotly == 5.15+             # Interactive charts
```

### 🎨 Frontend Technologies
- **CSS3**: Modern styling with gradients and animations
- **HTML5**: Semantic markup
- **JavaScript**: Interactive elements (via Streamlit)
- **Responsive Design**: Mobile-first approach

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Step-by-Step Setup

1. **Clone the Repository**
```bash
git clone https://github.com/Efe-Eroglu/depression-risk-analysis.git
cd depression-risk-analysis
```

2. **Create Virtual Environment**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Install Dependencies**
```bash
pip install -r requirements.txt
```

4. **Verify Installation**
```bash
python -c "import streamlit, pandas, sklearn, xgboost, shap; print('All packages installed successfully!')"
```

### 🚀 Quick Start
```bash
# Run the application
streamlit run app.py

# Access the web interface
# Open http://localhost:8501 in your browser
```

## Usage

### 📱 User Guide

1. **Model Selection**
   - Choose your preferred ML algorithm from the sidebar
   - Each model has different accuracy characteristics

2. **Data Input**
   - Fill in personal information in organized categories:
     - 👤 Demographic Information
     - 🎓 Academic Status
     - 💼 Work/School Life
     - 🏥 Health Status

3. **Analysis Process**
   - Click "Start Analysis" to process your data
   - View real-time risk assessment results

4. **Results Interpretation**
   - Review risk percentage and classification
   - Examine SHAP analysis for factor importance
   - Read personalized recommendations

### 🔧 Advanced Usage

#### Model Comparison
```python
# Compare different models
models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "SVM": "models/svm.pkl"
}
```

#### Custom Data Input
```python
# Example input format
input_data = {
    "Gender": 1,  # 1 for Male, 0 for Female
    "Age": 25,
    "Academic Pressure": 5,
    "Work Pressure": 3,
    "CGPA": 7.5,
    "Study Satisfaction": 6,
    "Job Satisfaction": 7,
    "Sleep Duration": 2,  # 0-3 scale
    "Dietary Habits": 2,  # 0-3 scale
    "Work/Study Hours": 8,
    "Financial Stress": 4,
    "Family History of Mental Illness": 0,
    "Suicidal Thoughts": 0
}
```

### 🎯 Model Characteristics

- **XGBoost**: Best overall performance, handles non-linear relationships
- **Random Forest**: Good interpretability, robust to outliers
- **Logistic Regression**: Fast inference, good baseline
- **SVM**: Effective for high-dimensional data

## Screenshots

### 🖥️ Main Interface
![Main Interface](https://via.placeholder.com/800x400/667eea/ffffff?text=Main+Interface)

### 📊 Risk Assessment
![Risk Assessment](https://via.placeholder.com/800x400/764ba2/ffffff?text=Risk+Assessment)

### 🧠 SHAP Analysis
![SHAP Analysis](https://via.placeholder.com/800x400/11998e/ffffff?text=SHAP+Analysis)

### 📱 Mobile Responsive
![Mobile View](https://via.placeholder.com/400x600/ff6b6b/ffffff?text=Mobile+View)


### 📁 Project Structure
```
depression-risk-analysis/
├── app.py                 # Main application file
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── models/               # Trained ML models
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── svm.pkl
├── utils/                # Utility functions
│   ├── __init__.py
│   ├── preprocessing.py  # Data preprocessing
│   └── shap_helper.py    # SHAP analysis utilities
├── assets/               # Static assets
└── depression-risk-xai.ipynb  # Jupyter notebook for model training
```

## Contributing

### 🤝 How to Contribute

1. **Fork the Repository**
```bash
git clone https://github.com/Efe-Eroglu/depression-risk-analysis.git
```

2. **Create Feature Branch**
```bash
git checkout -b feature/amazing-feature
```

3. **Make Changes**
   - Follow PEP 8 style guidelines
   - Add tests for new features
   - Update documentation

4. **Commit Changes**
```bash
git commit -m "Add amazing feature"
git push origin feature/amazing-feature
```

5. **Create Pull Request**
   - Describe your changes
   - Link related issues
   - Request code review

### 📋 Development Guidelines

- **Code Style**: Follow PEP 8 conventions
- **Documentation**: Update README for new features
- **Testing**: Add unit tests for new functionality
- **Performance**: Optimize for speed and accuracy

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

# Türkçe

## Genel Bakış

Depresyon Risk Analizi Platformu, çeşitli yaşam tarzı, akademik ve sağlık faktörlerine dayalı olarak depresyon riskini değerlendirmek için makine öğrenmesi algoritmalarını kullanan gelişmiş bir yapay zeka destekli web uygulamasıdır. Platform, SHAP (SHapley Additive exPlanations) metodolojisi kullanarak açıklanabilir yapay zeka ile kişiselleştirilmiş risk değerlendirmeleri sunar.

### 🎯 Temel Hedefler
- **Erken Tespit**: Potansiyel depresyon risk faktörlerini erken belirleme
- **Kişiselleştirilmiş Değerlendirme**: Bireyselleştirilmiş risk analizi
- **Açıklanabilir Yapay Zeka**: Şeffaf karar verme süreci
- **Kullanıcı Dostu Arayüz**: Modern, responsive web tasarımı
- **Profesyonel Öneriler**: Uygulanabilir sağlık tavsiyeleri

## Özellikler

### 🧠 Temel Özellikler
- **Çoklu Model Desteği**: 4 farklı ML algoritmasından seçim
  - Lojistik Regresyon
  - Rastgele Orman
  - XGBoost
  - Destek Vektör Makinesi (SVM)
- **Gerçek Zamanlı Analiz**: Anında risk değerlendirmesi
- **SHAP Açıklamaları**: Tahmin faktörlerini anlama
- **Risk Seviyesi Sınıflandırması**: Yüksek/Orta/Düşük risk kategorilendirmesi

### 🎨 Kullanıcı Arayüzü
- **Modern Tasarım**: Glassmorphism ve gradient efektleri
- **Responsive Düzen**: Tüm cihazlarda çalışır
- **İnteraktif Öğeler**: Yumuşak animasyonlar ve geçişler
- **Profesyonel Stil**: Temiz, tıbbi kalite görünüm

### 📊 Analitik ve İçgörüler
- **Model Performans Metrikleri**: Doğruluk, güven skorları
- **Özellik Önemliliği**: SHAP waterfall grafikleri
- **Kişiselleştirilmiş Öneriler**: Sağlık iyileştirme önerileri
- **İstatistiksel Genel Bakış**: Kapsamlı risk analizi

## Teknoloji Altyapısı

### 🛠️ Backend Teknolojileri
```python
# Temel Framework
Streamlit == 1.28+          # Web uygulama framework'ü
Python == 3.8+              # Programlama dili

# Makine Öğrenmesi
scikit-learn == 1.3+        # ML algoritmaları ve ön işleme
XGBoost == 1.7+             # Gradient boosting
SHAP == 0.43+               # Açıklanabilir AI
joblib == 1.3+              # Model serileştirme

# Veri İşleme
pandas == 2.0+              # Veri manipülasyonu
numpy == 1.24+              # Sayısal hesaplama

# Görselleştirme
matplotlib == 3.7+          # Grafik kütüphanesi
plotly == 5.15+             # İnteraktif grafikler
```

### 🎨 Frontend Teknolojileri
- **CSS3**: Gradient ve animasyonlarla modern stil
- **HTML5**: Semantik işaretleme
- **JavaScript**: İnteraktif öğeler (Streamlit üzerinden)
- **Responsive Tasarım**: Mobile-first yaklaşım

## Kurulum

### Ön Gereksinimler
- Python 3.8 veya üzeri
- pip paket yöneticisi
- Git

### Adım Adım Kurulum

1. **Repository'yi Klonlayın**
```bash
git clone https://github.com/Efe-Eroglu/depression-risk-analysis.git
cd depression-risk-analysis
```

2. **Sanal Ortam Oluşturun**
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

3. **Bağımlılıkları Yükleyin**
```bash
pip install -r requirements.txt
```

4. **Kurulumu Doğrulayın**
```bash
python -c "import streamlit, pandas, sklearn, xgboost, shap; print('Tüm paketler başarıyla yüklendi!')"
```

### 🚀 Hızlı Başlangıç
```bash
# Uygulamayı çalıştırın
streamlit run app.py

# Web arayüzüne erişin
# Tarayıcınızda http://localhost:8501 adresini açın
```

## Kullanım

### 📱 Kullanıcı Kılavuzu

1. **Model Seçimi**
   - Kenar çubuğundan tercih ettiğiniz ML algoritmasını seçin
   - Her modelin farklı doğruluk özellikleri vardır

2. **Veri Girişi**
   - Kişisel bilgileri organize kategorilerde doldurun:
     - 👤 Demografik Bilgiler
     - 🎓 Akademik Durum
     - 💼 İş/Okul Yaşamı
     - 🏥 Sağlık Durumu

3. **Analiz Süreci**
   - Verilerinizi işlemek için "Analizi Başlat"a tıklayın
   - Gerçek zamanlı risk değerlendirme sonuçlarını görüntüleyin

4. **Sonuç Yorumlama**
   - Risk yüzdesini ve sınıflandırmayı inceleyin
   - Faktör önemliliği için SHAP analizini inceleyin
   - Kişiselleştirilmiş önerileri okuyun

### 🔧 Gelişmiş Kullanım

#### Model Karşılaştırması
```python
# Farklı modelleri karşılaştırın
models = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl",
    "SVM": "models/svm.pkl"
}
```

#### Özel Veri Girişi
```python
# Örnek giriş formatı
input_data = {
    "Gender": 1,  # Erkek için 1, Kadın için 0
    "Age": 25,
    "Academic Pressure": 5,
    "Work Pressure": 3,
    "CGPA": 7.5,
    "Study Satisfaction": 6,
    "Job Satisfaction": 7,
    "Sleep Duration": 2,  # 0-3 ölçeği
    "Dietary Habits": 2,  # 0-3 ölçeği
    "Work/Study Hours": 8,
    "Financial Stress": 4,
    "Family History of Mental Illness": 0,
    "Suicidal Thoughts": 0
}
```

### 🎯 Model Özellikleri

- **XGBoost**: En iyi genel performans, doğrusal olmayan ilişkileri işler
- **Random Forest**: İyi yorumlanabilirlik, aykırı değerlere karşı dayanıklı
- **Logistic Regression**: Hızlı çıkarım, iyi temel
- **SVM**: Yüksek boyutlu veriler için etkili

## Ekran Görüntüleri

### 🖥️ Ana Arayüz
![Ana Arayüz](https://via.placeholder.com/800x400/667eea/ffffff?text=Ana+Arayüz)

### 📊 Risk Değerlendirmesi
![Risk Değerlendirmesi](https://via.placeholder.com/800x400/764ba2/ffffff?text=Risk+Değerlendirmesi)

### 🧠 SHAP Analizi
![SHAP Analizi](https://via.placeholder.com/800x400/11998e/ffffff?text=SHAP+Analizi)

### 📱 Mobil Responsive
![Mobil Görünüm](https://via.placeholder.com/400x600/ff6b6b/ffffff?text=Mobil+Görünüm)

### 📁 Proje Yapısı
```
depression-risk-analysis/
├── app.py                 # Ana uygulama dosyası
├── requirements.txt       # Python bağımlılıkları
├── README.md             # Proje dokümantasyonu
├── models/               # Eğitilmiş ML modelleri
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   ├── xgboost.pkl
│   └── svm.pkl
├── utils/                # Yardımcı fonksiyonlar
│   ├── __init__.py
│   ├── preprocessing.py  # Veri ön işleme
│   └── shap_helper.py    # SHAP analiz yardımcıları
├── assets/               # Statik dosyalar
└── depression-risk-xai.ipynb  # Model eğitimi için Jupyter notebook
```

## Katkıda Bulunma

### 🤝 Nasıl Katkıda Bulunulur

1. **Repository'yi Fork Edin**
```bash
git clone https://github.com/Efe-Eroglu/depression-risk-analysis.git
```

2. **Özellik Dalı Oluşturun**
```bash
git checkout -b feature/harika-ozellik
```

3. **Değişiklikleri Yapın**
   - PEP 8 stil rehberlerini takip edin
   - Yeni özellikler için test ekleyin
   - Dokümantasyonu güncelleyin

4. **Değişiklikleri Commit Edin**
```bash
git commit -m "Harika özellik eklendi"
git push origin feature/harika-ozellik
```

5. **Pull Request Oluşturun**
   - Değişikliklerinizi açıklayın
   - İlgili sorunları bağlayın
   - Kod incelemesi isteyin

### 📋 Geliştirme Rehberleri

- **Kod Stili**: PEP 8 kurallarını takip edin
- **Dokümantasyon**: Yeni özellikler için README'yi güncelleyin
- **Test**: Yeni işlevsellik için birim testleri ekleyin
- **Performans**: Hız ve doğruluk için optimize edin

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

---

## 📞 İletişim / Contact

[![Email](https://img.shields.io/badge/Email-efeeroglu.dev@gmail.com-red?style=flat&logo=gmail&logoColor=white)](mailto:efeeroglu.dev@gmail.com)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-efeeroglu-blue?style=flat&logo=linkedin)](https://www.linkedin.com/in/efeeroglu/)
[![GitHub](https://img.shields.io/badge/GitHub-Efe--Eroglu-181717?style=flat&logo=github)](https://github.com/Efe-Eroglu)

## 🙏 Teşekkürler / Acknowledgments

- Streamlit ekibine harika framework için
- SHAP geliştiricilerine açıklanabilir AI için
- Tüm katkıda bulunanlara

---

<div align="center">
  <p>Made with ❤️ for mental health awareness</p>
</div> 
