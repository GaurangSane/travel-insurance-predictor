# ✈️ Travel Insurance Prediction System

> **End-to-End Machine Learning project** — from raw data to live cloud deployment.  
> Built by an aspiring AI/ML Engineer to demonstrate real-world ML engineering skills.

🔗 **[Live Demo → travel-insurance-predictor.vercel.app](https://travel-insurance-predictor.vercel.app/)**

---

## 📌 What is this project?

Travel insurance companies need to identify customers who are likely to purchase insurance products. This project solves that problem using a complete ML pipeline — from data preprocessing to a **publicly deployed prediction API with an interactive UI**.

**Given customer attributes** (age, income, travel history, etc.), the model predicts whether the customer will buy travel insurance.

---

## 🚀 Live Demo

| Layer | URL |
|---|---|
| 🌐 Web App | [travel-insurance-predictor.vercel.app](https://travel-insurance-predictor.vercel.app/) |
| ⚡ API Endpoint | `POST /predict` |

---

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| **ML & Data** | Python, Scikit-learn, Pandas, NumPy |
| **Backend** | FastAPI, Uvicorn, Pydantic |
| **Frontend** | HTML, CSS, JavaScript |
| **Deployment** | Vercel, GitHub |
| **Model Export** | Joblib |

---

## 🧠 ML Workflow

```
Raw CSV Data
    │
    ▼
Data Preprocessing
(ColumnTransformer → OneHotEncoder + StandardScaler)
    │
    ▼
Model Training & Comparison
(5-Fold Cross Validation)
    │
    ▼
Best Model Selected → Exported with joblib
    │
    ▼
FastAPI Backend → Deployed on Vercel
    │
    ▼
Live Predictions via Interactive Frontend
```

### Models Compared

| Model | Notes |
|---|---|
| Logistic Regression | Baseline |
| Decision Tree | Interpretable |
| Random Forest | Ensemble — strong performer |
| K-Nearest Neighbors | Distance-based |
| Support Vector Machine | Kernel-based |

**Selection Criteria:** Cross-validation accuracy + generalization + prediction consistency

---

## 📊 Features Used for Prediction

| Feature | Description |
|---|---|
| `Age` | Customer age |
| `Employment Type` | Government / Private sector |
| `Graduated` | Graduation status (Yes/No) |
| `Chronic Disease` | Existing disease condition (Yes/No) |
| `Frequent Flyer` | Frequent travel behavior (Yes/No) |
| `Ever Travelled Abroad` | International travel history (Yes/No) |
| `Annual Income` | Customer income (₹) |
| `Family Members` | Family size |

---

## ⚙️ API Usage

### Endpoint

```
POST /predict
```

### Sample Request

```json
{
  "Age": 30,
  "Employment_type": "Government Sector",
  "graduated": "Yes",
  "cronic_disease": "No",
  "frequent_flyer": "Yes",
  "ever_travelled_abroad": "Yes",
  "annual_income": 1000000,
  "family_members": 2
}
```

### Sample Response

```json
{
  "prediction": "Will Buy Insurance",
  "confidence": 0.87
}
```

---

## 📂 Project Structure

```
TRAVEL_INSURANCE/
│
├── app.py                          # FastAPI backend
├── index.html                      # Frontend UI
├── requirements.txt
├── vercel.json                     # Vercel deployment config
├── travel_insurance.ipynb          # ML training notebook
├── TravelInsurancePrediction.csv   # Dataset
├── travel_insurance_model.pkl      # Trained model
└── README.md
```

---

## 🔥 Real-World Challenges Solved

This wasn't just a notebook project — deploying to production surfaced real engineering problems:

- ✅ Pickle/Joblib model compatibility across environments
- ✅ Dependency version mismatches between local and Vercel runtime
- ✅ Frontend ↔ Backend CORS and API communication
- ✅ Vercel cold-start and model loading optimization
- ✅ Production error tracing and runtime debugging

---

## 📈 What I Learned

**Machine Learning**
- Classification algorithms, cross-validation, feature preprocessing
- Scikit-learn Pipelines and ColumnTransformer
- Model serialization and serving

**Backend Engineering**
- FastAPI REST API development with Pydantic validation
- Serving ML models in production

**Deployment Engineering**
- Cloud deployment on Vercel
- Environment consistency and dependency management
- Production debugging

**Frontend**
- Fetch API integration with a live ML backend
- Dynamic UI updates and error handling

---

## 🔮 Planned Improvements

- [ ] Docker containerization
- [ ] CI/CD pipelines (GitHub Actions)
- [ ] Streamlit version for quick demos
- [ ] Advanced analytics dashboard
- [ ] Model monitoring and drift detection
- [ ] Database integration and user authentication

---

## 👨‍💻 About Me

**Gaurang Sane** — Aspiring AI/ML Engineer & Data Scientist

I'm passionate about building **end-to-end AI systems** that go beyond Jupyter notebooks — covering model training, API development, frontend integration, and cloud deployment.

This project reflects my ability to take an ML idea from raw data all the way to a live, usable product.

📫 **[LinkedIn](https://www.linkedin.com/in/gaurang-sane-84b5b1254)** · **[GitHub](https://github.com/GaurangSane)** · **[Portfolio](https://portfolio-site-alpha-dun.vercel.app/)**

> ⭐ If you found this project useful or interesting, consider giving it a star!
