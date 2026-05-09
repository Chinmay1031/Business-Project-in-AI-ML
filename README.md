# IPL Live Score Predictor — ML Powered

A machine learning web application that predicts the final score of an IPL innings in real time. Enter live match stats and the app returns a predicted score range along with current/required run rates.

---

## How It Works

The model is a **Random Forest Regressor** trained on ball-by-ball IPL data. It takes a mid-innings snapshot as input and predicts what the batting team will finish on.

**Input features:**
- Batting team (one-hot encoded)
- Bowling team (one-hot encoded)
- Current runs & wickets
- Current over (must be ≥ 5.0)
- Runs and wickets in the last 5 overs

**Output:**
- Predicted final score range (± 5 runs)
- Most likely score
- Current run rate, projected required run rate, balls remaining

---

## Model Performance

Trained on 40,108 ball-level records from 8 IPL franchises (overs 5.0–19.5).

| Model | Train Accuracy | Test Accuracy |
|---|---|---|
| Linear Regression | 65.89% | 66.00% |
| **Random Forest** | **99.06%** | **93.58%** |

The trained model is hosted on Hugging Face Hub at [`chinmay1031/ipl-model`](https://huggingface.co/chinmay1031/ipl-model) and downloaded automatically at app startup.

---

## Supported Teams

| Team | |
|---|---|
| Chennai Super Kings | Mumbai Indians |
| Delhi Capitals | Rajasthan Royals |
| Punjab Kings | Royal Challengers Bangalore |
| Kolkata Knight Riders | Sunrisers Hyderabad |
| Gujarat Titans | Lucknow Super Giants |

> Gujarat Titans and Lucknow Super Giants are newer franchises not in the training data; they are mapped to the closest historical team for prediction.

---

## Project Structure

```
├── app.py                  # Dash web application
├── dataset/
│   └── ipl_data.csv        # Ball-by-ball IPL match data (76,014 rows)
├── notebooks/
│   └── ML_Model.ipynb      # Model training, EDA, and evaluation
└── requirements.txt        # Python dependencies
```

---

## Getting Started

### Prerequisites

- Python 3.10+

### Installation

```bash
git clone https://github.com/your-username/Business-Project-in-AI-ML.git
cd Business-Project-in-AI-ML
pip install -r requirements.txt
```

### Run locally

```bash
python app.py
```

Open `http://127.0.0.1:8050` in your browser.

The model is fetched from Hugging Face on first launch — no manual download needed.

---

## Dependencies

| Package | Purpose |
|---|---|
| `dash` + `dash-bootstrap-components` | Web UI framework |
| `scikit-learn` | Random Forest model |
| `huggingface_hub` + `joblib` | Model hosting and loading |
| `numpy` / `pandas` | Data processing |
| `gunicorn` | Production WSGI server |

---

## Deployment

The app is deployed on **Azure App Service** using the included GitHub Actions workflow. Gunicorn serves the Dash app in production via the `server` export in `app.py`.

---

## Dataset

`ipl_data.csv` contains ball-by-ball records from IPL seasons 2008 onward. It includes 76,014 rows across 15 columns covering match ID, venue, teams, batsman, bowler, and running totals. Only overs 5.0 and above are used for training, since early-over data is too noisy for accurate end-score prediction.

---

## License

This repository is open for academic and educational use.
