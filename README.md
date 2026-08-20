# Titanic Survival Predictor

A machine learning web app that predicts Titanic passenger survival using **Gaussian Naive Bayes**.

## Model Performance
- **Accuracy:** 77.6%
- **Cross-validation mean:** 77.4% (5-fold)

## Features Used
| Feature | Description |
|---|---|
| Pclass | Passenger class (1, 2, 3) |
| Age | Passenger age (mean-imputed) |
| Fare | Ticket fare |
| female | Sex encoded (1 = female, 0 = male) |

## Run Locally
```bash
pip install -r requirements.txt
python app.py
```
Then open http://localhost:5000

## Deploy to Render
1. Push this folder to a GitHub repo
2. Go to https://render.com → New Web Service
3. Connect your repo
4. Set **Start Command** to: `gunicorn app:app`
5. Deploy — you'll get a public URL

## API Usage
```bash
curl -X POST https://your-app.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"Pclass": 3, "Age": 22, "Fare": 7.25, "female": 0}'
```

**Response:**
```json
{ "survived": 0, "survival_probability": 13.2, "death_probability": 86.8 }
```

## Project Structure
```
titanic-app/
├── app.py          # Flask API + frontend
├── model.pkl       # Trained GaussianNB model
├── titanic.csv     # Dataset
├── requirements.txt
└── README.md
```
