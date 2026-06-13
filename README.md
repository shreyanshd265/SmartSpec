# SmartSpec — Laptop Price Predictor

SmartSpec is a machine learning web app that predicts the price of a laptop based on its specifications like RAM, SSD, processor generation, graphics card, display, brand, etc.

🔗 Live: https://smartspec-3x3m.onrender.com

## About

I built this project to understand the complete ML workflow — from cleaning raw data to deploying a working model. The dataset has 920+ real laptop listings scraped from an e-commerce site.

## How it works

1. User enters laptop specs in the form
2. Data is cleaned and preprocessed (same way as training data)
3. Trained model predicts the price
4. Price is shown on the page

## Tech Stack

- Python, Pandas, NumPy
- Scikit-learn, XGBoost, CatBoost
- Flask
- HTML/CSS
- Deployed on Render

## Model

Tried multiple regression models (Linear Regression, Random Forest, XGBoost, CatBoost, etc.) with GridSearchCV for hyperparameter tuning. Best model gives around **87% R² score**.

Price was log-transformed during training since the original price distribution was skewed.

## Run locally

```bash
git clone https://github.com/yourusername/SmartSpec.git
cd SmartSpec
pip install -r requirements.txt
python main.py
```

Then open `http://localhost:5000`

## Project Structure

```
app/
├── main.py
├── model.pkl
├── preprocessor.pkl
├── requirements.txt
├── src/
│   ├── components/
│   ├── pipeline/
│   └── utils.py
└── templates/
```


