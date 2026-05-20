# ============================================================
#  PREMIUM PRODUCT LAUNCH ANALYTICS – FINAL FULL VERSION (FIXED)
# ============================================================

import os
from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neighbors import NearestNeighbors

import plotly.express as px
import plotly.io as pio

# ============================================================
#  FLASK SETUP
# ============================================================

app = Flask(__name__)
app.secret_key = "super-secret-key"

DATA_FILE = "product_launch_data.csv"
MODEL_FILE = "model.pkl"


# ============================================================
#  LOAD DATASET *EVERY TIME* (MAIN FIX)
# ============================================================

def load_df():
    df = pd.read_csv(DATA_FILE)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    df["launch_year"] = df["launch_year"].astype(int)
    df["rating"] = df["rating"].astype(float)
    df["popularity"] = df["popularity"].astype(float)
    df["hits"] = df["hits"].astype(float)
    df["success_rate"] = df["success_rate"].astype(float)
    return df


# Load once for model training
if not os.path.exists(DATA_FILE):
    raise FileNotFoundError("Dataset missing!")

df_initial = load_df()


# ============================================================
#  FEATURE PREPARATION
# ============================================================

def prepare_features(df_in):
    X = pd.DataFrame()
    X["rating"] = df_in["rating"]
    X["popularity"] = df_in["popularity"]
    X["hits_log"] = np.log1p(df_in["hits"])
    X["year"] = df_in["launch_year"]
    y = df_in["success_rate"]
    return X, y


# ============================================================
#  TRAIN MODEL (ONLY ONCE)
# ============================================================

if not os.path.exists(MODEL_FILE):
    X, y = prepare_features(df_initial)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=300, random_state=42)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    print("Model MSE:", mean_squared_error(y_test, preds))
    print("Model R2:", r2_score(y_test, preds))

    joblib.dump(model, MODEL_FILE)

model = joblib.load(MODEL_FILE)


# ============================================================
#  METRIC FUNCTIONS
# ============================================================

def compute_weighted_success(brand_df):
    brand_df = brand_df.sort_values("launch_year")
    years = brand_df["launch_year"].values
    rates = brand_df["success_rate"].values
    weights = years - years.min() + 1
    return round(float(np.sum(rates * weights) / np.sum(weights)), 2)


def compute_momentum(brand_df):
    brand_df = brand_df.sort_values("launch_year")
    years = brand_df["launch_year"].values
    rates = brand_df["success_rate"].values
    if len(years) <= 1:
        return 0.0
    slope, _ = np.polyfit(years, rates, 1)
    return round(float(slope), 3)


def compute_stability(brand_df):
    std = np.std(brand_df["success_rate"].values)
    return round(max(0, 100 - std * 5), 2)


# ============================================================
#  PLOTLY HELPER
# ============================================================

def fig_to_html(fig):
    fig.update_layout(
        height=430, 
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)", 
        plot_bgcolor="rgba(0,0,0,0)"
    )
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


# ============================================================
#  ROUTES
# ============================================================

@app.route("/")
def home():
    df = load_df()
    latest_year = int(df["launch_year"].max())
    total_brands = int(df["brand"].nunique())
    total_models = int(df["model"].nunique())
    return render_template(
        "index.html", 
        latest_year=latest_year, 
        total_brands=total_brands, 
        total_models=total_models
    )


# ===================== DASHBOARD ============================

@app.route("/dashboard")
def dashboard():

    df = load_df()   # <--- FIX: Always fresh data

    fig_timeline = px.line(
        df.sort_values("launch_year"),
        x="launch_year",
        y="success_rate",
        color="brand",
        markers=True,
        title="Fig.No.7.2.1 – Year-by-Year Success Timeline"
    )
    chart_timeline = fig_to_html(fig_timeline)

    fig_success = px.bar(
        df,
        x="model",
        y="success_rate",
        color="brand",
        title="Fig.No.7.2.2 – Success Rates Per Model"
    )
    chart_success = fig_to_html(fig_success)

    fig_scatter = px.scatter(
        df,
        x="rating",
        y="popularity",
        size="success_rate",
        color="brand",
        hover_data=["model", "hits"],
        title="Fig.No.7.2.3 – Rating vs Popularity"
    )
    chart_scatter = fig_to_html(fig_scatter)

    brand_avg = df.groupby("brand")["success_rate"].mean().reset_index()
    fig_race = px.bar(
        brand_avg.sort_values("success_rate"),
        x="success_rate",
        y="brand",
        color="brand",
        orientation="h",
        title="Fig.No.7.2.4 – Competitor Brand Ranking"
    )
    chart_race = fig_to_html(fig_race)

    return render_template(
        "dashboard.html",
        chart_timeline=chart_timeline,
        chart_success=chart_success,
        chart_scatter=chart_scatter,
        chart_race=chart_race
    )


# ===================== ANALYZE BRAND =========================

@app.route("/analyze", methods=["GET", "POST"])
def analyze():

    df = load_df()  # <--- FIX
    brands = sorted(df["brand"].unique())

    if request.method == "POST":
        brand = request.form.get("brand")

        if not brand:
            flash("Select a brand!", "danger")
            return render_template("analyze.html", brands=brands)

        sub = df[df["brand"] == brand]

        fig_pop = px.line(sub, x="launch_year", y="popularity", markers=True,
                          title=f"{brand} – Popularity Trend")
        chart_pop = fig_to_html(fig_pop)

        fig_suc = px.bar(sub, x="model", y="success_rate",
                         title=f"{brand} – Success by Model")
        chart_suc = fig_to_html(fig_suc)

        metrics = {
            "weighted_success": compute_weighted_success(sub),
            "momentum": compute_momentum(sub),
            "stability": compute_stability(sub),
        }

        return render_template(
            "analyze.html",
            brands=brands,
            selected_brand=brand,
            pop_chart=chart_pop,
            suc_chart=chart_suc,
            metrics=metrics,
            records=sub.to_dict(orient="records")
        )

    return render_template("analyze.html", brands=brands)


# ===================== UPLOAD DATASET =========================

@app.route("/upload-dataset", methods=["GET", "POST"])
def upload_dataset():
    if request.method == "POST":
        file = request.files.get("dataset")

        if not file:
            flash("No file selected!", "danger")
            return redirect(url_for("upload_dataset"))

        if not file.filename.lower().endswith(".csv"):
            flash("Only CSV files allowed!", "danger")
            return redirect(url_for("upload_dataset"))

        file.save(DATA_FILE)

        flash("Dataset uploaded successfully!", "success")
        return redirect(url_for("dashboard"))

    return render_template("upload.html")


# ===================== PRODUCT PREDICTION ====================

@app.route("/predict", methods=["GET", "POST"])
def predict():

    df = load_df()  # <--- FIX
    brands = sorted(df["brand"].unique())

    if request.method == "POST":
        brand = request.form["brand"]
        model_name = request.form["model"]
        rating = float(request.form["rating"])
        popularity = float(request.form["popularity"])
        hits = float(request.form["hits"])
        launch_year = int(request.form["launch_year"])

        X = pd.DataFrame([{
            "rating": rating,
            "popularity": popularity,
            "hits_log": np.log1p(hits),
            "year": launch_year
        }])

        score = float(model.predict(X)[0])
        score = round(score, 2)

        expected_views = int(popularity * 5000 + hits * 1.2)

        if score > 90:
            platform = "Instagram & YouTube (High viral potential)"
        elif score > 80:
            platform = "YouTube & Twitter/X"
        else:
            platform = "Instagram Reels (boost engagement)"

        hashtags = [
            f"#{brand.replace(' ', '')}",
            f"#{model_name.replace(' ', '')}",
            "#TechLaunch",
            "#TrendingNow",
            "#FutureTech",
            "#SmartphoneLaunch",
            "#Innovation"
        ]

        captions = [
            f"🚀 The future arrives with {brand} {model_name}! {' '.join(hashtags)}",
            f"🔥 {brand} drops the next big thing — {model_name}! {' '.join(hashtags)}",
            f"⚡ Experience raw power and innovation with {model_name}! {' '.join(hashtags)}",
        ]

        brand_df = df[df["brand"] == brand]
        fig_past = px.line(
            brand_df,
            x="launch_year",
            y="success_rate",
            markers=True,
            title=f"{brand} – Past Success Trend"
        )
        past_chart = fig_to_html(fig_past)

        new_df = pd.DataFrame({
            "metric": ["Rating", "Popularity", "Hits Score", "AI Success Score"],
            "value": [rating, popularity, hits, score]
        })

        fig_new = px.bar(
            new_df,
            x="metric",
            y="value",
            title=f"{model_name} – Predicted Performance Overview",
            color="metric"
        )
        new_chart = fig_to_html(fig_new)

        timeline_df = brand_df.copy()
        timeline_df.loc[len(timeline_df)] = [brand, model_name, launch_year, rating, popularity, hits, score]

        fig_combined = px.line(
            timeline_df.sort_values("launch_year"),
            x="launch_year",
            y="success_rate",
            markers=True,
            title=f"{brand}: Old Models vs New Model Trend Line",
            color="model"
        )
        combined_chart = fig_to_html(fig_combined)

        return render_template(
            "prediction_result.html",
            score=score,
            brand=brand,
            model=model_name,
            hashtags=hashtags,
            captions=captions,
            expected_views=expected_views,
            platform=platform,
            past_chart=past_chart,
            new_chart=new_chart,
            combined_chart=combined_chart
        )

    return render_template("prediction.html", brands=brands)


# ============================================================
#  RUN APP
# ============================================================

if __name__ == "__main__":
    app.run(debug=True, port=5000)
