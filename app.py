from flask import Flask, render_template, request, session, redirect
from utils.database import init_db, save_input, get_overall_status, get_user_history, get_dashboard_stats
import os
import sys
import time
import emoji
import re

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.loader import load_model, load_image_models, load_emoji_model
from utils.predictor import predict_text, predict_image, predict_emoji
from utils.gradcam import gradcam_plus_plus
from utils.shap_explainer import get_shap_explanation

app = Flask(__name__)
app.secret_key = "secret123"

# ================= INIT DB =================
init_db()

# ================= LOAD MODELS =================
print("🚀 Loading models...")

text_model = load_model("models/stage3_model")

image_models = load_image_models([
    "models/final_model_0.h5",
    "models/final_model_1.h5",
    "models/final_model_2.h5"
])

emoji_model = load_emoji_model("models/emoji_model.keras")

print("✅ Models loaded")


# ================= HELPERS =================
def get_dep_score(probs):
    # ✅ FIX: force python float
    return float(probs[1]) if len(probs) > 1 else float(probs[0])


def safe(x):
    # ✅ FIX: ensure JSON safe
    return float(round(float(x) * 100, 2)) if x is not None else None


def remove_emojis(text):
    text = emoji.replace_emoji(text, replace='')
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


# ================= NEW FUNCTIONS =================
def generate_explanation(t_score, i_score, e_score):
    reasons = []

    if t_score and t_score > 0.6:
        reasons.append("Text shows negative emotional patterns")

    if e_score and e_score > 0.6:
        reasons.append("Emojis indicate sadness or distress")

    if i_score and i_score > 0.6:
        reasons.append("Facial expression appears low or inactive")

    if len(reasons) == 0:
        return "No strong depression signals detected"

    return ", ".join(reasons)


# ✅ REAL CONFIDENCE (MODEL AGREEMENT BASED)
def get_confidence(t_score, i_score, e_score):

    scores = [s for s in [t_score, i_score, e_score] if s is not None]

    if not scores:
        return "Low Confidence"

    # variance = disagreement
    mean = sum(scores) / len(scores)
    variance = sum((s - mean) ** 2 for s in scores) / len(scores)

    if variance < 0.01:
        return "High Confidence"
    elif variance < 0.05:
        return "Moderate Confidence"
    else:
        return "Low Confidence"


def get_risk_level(score):
    if score > 0.7:
        return "High Risk"
    elif score > 0.5:
        return "Moderate Risk"
    else:
        return "Low Risk"


def get_suggestions(risk):
    if risk == "High Risk":
        return [
            "Consider talking to a mental health professional",
            "Reach out to close friends or family",
            "Avoid isolation and stay connected",
            "Practice relaxation techniques (breathing, meditation)"
        ]
    elif risk == "Moderate Risk":
        return [
            "Take regular breaks and reduce stress",
            "Engage in physical activity or hobbies",
            "Maintain a healthy sleep routine",
            "Talk to someone you trust"
        ]
    else:
        return [
            "Maintain your current healthy lifestyle",
            "Stay socially active",
            "Keep monitoring your mental well-being"
        ]


# ================= LOGIN =================
@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user_id = request.form.get("user_id")

        if user_id:
            session["user_id"] = user_id
            return redirect("/dashboard")
        else:
            return render_template("login.html", error="Enter user id")

    return render_template("login.html")


# ================= DASHBOARD =================
@app.route("/dashboard")
def dashboard():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]

    history = get_user_history(user_id)

    try:
        stats = get_dashboard_stats(user_id)
    except:
        stats = {
            "total_posts": 0,
            "text_posts": 0,
            "image_posts": 0,
            "emoji_posts": 0,
            "avg_fusion": 0,
            "depression_count": 0
        }

    history_scores = []

    for row in history:
        try:
            history_scores.append(round(float(row[2]) * 100, 2))
        except:
            history_scores.append(0)

    return render_template(
        "dashboard.html",
        stats=stats,
        history_scores=history_scores,
        history=history
    )


# ================= INPUT =================
@app.route("/")
def input_page():
    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]

    history = get_user_history(user_id)
    history_scores = []

    for row in history:
        try:
            score = row[2]
            history_scores.append(round(score * 100, 2))
        except:
            history_scores.append(0)

    return render_template(
        "input.html",
        history=history,
        history_scores=history_scores
    )


# ================= RESULT =================
@app.route("/result", methods=["POST"])
def result():

    if "user_id" not in session:
        return redirect("/login")

    user_id = session["user_id"]

    text = request.form.get("text", "")
    image_file = request.files.get("image")

    scores = []
    heatmap = None
    attention = []
    filename = None

    # ---------- TEXT ----------
    clean_text = remove_emojis(text)

    if clean_text and len(clean_text.split()) > 0:
        t_probs = predict_text(text_model, clean_text)
        t_score = get_dep_score(t_probs)
        t_label = "Depression" if t_score > 0.5 else "Normal"
        scores.append(("text", t_score))

        try:
            attention = get_shap_explanation(text_model, clean_text)
        except:
            attention = []
    else:
        t_label, t_score = "-", None

    # ---------- IMAGE ----------
    if image_file and image_file.filename != "":
        filename = f"{user_id}_{int(time.time())}.jpg"
        image_path = os.path.join("static", filename)

        image_file.save(image_path)
        image_file.seek(0)

        i_probs = predict_image(image_models, image_file)
        i_score = get_dep_score(i_probs)
        i_label = "Depression" if i_score > 0.5 else "Normal"
        scores.append(("image", i_score))

        try:
            heatmap = gradcam_plus_plus(image_models[0], image_path)
        except:
            heatmap = None
    else:
        i_label, i_score = "-", None

    # ---------- EMOJI ----------
    if text:
        e_probs = predict_emoji(emoji_model, text)

        if e_probs is not None:
            e_score = get_dep_score(e_probs)
            e_label = "Depression" if e_score > 0.4 else "Normal"
            scores.append(("emoji", e_score))
        else:
            e_label, e_score = "-", None
    else:
        e_label, e_score = "-", None

    # ---------- FUSION ----------
    if scores:
        total = 0
        weight_sum = 0

        for source, score in scores:
            weight = 0.6 if source == "text" else 0.3 if source == "emoji" else 0.1
            total += score * weight
            weight_sum += weight

        fusion_score = total / weight_sum
    else:
        fusion_score = 0.5

    f_label = "Depression" if fusion_score > 0.5 else "Normal"

    # ---------- FIXED CONFIDENCE ----------
    confidence = get_confidence(t_score, i_score, e_score)

    explanation = generate_explanation(t_score, i_score, e_score)
    risk = get_risk_level(fusion_score)
    suggestions = get_suggestions(risk)

    # ---------- SAVE ----------
    save_input(
        user_id,
        text,
        filename if filename else "",
        float(t_score) if t_score else 0,
        float(i_score) if i_score else 0,
        float(e_score) if e_score else 0,
        float(fusion_score)
    )

    overall_label, overall_score = get_overall_status(user_id)

    # ---------- SESSION ----------
    session["results"] = {
        "text": (t_label, safe(t_score)),
        "image": (i_label, safe(i_score)),
        "emoji": (e_label, safe(e_score)),
        "fusion": (f_label, safe(fusion_score)),
        "attention": attention,
        "heatmap": str(heatmap) if heatmap is not None else None,  # ✅ FIX
        "image_path": filename,
        "overall": (overall_label, overall_score),
        "explanation": explanation,
        "confidence": confidence,
        "risk": risk,
        "suggestions": suggestions
    }

    return redirect("/results")


# ================= SHOW =================
@app.route("/results")
def show_results():
    if "results" not in session:
        return redirect("/")
    return render_template("result.html", results=session["results"])


@app.route("/analysis")
def analysis():
    if "results" not in session:
        return redirect("/")
    return render_template("analysis.html", results=session["results"])


@app.route("/history")
def history():
    if "user_id" not in session:
        return redirect("/login")

    data = get_user_history(session["user_id"])
    return render_template("history.html", history=data)


@app.route("/logout")
def logout():
    session.clear()
    return redirect("/login")


if __name__ == "__main__":
    app.run(debug=False)