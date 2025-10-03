from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
from pymongo import MongoClient
import os
import cv2
import torch
import uuid
import numpy as np
from torchreid.utils import FeatureExtractor
from werkzeug.utils import secure_filename
import datetime
from zoneinfo import ZoneInfo
from bson.objectid import ObjectId



# Flask
app = Flask(__name__)
app.secret_key = "your_secret_key"

UPLOAD_FOLDER = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["person_reid"]
people_col = db["people"]
history_col = db["track_history"]

now = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))

# Torchreid extractor
extractor = FeatureExtractor(
    model_name='osnet_x1_0',
    model_path='C:/Users/adity/.cache/torch/checkpoints/osnet_x1_0_imagenet.pth',
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

CROP_FOLDER = "crops/unknown"

def extract_feature(img_path):
    img = cv2.imread(img_path)
    if img is None:
        return None
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    feat = extractor([img_rgb])
    return feat[0].cpu().numpy()

@app.route("/")
def dashboard():
    """Main Dashboard with links to all features."""
    return render_template("dashboard.html")

@app.route("/enrollment")
def enrollment():
    # list unknown crops
    images = os.listdir(CROP_FOLDER) if os.path.exists(CROP_FOLDER) else []
    images = [f for f in images if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    return render_template("enrollment.html", images=images)



@app.route("/enroll", methods=["POST"])
def enroll():
    name = request.form.get("name")
    role = request.form.get("role")
    selected_images = request.form.getlist("selected")
    uploaded_files = request.files.getlist("uploads")

    if not name or not role:
        return "Missing fields", 400

    features = []

    # --- From saved crops ---
    for img_file in selected_images:
        img_path = os.path.join(CROP_FOLDER, img_file)
        feat = extract_feature(img_path)
        if feat is not None:
            features.append(feat.tolist())

    # --- From uploaded files ---
    for file in uploaded_files:
        if file and file.filename != "":
            filename = secure_filename(file.filename)
            save_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(save_path)

            feat = extract_feature(save_path)
            if feat is not None:
                features.append(feat.tolist())

    if features:
        now = datetime.datetime.now(ZoneInfo("Asia/Kolkata"))
        record = {
            "name": name,
            "role": role,
            "features": features,
            "registered_at": now.strftime('%Y-%m-%d %H:%M:%S')
        }
        people_col.insert_one(record)
        print(f"[+] Enrolled {name} ({role}) with {len(features)} images")

        # Optional: move used crops to archive
        for img_file in selected_images:
            os.rename(os.path.join(CROP_FOLDER, img_file), f"crops/enrolled/{img_file}")

    return redirect(url_for("enrollment"))

@app.route("/people")
def people():
    """Show all registered people with images."""
    all_people = list(people_col.find())
    return render_template("people.html", people=all_people)

@app.route("/edit/<person_id>", methods=["GET", "POST"])
def edit_person(person_id):
    """Edit person details."""
    person = people_col.find_one({"_id": ObjectId(person_id)})

    if not person:
        flash("Person not found.", "danger")
        return redirect(url_for("people"))

    if request.method == "POST":
        name = request.form.get("name")
        role = request.form.get("role")

        # Ensure person["images"] exists
        new_image_paths = person.get("images", [])

        # Handle optional new uploads
        uploaded_files = request.files.getlist("images")
        for file in uploaded_files:
            if file and file.filename != "":
                filename = secure_filename(f"{uuid.uuid4()}_{file.filename}")
                filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
                file.save(filepath)
                new_image_paths.append(filepath)

        people_col.update_one(
            {"_id": ObjectId(person_id)},
            {"$set": {"name": name, "role": role, "images": new_image_paths}}
        )
        flash("Person updated successfully!", "success")
        return redirect(url_for("people"))

    # Ensure images list exists for rendering
    if "images" not in person:
        person["images"] = []

    return render_template("edit_person.html", person=person)


@app.route("/delete/<person_id>")
def delete_person(person_id):
    """Delete a person and their images."""
    person = people_col.find_one({"_id": ObjectId(person_id)})
    if person:
        # Delete associated images
        for img_path in person.get("images", []):
            if os.path.exists(img_path):
                os.remove(img_path)
        people_col.delete_one({"_id": ObjectId(person_id)})
        flash("Person deleted successfully!", "success")
    else:
        flash("Person not found.", "danger")
    return redirect(url_for("people"))

@app.route("/history", methods=["GET", "POST"])
def history():
    query_name = None
    results = []

    if request.method == "POST":
        query_name = request.form.get("name", "").strip()
        if query_name:
            # Fetch logs for that person sorted by latest first
            results = list(history_col.find({"person_name": {"$regex": f"^{query_name}$", "$options": "i"}})
                                         .sort("timestamp", -1))
    return render_template("history.html", results=results, query_name=query_name)

@app.route("/logs/<path:filename>")
def logs(filename):
    return send_from_directory("thumbnails", filename)

if __name__ == "__main__":
    os.makedirs(CROP_FOLDER, exist_ok=True)
    os.makedirs("crops/enrolled", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
