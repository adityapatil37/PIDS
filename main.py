from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, send_file
from pymongo import MongoClient
import os
import cv2
import torch
import uuid
import numpy as np
from torchreid.utils import FeatureExtractor
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from bson.objectid import ObjectId
from io import BytesIO
import pandas as pd
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from fpdf import FPDF



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
access_col = db["access_control"]
alerts_col = db["alerts"]

now = datetime.now(ZoneInfo("Asia/Kolkata"))

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
        now = datetime.now(ZoneInfo("Asia/Kolkata"))
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

from datetime import datetime

@app.route("/history", methods=["GET", "POST"])
def history():
    query_name = None
    results = []
    camera_filter = None
    start_date = None
    end_date = None
    sort_order = -1  # newest first by default

    if request.method == "POST":
        query_name = request.form.get("name", "").strip()
        camera_filter = request.form.get("camera", "").strip()
        start_date = request.form.get("start_date")
        end_date = request.form.get("end_date")
        sort_order = int(request.form.get("sort_order", "-1"))

        query = {}
        if query_name:
            query["person_name"] = {"$regex": f"^{query_name}$", "$options": "i"}

        if camera_filter:
            query["camera_name"] = {"$regex": f"^{camera_filter}$", "$options": "i"}

        # Date range filter
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = datetime.strptime(start_date, "%Y-%m-%d")
            if end_date:
                query["timestamp"]["$lte"] = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

        # Fetch results
        results = list(
            history_col.find(query).sort("timestamp", sort_order)
        )

        # Format time for UI
        for r in results:
            r["formatted_time"] = r["timestamp"].strftime("%Y-%m-%d %H:%M:%S")
            if r.get("thumbnail") and not r["thumbnail"].startswith("/static/"):
                r["thumbnail"] = "/static/" + os.path.relpath(r["thumbnail"], "static").replace("\\", "/")

    # For filter dropdowns
    all_cameras = sorted(set([x["camera_name"] for x in history_col.find({}, {"camera_name": 1})]))

    return render_template(
        "history.html",
        results=results,
        query_name=query_name,
        all_cameras=all_cameras,
        camera_filter=camera_filter,
        start_date=start_date,
        end_date=end_date,
        sort_order=sort_order,
    )

@app.route("/export/excel/<name>")
def export_excel(name):
    logs = list(history_col.find({"person_name": {"$regex": f"^{name}$", "$options": "i"}})
                          .sort("timestamp", -1))

    if not logs:
        return "No records found", 404

    df = pd.DataFrame(logs)
    df["_id"] = df["_id"].astype(str)
    df["timestamp"] = df["timestamp"].astype(str)

    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name="History")

    output.seek(0)
    filename = f"{name}_history_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx"
    return send_file(output, as_attachment=True, download_name=filename, mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")


@app.route("/export/pdf/<name>")
def export_pdf(name):
    logs = list(history_col.find({"person_name": {"$regex": f"^{name}$", "$options": "i"}})
                          .sort("timestamp", -1))

    if not logs:
        return "No records found", 404

    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    pdf.setFont("Helvetica", 12)
    pdf.drawString(200, height - 40, f"Attendance History for {name}")
    y = height - 80

    for log in logs:
        pdf.drawString(50, y, f"Time: {log['timestamp']} | Status: {log.get('status', 'Unknown')}")
        y -= 20
        if y < 50:
            pdf.showPage()
            pdf.setFont("Helvetica", 12)
            y = height - 50

    pdf.save()
    buffer.seek(0)
    filename = f"{name}_history_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
    return send_file(buffer, as_attachment=True, download_name=filename, mimetype="application/pdf")

@app.route("/logs/<path:filename>")
def logs(filename):
    return send_from_directory("thumbnails", filename)

@app.route("/access-control")
def access_control():
    # Fetch all access configs
    access_docs = list(access_col.find())
    people = list(people_col.find({}, {"name": 1}))  # only get names
    cameras = sorted({d["camera_name"] for d in access_docs} | {"Camera_1", "Camera_2", "Camera_3"})
    return render_template("access_control.html", access_docs=access_docs, people=people, cameras=cameras)


@app.route("/access-control/update", methods=["POST"])
def update_access_control():
    cam_name = request.form.get("camera_name")
    allowed_people = request.form.getlist("allowed_people")

    if not cam_name:
        flash("Camera name missing!", "error")
        return redirect(url_for("access_control"))

    access_col.update_one(
        {"camera_name": cam_name},
        {"$set": {"allowed_people": allowed_people}},
        upsert=True
    )
    flash(f"Access list updated for {cam_name}", "success")
    return redirect(url_for("access_control"))


@app.route("/alerts", methods=["GET"])
def alerts():
    """Display all generated alerts from the alerts collection."""
    alerts_data = list(alerts_col.find().sort("timestamp", -1))  # newest first
    return render_template("alerts.html", alerts=alerts_data)


@app.route("/export_alerts_excel")
def export_alerts_excel():
    alerts_data = list(alerts_col.find())
    if not alerts_data:
        return "No data available", 404

    df = pd.DataFrame(alerts_data)
    df.drop("_id", axis=1, inplace=True)
    file_path = "static/exports/alerts_log.xlsx"
    df.to_excel(file_path, index=False)
    return send_file(file_path, as_attachment=True)

@app.route("/export_alerts_pdf")
def export_alerts_pdf():
    alerts_data = list(alerts_col.find())
    if not alerts_data:
        return "No data available", 404

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Alerts Log", ln=True, align="C")
    pdf.ln(10)

    for alert in alerts_data:
        pdf.multi_cell(0, 10, txt=f"Person: {alert.get('person_name')}\n"
                                  f"Camera: {alert.get('camera_name')}\n"
                                  f"Type: {alert.get('alert_type')}\n"
                                  f"Timestamp: {alert.get('timestamp')}\n", border=1)
        pdf.ln(5)

    file_path = "static/exports/alerts_log.pdf"
    pdf.output(file_path)
    return send_file(file_path, as_attachment=True)

if __name__ == "__main__":
    os.makedirs(CROP_FOLDER, exist_ok=True)
    os.makedirs("crops/enrolled", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
