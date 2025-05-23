import sys
import os
import re
import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
from collections import Counter
from fuzzywuzzy import process
from paddleocr import PaddleOCR
from twilio.rest import Client
import streamlit as st
import mimetypes
import google.generativeai as genai

# Fix unicode issue on Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Twilio credentials
TWILIO_SID = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_TOKEN = "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
TWILIO_PHONE = "xxxxxxxxxxxx"
twilio_client = Client(TWILIO_SID, TWILIO_TOKEN)

# Gemini API Key
genai.configure(api_key="xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx")

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_names = ["High", "Medium"]

# OCR
ocr = PaddleOCR(use_angle_cls=True, lang='en')

# Image transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load model
def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found!")
    try:
        model = torch.load(model_path, map_location=device, weights_only=False)
        print("✅ Full model loaded successfully.")
    except:
        print("⚠️ Full model load failed! Trying to load weights into ResNet-50...")
        model = models.resnet50(pretrained=False)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 2)
        )
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model weights loaded into ResNet-50.")
    model = model.to(device)
    model.eval()
    return model

model = load_model("D:/final project/AI Driven  IoT Application for Automated Emergency Response and Accident Management/working code/injury_severity_full_model.pth")

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(image)
        _, predicted_class = torch.max(output, 1)
    return class_names[predicted_class.item()]

def predict_video(video_path, frame_interval=10):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            input_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input_tensor)
                _, pred = torch.max(output, 1)
                predictions.append(pred.item())
        count += 1
    cap.release()
    if predictions:
        final = max(set(predictions), key=predictions.count)
        return class_names[final]
    return "No valid frames"

def extract_license_plate_text(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    results = ocr.ocr(gray, cls=True)
    if not results or results[0] is None:
        return "No plate detected"
    texts = []
    for line in results[0]:
        bbox, (text, _) = line
        height = bbox[2][1] - bbox[0][1]
        if height > 20:
            texts.append(text)
    plate = " ".join(texts)
    return re.sub(r'[^A-Z0-9]', '', plate.upper()) or "No plate detected"

def process_video(file_path, frame_interval=10):
    cap = cv2.VideoCapture(file_path)
    plates = []
    count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            plate = extract_license_plate_text(frame)
            if plate != "No plate detected":
                plates.append(plate)
        count += 1
    cap.release()
    return Counter(plates).most_common(1)[0][0] if plates else "No plate detected"

def predict_severity(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith("image"):
        return predict_image(file_path)
    elif mime_type and mime_type.startswith("video"):
        return predict_video(file_path)
    return "Unsupported file"

vehicle_df = pd.read_csv(r"D:\final project\AI Driven  IoT Application for Automated Emergency Response and Accident Management\working code\dataset\Book1_m.csv")
family_df = pd.read_csv(r"D:\final project\AI Driven  IoT Application for Automated Emergency Response and Accident Management\working code\dataset\Book2_m.csv")
vehicle_df['Vehicle Number'] = vehicle_df['Vehicle Number'].str.replace(r'\s+', '', regex=True).str.upper()
family_df['Aadhaar_Number'] = family_df['Aadhaar_Number'].astype(str).str.strip()
family_df['Mobile_Number'] = family_df['Mobile_Number'].astype(str).str.strip().str.replace(r"\D", "", regex=True)

def get_vehicle_details(plate):
    plate = plate.replace(" ", "").upper()
    match = vehicle_df[vehicle_df['Vehicle Number'] == plate]
    if not match.empty:
        row = match.iloc[0]
    else:
        closest = process.extractOne(plate, vehicle_df['Vehicle Number'])
        if closest and closest[1] > 75:
            row = vehicle_df[vehicle_df['Vehicle Number'] == closest[0]].iloc[0]
        else:
            return {"Aadhar_Number": "Not Found", "Police_Number": "Not Found", "Ambulance_Number": "Not Found"}
    return {
        "Aadhar_Number": str(row["Aadhar Number"]),
        "Police_Number": str(row["Police"]),
        "Ambulance_Number": str(row["Ambulance"])
    }

def find_family_members(aadhar_number):
    match = family_df[family_df['Aadhaar_Number'] == str(aadhar_number)]
    if not match.empty:
        ration = match['Ration_Card_Number'].values[0]
        members = family_df[family_df['Ration_Card_Number'] == ration]
        return members[['Family_Member', 'Mobile_Number', 'Address']].to_dict(orient='records')
    return None

def is_valid_number(number):
    invalid_prefixes = ["000", "976", "1860", "1800"]
    for prefix in invalid_prefixes:
        if number.startswith(prefix):
            return False
    return True

def send_sms(to, message):
    try:
        if not to.startswith("+"):
            to = "+91" + to.strip()
        clean_number = to.replace(" ", "").replace("-", "")
        if not is_valid_number(clean_number[3:]):
            print(f"⚠️ Skipping premium/service number: {clean_number}")
            return
        twilio_client.messages.create(
            body=message,
            from_=TWILIO_PHONE,
            to=clean_number
        )
        print(f"✅ SMS sent to {clean_number}")
    except Exception as e:
        print(f"❌ Failed to send SMS: {e}")

def get_first_aid_suggestion(severity, license_plate_info):
    prompt = f"""
    An accident has occurred involving a vehicle with plate number: {license_plate_info}.
    The injury severity is classified as **{severity}**.

    1. Provide the appropriate **first aid procedures** that should be done immediately before professional help arrives.
    2. List **commonly used over-the-counter medicines** or basic medical supplies that can be used in such cases.
       - If the injury is minor, suggest ointments, pain relievers, antiseptics, etc.
       - If the injury is serious, suggest what medicines can help with bleeding, shock, or swelling (if applicable).
    3. Present this in a clear and easy-to-read format.
    """
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")  # ✅ Correct name
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"❌ Error fetching first aid info: {e}"



# Streamlit UI
st.set_page_config(page_title="Emergency Response System")
st.title("🚨 AI-Driven Emergency Accident Management")

uploaded = st.file_uploader("Upload Accident Image/Video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded:
    os.makedirs("temp", exist_ok=True)
    path = os.path.join("temp", uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.read())

    severity = predict_severity(path)
    st.success(f"🚑 Injury Severity: **{severity}**")

    license_plate = process_video(path) if path.endswith(('.mp4', '.avi', '.mov')) else extract_license_plate_text(cv2.imread(path))
    st.info(f"🔍 Detected Plate: **{license_plate}**")

    vehicle_details = get_vehicle_details(license_plate)
    st.write("📄 Vehicle Info")
    st.json(vehicle_details)

    family = find_family_members(vehicle_details["Aadhar_Number"])
    st.write("👨‍👩‍👧‍👦 Family Contacts")
    st.json(family)

    # Get and show first aid suggestion
    first_aid = get_first_aid_suggestion(severity, license_plate)
    st.subheader("🩹 Suggested First Aid Procedure")
    st.write(first_aid)

    if severity == "High":
        numbers = [vehicle_details["Police_Number"], vehicle_details["Ambulance_Number"]]
        for number in numbers:
            if number != "Not Found":
                send_sms(number, f"🚨 Serious accident involving vehicle {license_plate} reported.")
        if family:
            for member in family:
                send_sms(member["Mobile_Number"], f"⚠️ {member['Family_Member']} may have been in a serious accident.")
