# ui_ai_enhanced.py — AI Agro Assistant (Corrected & runnable)
# Cleaned and fixed version of your enhanced UI.
# Fixes applied:
# - Tk variables created after root window
# - Syntax errors removed (broken strings, misplaced newlines)
# - Robust imports for arduino_reader and disease_module (safe stubs if missing)
# - Clear single language-change handler and proper trace binding
# - Minor UX improvements and safer threading

import os
import time
import locale
import threading
import random
from datetime import datetime

import customtkinter as ctk
from tkinter import messagebox
from PIL import Image, ImageTk
import cv2
import numpy as np

# attempt to import sklearn; if not available, use rule-based fallback
try:
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.neighbors import KNeighborsClassifier
    import joblib
    SKLEARN_AVAILABLE = True
except Exception as e:
    print("sklearn not available, AI modules will use rule-based fallbacks.", e)
    SKLEARN_AVAILABLE = False

# local helper (existing modules) — provide safe fallbacks if missing
try:
    import arduino_reader
except Exception:
    arduino_reader = None

try:
    import disease_module
except Exception:
    disease_module = None

# --- Fertilizer translations (for multi-language display) ---
fertilizer_translations = {
    "Apply Lime (Calcium)": {"English":"Apply Lime (Calcium)", "Marathi":"लायम वापरा (कॅल्शियम)", "Hindi":"चूना लगाएँ (कैल्शियम)", "Tamil":"லைம் பயன்படுத்தவும் (கால்சியம்)", "Kannada":"ಲೈಂ ಅನ್ವಯಿಸಿ (ಕ್ಯಾಲ್ಸಿಯಂ)"},
    "Use balanced NPK (limited Urea)": {"English":"Use balanced NPK (limited Urea)", "Marathi":"संतुलित NPK वापरा (मर्यादित युरिया)", "Hindi":"संतुलित NPK प्रयोग करें (सीमित यूरिया)", "Tamil":"சரியான NPK பயன்படுத்தவும் (குறைந்த யூரியா)", "Kannada":"ಸಂತುಲನ NPK ಬಳಸಿ (Marita Urea)"},
    "Apply Sulphur (acidifying)": {"English":"Apply Sulphur (acidifying)", "Marathi":"सल्फर लावा (अम्लीकरण)", "Hindi":"सल्फर लगाएँ (अम्लीय)", "Tamil":"சல்பர் செலுத்தவும் (அமிலப்படுத்துதல்)", "Kannada":"ಸಲ್ಫರ್ ಅನ್ವಯಿಸಿ (ಆಮೀಲಿಕರಣ)"},
    "Add Compost/Organic Matter": {"English":"Add Compost/Organic Matter", "Marathi":"कंपोस्ट/सेंद्रिय द्रव्य जोडा", "Hindi":"कम्पोस्ट/जैविक पदार्थ जोड़ें", "Tamil":"கம்போஸ்ட்/உர வாயு சேர்க்கவும்", "Kannada":"ಕಂಪೋಸ್ಟ್/ಸೇಂದ್ರಿಯ ವಸ್ತು ಸೇರಿಸಿ"}
}

# voice (gTTS) helper — supports language selection; safe no-op if gTTS missing
try:
    from gtts import gTTS
    import playsound
    TTS_AVAILABLE = True
except Exception as e:
    print("gTTS/playsound not available:", e)
    TTS_AVAILABLE = False


def speak_text_online(text, lang_code="en"):
    if not TTS_AVAILABLE:
        print("TTS disabled. Message:", text)
        return
    try:
        lang_map = {"English":"en", "Marathi":"mr", "Kannada":"kn", "Tamil":"ta", "Hindi":"hi",
                    "en":"en","mr":"mr","kn":"kn","ta":"ta","hi":"hi"}
        lang = lang_map.get(lang_code, "en")
        filename = f"voice_{random.randint(1000,9999)}.mp3"
        tts = gTTS(text=text, lang=lang)
        tts.save(filename)
        playsound.playsound(filename, True)
        time.sleep(0.2)
        os.remove(filename)
    except Exception as e:
        print("gTTS error:", e)

# ----------------- AI COMPONENTS (TRAIN/SAVE/LOAD) -----------------
MODEL_DIR = "ai_models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Simple synthetic-data trainers — small and deterministic for local use
# (Keep these as-is; they only run if sklearn + joblib available)
def train_fertilizer_model():
    if not SKLEARN_AVAILABLE:
        return None
    crops = ["Wheat","Rice","Sugarcane","Cotton","Soybean","Maize","Tomato","Potato","Onion","Banana"]
    X = []
    y = []
    for i, crop in enumerate(crops):
        for ph in [4.5,5.5,6.0,6.8,7.2,7.8,8.5]:
            X.append([ph, i])
            if ph < 5.8:
                lbl = 0
            elif ph <= 7.2:
                lbl = 1
            elif ph <= 8.0:
                lbl = 2
            else:
                lbl = 3
            y.append(lbl)
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=30, random_state=42)
    clf.fit(X, y)
    joblib.dump((clf, crops), os.path.join(MODEL_DIR, "fert_recommender.joblib"))
    return (clf, crops)


def train_yield_model():
    if not SKLEARN_AVAILABLE:
        return None
    crops = ["Wheat","Rice","Sugarcane","Cotton","Soybean","Maize","Tomato","Potato","Onion","Banana"]
    X = []; y = []
    random.seed(42)
    for i, crop in enumerate(crops):
        base = 3.0 + (i % 5) * 0.5
        for ph in [5.0,5.8,6.5,7.0,7.5,8.0]:
            for rain in [200,400,600,800]:
                for temp in [18,22,26,30]:
                    for fert_amt in [20,50,80]:
                        X.append([ph, rain, temp, fert_amt, i])
                        ph_factor = max(0.5, 1 - abs(ph - 6.8) * 0.08)
                        rain_factor = min(1.5, 0.5 + rain / 800.0)
                        fert_factor = 0.6 + (fert_amt / 100.0)
                        y.append(base * ph_factor * rain_factor * fert_factor + random.uniform(-0.3,0.3))
    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor(n_estimators=40, random_state=1)
    reg.fit(X, y)
    joblib.dump((reg, crops), os.path.join(MODEL_DIR, "yield_predictor.joblib"))
    return (reg, crops)


def train_soil_classifier():
    if not SKLEARN_AVAILABLE:
        return None
    types = ["Sandy","Loamy","Clay"]
    X = []; y = []
    for ph in [5.0,6.0,7.0,8.0]:
        for moisture in [10,20,30,40,50,60]:
            for ec in [0.2,0.5,1.0,1.5,2.0]:
                X.append([ph, moisture, ec])
                if moisture < 20 and ec < 0.6:
                    y.append(0)
                elif 20 <= moisture < 45 and ec < 1.2:
                    y.append(1)
                else:
                    y.append(2)
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X, y)
    joblib.dump((knn, types), os.path.join(MODEL_DIR, "soil_classifier.joblib"))
    return (knn, types)

FERT_MODEL = None; YIELD_MODEL = None; SOIL_MODEL = None
if SKLEARN_AVAILABLE:
    try:
        FERT_MODEL = joblib.load(os.path.join(MODEL_DIR, "fert_recommender.joblib"))
    except Exception:
        FERT_MODEL = train_fertilizer_model()
    try:
        YIELD_MODEL = joblib.load(os.path.join(MODEL_DIR, "yield_predictor.joblib"))
    except Exception:
        YIELD_MODEL = train_yield_model()
    try:
        SOIL_MODEL = joblib.load(os.path.join(MODEL_DIR, "soil_classifier.joblib"))
    except Exception:
        SOIL_MODEL = train_soil_classifier()

# ----------------- UI + Handlers -----------------
# language setup
LANG_OPTIONS = {"English":"en","Marathi":"mr","Hindi":"hi","Tamil":"ta","Kannada":"kn"}

# detect locale (optional)
def detect_lang_code():
    loc = locale.getdefaultlocale()[0] or "en"
    if isinstance(loc, str):
        if loc.startswith("mr"): return "mr"
        if loc.startswith("hi"): return "hi"
        if loc.startswith("kn"): return "kn"
        if loc.startswith("ta"): return "ta"
    return "en"

CURRENT_LANG_CODE = detect_lang_code()

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")
app = ctk.CTk()
app.geometry("980x720")
app.title("AI Agro Assistant — Enhanced")

# Header + language menu
welcome_lbl = ctk.CTkLabel(app, text="AI Agro Assistant — Enhanced with more AI", font=("Arial", 20, "bold"))
welcome_lbl.pack(pady=8)

# create StringVar AFTER root
selected_lang = ctk.StringVar(value="English")
lang_menu = ctk.CTkOptionMenu(app, values=list(LANG_OPTIONS.keys()), variable=selected_lang)
lang_menu.place(x=820, y=18)

# language change handler
def on_lang_change(choice):
    global CURRENT_LANG_CODE
    CURRENT_LANG_CODE = LANG_OPTIONS.get(choice, "en")
    msgs = {
        "en": f"You selected {choice}.",
        "mr": "तुम्ही मराठी निवडली आहे.",
        "hi": "आपने हिंदी चुनी है।",
        "ta": "நீங்கள் தமிழ் தேர்ந்தெடுத்துள்ளீர்கள்.",
        "kn": "ನೀವು ಕನ್ನಡ ಆಯ್ಕೆಮಾಡಿದ್ದೀರಿ."
    }
    t = msgs.get(CURRENT_LANG_CODE, msgs["en"]) if isinstance(msgs.get(CURRENT_LANG_CODE), str) else msgs["en"]
    threading.Thread(target=speak_text_online, args=(t, CURRENT_LANG_CODE), daemon=True).start()

# attach trace AFTER handler defined
selected_lang.trace_add('write', lambda *args: on_lang_change(selected_lang.get()))

main_frame = ctk.CTkFrame(app, width=940, height=600, corner_radius=12)
main_frame.pack(padx=10, pady=6)

left_frame = ctk.CTkFrame(main_frame, width=460, height=560, corner_radius=8)
left_frame.place(x=10, y=10)
right_frame = ctk.CTkFrame(main_frame, width=450, height=560, corner_radius=8)
right_frame.place(x=480, y=10)

# Left: pH inputs (manual + Arduino) and crop selectors
ctk.CTkLabel(left_frame, text="Soil & Crop Inputs", font=("Arial",16,"bold")).pack(pady=10)
ph_entry = ctk.CTkEntry(left_frame, placeholder_text="Soil pH (e.g., 6.8)")
ph_entry.pack(pady=6)
# add a dropdown to pick input source
ph_source = ctk.StringVar(value="Manual")
ph_source_menu = ctk.CTkOptionMenu(left_frame, values=["Manual","Arduino"], variable=ph_source)
ph_source_menu.pack(pady=4)

moist_entry = ctk.CTkEntry(left_frame, placeholder_text="Soil moisture (%) — optional")
moist_entry.pack(pady=6)
ec_entry = ctk.CTkEntry(left_frame, placeholder_text="EC (dS/m) — optional")
ec_entry.pack(pady=6)
ctk.CTkLabel(left_frame, text="Select Crop:").pack(pady=4)
CROPS = ["Wheat","Rice","Sugarcane","Cotton","Soybean","Maize","Tomato","Potato","Onion","Banana"]
selected_crop = ctk.StringVar(value=CROPS[0])
crop_menu = ctk.CTkOptionMenu(left_frame, values=CROPS, variable=selected_crop)
crop_menu.pack(pady=6)

# Arduino read: integrates with arduino_reader.read_ph_from_any_port()

def read_ph_from_arduino_threaded():
    def worker():
        try:
            if arduino_reader is None:
                messagebox.showerror("Arduino","arduino_reader module not found")
                return
            res = arduino_reader.read_ph_from_any_port()
            if res is None:
                messagebox.showerror("Arduino","No reading from Arduino")
                return
            # res may be tuple (value, port) or just value depending on implementation
            if isinstance(res, tuple) or isinstance(res, list):
                val, port = res
            else:
                val = res; port = "Unknown"
            ph_entry.delete(0,'end'); ph_entry.insert(0,str(val))
            messagebox.showinfo("Arduino",f"Read pH: {val} from {port}")
            threading.Thread(target=speak_text_online, args=(f"Arduino pH {val}", CURRENT_LANG_CODE), daemon=True).start()
        except Exception as e:
            print('Arduino read error', e)
            messagebox.showerror("Arduino","Error reading from Arduino")
    threading.Thread(target=worker, daemon=True).start()

arduino_btn = ctk.CTkButton(left_frame, text="🔌 Read pH from Arduino", command=read_ph_from_arduino_threaded, width=220, fg_color="#2f7bd6")
arduino_btn.pack(pady=6)

# NEW: fertilizer recommender (multi-language output)

def fertilizer_recommendation(pH, crop):
    if SKLEARN_AVAILABLE and FERT_MODEL is not None:
        clf, crops = FERT_MODEL
        crop_index = crops.index(crop) if crop in crops else 0
        pred = clf.predict([[pH, crop_index]])[0]
        mapping_key = {0: "Apply Lime (Calcium)", 1: "Use balanced NPK (limited Urea)", 2: "Apply Sulphur (acidifying)", 3: "Add Compost/Organic Matter"}.get(pred)
    else:
        if pH < 5.8:
            mapping_key = "Apply Lime (Calcium)"
        elif pH <= 7.2:
            mapping_key = "Use balanced NPK (limited Urea)"
        elif pH <= 8.0:
            mapping_key = "Apply Sulphur (acidifying)"
        else:
            mapping_key = "Add Compost/Organic Matter"
    # translate: find language name by code
    lang_name = [k for k,v in LANG_OPTIONS.items() if v==CURRENT_LANG_CODE]
    lang_name = lang_name[0] if lang_name else "English"
    translated = fertilizer_translations.get(mapping_key, {}).get(lang_name, mapping_key)
    return mapping_key, translated


def on_fert_recommend():
    source = ph_source.get()
    if source == "Arduino":
        # attempt to read from arduino first (async)
        read_ph_from_arduino_threaded()
        # small delay is still not guaranteed; prefer user to press again after read
        time.sleep(0.6)
    v = ph_entry.get().strip()
    if not v:
        messagebox.showwarning("Input", "Enter pH value first")
        return
    try:
        pval = float(v)
    except Exception:
        messagebox.showerror("Error","Enter numeric pH value")
        return
    crop = selected_crop.get()
    key, translated = fertilizer_recommendation(pval, crop)
    msg_en = f"Fertilizer suggestion for {crop} (pH {pval}): {key}"
    msg_local = translated
    # show both
    messagebox.showinfo("Fertilizer Recommender", msg_en + "\n" + msg_local)
    threading.Thread(target=speak_text_online, args=(msg_local or msg_en, CURRENT_LANG_CODE), daemon=True).start()

fert_btn = ctk.CTkButton(left_frame, text="Smart Fertilizer Recommender", command=on_fert_recommend)
fert_btn.pack(pady=8)

# Yield predictor UI
ctk.CTkLabel(left_frame, text="\nYield Predictor Inputs", font=("Arial",14)).pack(pady=8)
temp_entry = ctk.CTkEntry(left_frame, placeholder_text="Avg Temp (°C) e.g., 25")
temp_entry.pack(pady=6)
rain_entry = ctk.CTkEntry(left_frame, placeholder_text="Rainfall (mm) e.g., 500")
rain_entry.pack(pady=6)
fertamt_entry = ctk.CTkEntry(left_frame, placeholder_text="Fertilizer amount (kg/ha) e.g., 50")
fertamt_entry.pack(pady=6)


def predict_yield(pH, rainfall, temp, fert_amt, crop):
    if SKLEARN_AVAILABLE and YIELD_MODEL is not None:
        reg, crops = YIELD_MODEL
        crop_index = crops.index(crop) if crop in crops else 0
        pred = reg.predict([[pH, rainfall, temp, fert_amt, crop_index]])[0]
        return max(0.0, pred)
    # fallback heuristic
    base = 3.0
    ph_factor = max(0.5, 1 - abs(pH - 6.8)*0.08)
    rain_factor = min(1.5, 0.5 + rainfall / 800.0)
    fert_factor = 0.6 + (fert_amt / 100.0)
    return max(0.0, base * ph_factor * rain_factor * fert_factor)


def on_yield_predict():
    try:
        pH = float(ph_entry.get().strip())
        temp = float(temp_entry.get().strip())
        rain = float(rain_entry.get().strip())
        fert_amt = float(fertamt_entry.get().strip())
    except Exception as e:
        messagebox.showerror("Error","Please enter valid numeric inputs for all yield fields")
        return
    crop = selected_crop.get()
    y = predict_yield(pH, rain, temp, fert_amt, crop)
    msg = f"Predicted yield for {crop}: {y:.2f} tons/ha"
    messagebox.showinfo("Yield Predictor", msg)
    threading.Thread(target=speak_text_online, args=(msg, CURRENT_LANG_CODE), daemon=True).start()

yield_btn = ctk.CTkButton(left_frame, text="Predict Yield", command=on_yield_predict)
yield_btn.pack(pady=8)

# Soil classifier
ctk.CTkLabel(left_frame, text="\nSoil Type Classifier (pH, Moisture, EC)", font=("Arial",14)).pack(pady=8)

def classify_soil(pH, moisture, ec):
    if SKLEARN_AVAILABLE and SOIL_MODEL is not None:
        knn, types = SOIL_MODEL
        pred = knn.predict([[pH, moisture, ec]])[0]
        return types[pred]
    if moisture < 20 and ec < 0.6:
        return "Sandy"
    elif moisture < 45 and ec < 1.2:
        return "Loamy"
    else:
        return "Clay"


def on_soil_classify():
    try:
        pH = float(ph_entry.get().strip())
        moisture = float(moist_entry.get().strip() or 30)
        ec = float(ec_entry.get().strip() or 0.8)
    except Exception as e:
        messagebox.showerror("Error","Please enter valid numeric inputs for pH/moisture/ec")
        return
    stype = classify_soil(pH, moisture, ec)
    msg = f"Soil type: {stype}"
    messagebox.showinfo("Soil Classifier", msg)
    threading.Thread(target=speak_text_online, args=(msg, CURRENT_LANG_CODE), daemon=True).start()

soil_btn = ctk.CTkButton(left_frame, text="Classify Soil Type", command=on_soil_classify)
soil_btn.pack(pady=8)

# Right: disease detection (existing) + visualization + explainability hint
ctk.CTkLabel(right_frame, text="Plant Disease Detection (camera)", font=("Arial",16,"bold")).pack(pady=10)
status_lbl = ctk.CTkLabel(right_frame, text="Camera inactive")
status_lbl.pack(pady=6)

# Show last captured preview
preview_label = ctk.CTkLabel(right_frame, text="No image yet")
preview_label.pack(pady=6)

# Simple saliency/explainability: highlight green vs lesion areas via color mask

def compute_simple_explain(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25,40,40]); upper_green = np.array([95,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    lesion = cv2.bitwise_not(mask)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
    lesion = cv2.morphologyEx(lesion, cv2.MORPH_OPEN, k)
    overlay = frame.copy()
    overlay[lesion>0] = [0,0,255]
    vis = cv2.addWeighted(frame, 0.6, overlay, 0.4, 0)
    return vis


def find_working_camera_index(max_try=4):
    for i in range(max_try):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if os.name=='nt' else cv2.VideoCapture(i)
        if cap.isOpened():
            cap.release(); return i
        cap.release()
    return None


def capture_frame_from_camera(cam_index=None, save_dir="captured_images", file_prefix="plant"):
    if cam_index is None:
        cam_index = find_working_camera_index()
    if cam_index is None:
        return None, None
    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW) if os.name=='nt' else cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        return None, None
    for _ in range(3):
        ret, frame = cap.read()
        if not ret: time.sleep(0.05)
    ret, frame = cap.read()
    cap.release()
    if not ret: return None, None
    os.makedirs(save_dir, exist_ok=True)
    ts = int(time.time()); path = os.path.join(save_dir, f"{file_prefix}_{ts}.jpg")
    cv2.imwrite(path, frame)
    return path, frame



def on_capture_and_predict():
    status_lbl.configure(text="Opening camera...")
    img_path, frame = capture_frame_from_camera()
    if img_path is None:
        status_lbl.configure(text="Camera failed")
        threading.Thread(target=speak_text_online, args=("Camera failed", CURRENT_LANG_CODE), daemon=True).start()
        return
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).resize((300,200))
        photo = ImageTk.PhotoImage(img)
        preview_label.configure(image=photo, text="")
        preview_label.image = photo
    except Exception as e:
        print("preview error", e)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_green = np.array([25,40,40]); upper_green = np.array([95,255,255])
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_ratio = int(np.count_nonzero(mask)) / (frame.shape[0]*frame.shape[1]+1e-9)
    if green_ratio < 0.06:
        status_lbl.configure(text="No plant detected. Capture close-up leaf.")
        threading.Thread(target=speak_text_online, args=("No plant detected. Capture close-up leaf.", CURRENT_LANG_CODE), daemon=True).start()
        return
    status_lbl.configure(text="Plant detected — predicting...")
    try:
        result = None
        if disease_module is not None and hasattr(disease_module, 'capture_and_predict'):
            try:
                result = disease_module.capture_and_predict(image_path=img_path)
            except TypeError:
                result = disease_module.capture_and_predict()
        elif disease_module is not None and hasattr(disease_module, 'predict_from_image'):
            result = disease_module.predict_from_image(img_path)
        else:
            result = "Model not available"
        status_lbl.configure(text=f"Detected: {result}")
        threading.Thread(target=speak_text_online, args=(f"Detected: {result}", CURRENT_LANG_CODE), daemon=True).start()
        vis = compute_simple_explain(frame)
        vis_img = Image.fromarray(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)).resize((300,200))
        vis_photo = ImageTk.PhotoImage(vis_img)
        if hasattr(on_capture_and_predict, '_vis_label') and on_capture_and_predict._vis_label is not None:
            on_capture_and_predict._vis_label.configure(image=vis_photo)
            on_capture_and_predict._vis_label.image = vis_photo
        else:
            lbl = ctk.CTkLabel(right_frame, image=vis_photo, text="")
            lbl.pack(pady=6)
            on_capture_and_predict._vis_label = lbl
    except Exception as e:
        print('Prediction error', e)
        status_lbl.configure(text='Prediction error')

cap_btn = ctk.CTkButton(right_frame, text="Capture & Detect Disease", command=on_capture_and_predict)
cap_btn.pack(pady=8)

# Startup greeting in selected language

def startup_greeting():
    msgs = {
        "en": "Welcome to AI Agro Assistant! Ready to help you grow smarter.",
        "mr": "तुमच्या एआय अ‍ॅग्रो असिस्टंटमध्ये स्वागत आहे! मी तुम्हाला मदत करण्यासाठी तयार आहे.",
        "hi": "एआई एग्रो असिस्टेंट में आपका स्वागत है! मैं आपकी सहायता के लिए तैयार हूँ।",
        "ta": "ஏஐ அயிர் உதவிக்கருவிக்கு வரவேற்பு! நான் உங்களுக்கு உதவ தயாராக இருக்கிறேன்.",
        "kn": "ಎಐ ಅಗ್ರೋ ಸಹಾಯಕರಿಗೆ ಸ್ವಾಗತ! ನಾನು ನಿಮಗೆ ಸಹಾಯ ಮಾಡಲು ಸಿದ್ದನಿದ್ದೇನೆ."
    }
    txt = msgs.get(CURRENT_LANG_CODE, msgs["en"])
    threading.Thread(target=speak_text_online, args=(txt, CURRENT_LANG_CODE), daemon=True).start()

threading.Thread(target=startup_greeting, daemon=True).start()

# Footer
footer = ctk.CTkLabel(app, text="AI features: Fertilizer Recommender, Yield Predictor, Soil Classifier. Models trained on-device (synthetic data) if sklearn is installed.")
footer.pack(pady=6)

app.mainloop()
