import customtkinter as ctk
from tkinter import messagebox
from gtts import gTTS
import os
from playsound import playsound
import tempfile

# Function to play voice output
from gtts import gTTS
import pygame
import tempfile
import time
import os

def speak(text, lang_code):
    try:
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
            tts = gTTS(text=text, lang=lang_code)
            tts.save(fp.name)

        # Initialize pygame mixer and play
        pygame.mixer.init()
        pygame.mixer.music.load(fp.name)
        pygame.mixer.music.play()

        # Wait until sound finishes playing
        while pygame.mixer.music.get_busy():
            time.sleep(0.1)

        # Cleanup
        pygame.mixer.quit()
        os.remove(fp.name)
    except Exception as e:
        print(f"Speech error: {e}")


# AI analysis + voice output
def analyze_ph():
    ph_value = ph_entry.get()
    crop = crop_menu.get()
    lang = lang_menu.get()

    if not ph_value:
        messagebox.showwarning("Input Error", "Please enter a pH value.")
        return

    try:
        ph_value = float(ph_value)
    except ValueError:
        messagebox.showwarning("Input Error", "Please enter a valid number for pH.")
        return

    # Language codes mapping for gTTS
    lang_map = {
        "English": "en",
        "Marathi": "mr",
        "Kannada": "kn",
        "Tamil": "ta"
    }

    lang_code = lang_map.get(lang, "en")

    # Crop and fertilizer suggestion (sample logic)
    if 6.0 <= ph_value <= 7.5:
        result = f"{crop} is suitable for this pH value. Recommended fertilizer: Urea."
    elif ph_value < 6.0:
        result = f"Soil is acidic for {crop}. Use Lime or Calcium Carbonate."
    else:
        result = f"Soil is alkaline for {crop}. Use Ammonium Sulphate or Sulphur."

    # Show on screen
    messagebox.showinfo("AI Analysis Result", result)

    # Speak based on selected language
    speak_msg = f"You selected {lang}. {result}" if lang == "English" else f"{result}"
    speak(speak_msg, lang_code)

# ---------- UI Design -----------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("green")

app = ctk.CTk()
app.title("Smart ph value AI Assistant 🌾")
app.geometry("650x450")

# Heading
title_label = ctk.CTkLabel(app, text="Smart Krushi: AI-Powered Soil pH Analysis",
                           font=("Poppins", 20, "bold"))
title_label.pack(pady=20)

# pH Entry
ph_label = ctk.CTkLabel(app, text="Enter Soil pH Value:", font=("Poppins", 14))
ph_label.pack()
ph_entry = ctk.CTkEntry(app, placeholder_text="e.g., 6.5", width=200)
ph_entry.pack(pady=10)

# Crop Dropdown
crop_label = ctk.CTkLabel(app, text="Select Crop (Peke):", font=("Poppins", 14))
crop_label.pack()
crop_menu = ctk.CTkOptionMenu(app, values=[
    "Wheat (गहू)", "Rice (तांदूळ)", "Cotton (कापूस)",
    "Sugarcane (ऊस)", "Tomato (टोमॅटो)", "Maize (मका)", "Soybean (सोयाबीन)"
])
crop_menu.pack(pady=10)

# Language Dropdown
lang_label = ctk.CTkLabel(app, text="Select Language:", font=("Poppins", 14))
lang_label.pack()
lang_menu = ctk.CTkOptionMenu(app, values=["English", "Marathi", "Kannada", "Tamil"])
lang_menu.pack(pady=10)

# Analyze Button
analyze_button = ctk.CTkButton(app, text="Analyze Soil", command=analyze_ph, width=200)
analyze_button.pack(pady=25)

app.mainloop()
