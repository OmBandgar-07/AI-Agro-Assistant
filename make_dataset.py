# make_dataset.py
import csv
import random

# Define crops with approximate pH ranges (for synthetic data)
crop_ranges = {
    "Rice": (5.0, 6.5),
    "Wheat": (6.0, 7.5),
    "Sugarcane": (5.5, 7.5),
    "Cotton": (6.0, 8.5),
    "Maize": (5.5, 7.5),
    "Tomato": (5.8, 6.8),
    "Soybean": (6.0, 7.5),
    "Potato": (5.0, 6.5),
    "Groundnut": (5.0, 6.5),
    "Banana": (5.5, 7.0)
    
}

soil_types = ["red", "black", "alluvial", "laterite", "sandy"]

def sample_fertilizer_for_crop(crop):
    mapping = {
        "Rice": "Urea",
        "Wheat": "DAP",
        "Sugarcane": "FYM",
        "Cotton": "NPK",
        "Maize": "Urea",
        "Tomato": "NPK (19:19:19)",
        "Soybean": "Rhizobium + SSP",
        "Potato": "MOP + DAP",
        "Groundnut": "Gypsum + SSP",
        "Banana": "K-rich fertilizer"
    }
    return mapping.get(crop, "Balanced Fertilizer")

# Generate rows
rows = []
for _ in range(1200):  # 1200 samples
    crop = random.choice(list(crop_ranges.keys()))
    ph_min, ph_max = crop_ranges[crop]
    # a bit of noise: sometimes sample slightly outside range
    ph = round(random.uniform(ph_min - 0.6, ph_max + 0.6), 2)
    soil = random.choice(soil_types)
    temp = round(random.uniform(15, 35), 1)   # optional extra feature
    moisture = round(random.uniform(10, 60), 1)
    fertilizer = sample_fertilizer_for_crop(crop)
    rows.append([ph, soil, temp, moisture, crop, fertilizer])

# Save CSV
with open("soil_dataset.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["pH", "soil_type", "temperature", "moisture", "crop", "fertilizer"])
    writer.writerows(rows)

print("soil_dataset.csv created with", len(rows), "rows.")
