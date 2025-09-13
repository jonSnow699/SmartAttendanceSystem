import cv2
import pickle
import numpy as np
from deepface import DeepFace
from datetime import datetime
import csv
import os

# -------------------------------
# Load known encodings
# -------------------------------
with open("encodings.pickle", "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_names = data["names"]

print("📷 Starting camera...")
cap = cv2.VideoCapture(0)

# -------------------------------
# Attendance file setup
# -------------------------------
attendance_file = "attendance.csv"
if not os.path.isfile(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

# To avoid duplicate marking in one session
marked_names = set()

# -------------------------------
# Recognition loop
# -------------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from camera")
        break

    try:
        # Detect faces using DeepFace
        results = DeepFace.extract_faces(
            img_path=frame,
            detector_backend="opencv",
            enforce_detection=False
        )
    except Exception as e:
        print("⚠️ DeepFace detection error:", e)
        continue

    for res in results:
        # -------------------------------
        # Fix: safely unpack bounding box
        # -------------------------------
        fa = res.get("facial_area", {})
        if not all(k in fa for k in ["x", "y", "w", "h"]):
            continue

        x, y, w, h = fa["x"], fa["y"], fa["w"], fa["h"]

        # Crop face and get embedding
        face_img = frame[y:y+h, x:x+w]
        try:
            emb = DeepFace.represent(
                img_path=face_img,
                model_name="Facenet",
                enforce_detection=False
            )[0]["embedding"]
        except Exception:
            continue

        # -------------------------------
        # Compare with known encodings
        # -------------------------------
        distances = np.linalg.norm(np.array(known_encodings) - emb, axis=1)
        min_index = np.argmin(distances)
        min_distance = distances[min_index]

        threshold = 10  # tune if needed
        name = "Unknown"
        if min_distance < threshold:
            name = known_names[min_index]

        # -------------------------------
        # Draw results on frame
        # -------------------------------
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, name, (x, y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # -------------------------------
        # Mark attendance
        # -------------------------------
        if name != "Unknown" and name not in marked_names:
            now = datetime.now()
            with open(attendance_file, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([name,
                                 now.strftime("%Y-%m-%d"),
                                 now.strftime("%H:%M:%S")])
            marked_names.add(name)
            print(f"✅ Attendance marked for {name}")

    # Show camera window
    cv2.imshow("Attendance - Press Q to quit", frame)

    # Quit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------------------------------
# Cleanup
# -------------------------------
cap.release()
cv2.destroyAllWindows()
print("👋 Camera closed.")
