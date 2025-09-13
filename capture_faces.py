import cv2
import os

# Always find base dir (where this script lives)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(BASE_DIR, "dataset")

# Ask for student name
name = input("Enter student's name: ")
person_dir = os.path.join(dataset_path, name)
os.makedirs(person_dir, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
count = 0

print("📸 Press 's' to save an image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame")
        break

    cv2.imshow("Capture Faces", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s'):  # Save
        img_name = os.path.join(person_dir, f"{count}.jpg")
        cv2.imwrite(img_name, frame)
        print(f"✅ Saved: {img_name}")
        count += 1
    elif key == ord('q'):  # Quit
        break

cap.release()
cv2.destroyAllWindows()
