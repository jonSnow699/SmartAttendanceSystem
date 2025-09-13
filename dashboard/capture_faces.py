import cv2
import os

# Ask for student name
name = input("Enter student's name: ")
path = f"../dataset/{name}"
os.makedirs(path, exist_ok=True)

# Open webcam
cap = cv2.VideoCapture(0)
count = 0

print("Press 's' to save an image, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    cv2.imshow("Capturing Faces", frame)
    key = cv2.waitKey(1)

    if key == ord('s'):
        # Save image
        img_path = f"{path}/{name}_{count}.jpg"
        cv2.imwrite(img_path, frame)
        print(f"Saved: {img_path}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

