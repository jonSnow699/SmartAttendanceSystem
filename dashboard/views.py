from django.shortcuts import render, HttpResponse
import subprocess
import os
import csv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dataset_path = os.path.join(BASE_DIR, "dataset")
encodings_file = os.path.join(BASE_DIR, "encodings.pickle")
attendance_file = os.path.join(BASE_DIR, "attendance.csv")

def home(request):
    return render(request, "dashboard/home.html")

def capture_faces(request):
    script_path = os.path.join(BASE_DIR, "capture_faces.py")
    subprocess.Popen(["python", script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
    return HttpResponse("✅ Face capture started in a new window.")

def encode_faces(request):
    script_path = os.path.join(BASE_DIR, "encode_faces_deepface.py")
    subprocess.Popen(["python", script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
    return HttpResponse("✅ Encoding started in background.")

def start_attendance(request):
    script_path = os.path.join(BASE_DIR, "recognize_faces_deepface.py")
    subprocess.Popen(["python", script_path], creationflags=subprocess.CREATE_NEW_CONSOLE)
    return HttpResponse("✅ Attendance recognition started in background.")

def view_attendance(request):
    if not os.path.exists(attendance_file):
        return HttpResponse("⚠️ Attendance file not found.")

    records = []
    headers = []

    # Read CSV file
    with open(attendance_file, "r", newline="") as f:
        reader = csv.reader(f)
        headers = next(reader, [])
        for row in reader:
            records.append(row)

    # Render attendance in table
    return render(request, "dashboard/attendance.html", {
        "headers": headers,
        "records": records,
    })
