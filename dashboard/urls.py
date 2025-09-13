from django.urls import path
from . import views

urlpatterns = [
    path("", views.home, name="home"),  # ✅ root path points to dashboard
    path("dashboard/", views.home, name="dashboard"),
    path("capture-faces/", views.capture_faces, name="capture_faces"),
    path("encode-faces/", views.encode_faces, name="encode_faces"),
    path("start-attendance/", views.start_attendance, name="start_attendance"),
    path("view-attendance/", views.view_attendance, name="view_attendance"),
]
