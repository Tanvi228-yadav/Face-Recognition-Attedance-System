# Face Recognition Attendance System

A Python-based face recognition attendance application that recognizes known faces from a webcam and records attendance automatically in a CSV file.

## Features
- Real-time face detection using OpenCV
- Face recognition using `face_recognition`
- Attendance logging to `Attendance.csv`
- Generates `Absent_Students.csv`
- Optional email notification for absent students
- Simple webcam-based workflow

## Project Structure

```text
Face-Recognition-Attedance-System/
├── Attendance-main/
│   ├── AttendanceProject.py
│   ├── Attendance.csv
│   ├── Absent_Students.csv
│   ├── Images_Attendance/
│   │   ├── Aditya Patil.jpeg
│   │   └── Samarth Dongare.png
│   └── PyWhatKit_DB.txt
├── LICENSE
├── README.md
└── .gitignore
```

## Requirements
Install the following dependencies:

```bash
pip install opencv-python face_recognition numpy
```

## Setup
1. Place known face images inside `Attendance-main/Images_Attendance/`.
2. Name each image using the student’s name, for example:
   - `Samarth Dongare.png`
   - `Aditya Patil.jpeg`
3. Ensure your webcam is connected and available.

## Run the Project
From the repository root:

```bash
python "Attendance-main/AttendanceProject.py"
```

## Optional Email Configuration
To send absent-student emails, set these environment variables before running the script:

```bash
export ATTENDANCE_SENDER_EMAIL="your_email@gmail.com"
export ATTENDANCE_SENDER_PASSWORD="your_app_password"
export ATTENDANCE_RECIPIENT_EMAIL="recipient@gmail.com"
```

Then run the script again.

## Notes
- The application expects the training images to contain clear frontal face photos.
- If no face is found in a training image, the app will show an error and stop.
- The app records attendance only once per student per session.

## License
This project is licensed under the MIT License. See `LICENSE` for details.
