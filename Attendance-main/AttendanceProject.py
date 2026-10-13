import os
import smtplib
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path

import cv2
import face_recognition
import numpy as np

BASE_DIR = Path(__file__).resolve().parent
IMAGES_DIR = BASE_DIR / "Images_Attendance"
ATTENDANCE_FILE = BASE_DIR / "Attendance.csv"
ABSENT_FILE = BASE_DIR / "Absent_Students.csv"

SENDER_EMAIL = os.getenv("ATTENDANCE_SENDER_EMAIL")
SENDER_PASSWORD = os.getenv("ATTENDANCE_SENDER_PASSWORD")
RECIPIENT_EMAIL = os.getenv("ATTENDANCE_RECIPIENT_EMAIL", "sadongaredata@gmail.com")


def load_known_faces():
    """Load training images and names from the Images_Attendance folder."""
    if not IMAGES_DIR.exists():
        raise FileNotFoundError(
            f"Face images folder not found: {IMAGES_DIR}. "
            "Make sure the folder exists and contains face photos."
        )

    images = []
    class_names = []

    for image_path in sorted(IMAGES_DIR.iterdir()):
        if image_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".bmp"}:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Warning: Could not read image {image_path.name}. Skipping it.")
            continue

        images.append(image)
        class_names.append(image_path.stem.strip().upper())

    if not images:
        raise RuntimeError(
            f"No valid face images were found in {IMAGES_DIR}. "
            "Add PNG/JPG/JPEG images to that folder before running the app."
        )

    return images, class_names


def find_encodings(images):
    """Generate face encodings for the supplied list of images."""
    encodings = []
    for index, image in enumerate(images):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_image)

        if not face_locations:
            raise ValueError(
                f"No face was detected in image #{index + 1}. Check the training images."
            )

        encoding = face_recognition.face_encodings(rgb_image, face_locations)[0]
        encodings.append(encoding)

    return encodings


def mark_attendance(name):
    """Record a student as present in the attendance CSV."""
    normalized_name = name.strip().upper()
    ATTENDANCE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if not ATTENDANCE_FILE.exists():
        ATTENDANCE_FILE.touch()

    with ATTENDANCE_FILE.open("a+", encoding="utf-8") as attendance_file:
        attendance_file.seek(0)
        existing_names = {
            line.split(",", 1)[0].strip().upper()
            for line in attendance_file.readlines()
            if line.strip()
        }

        if normalized_name not in existing_names:
            now = datetime.now()
            time_now = now.strftime("%H:%M:%S")
            date_now = now.strftime("%d/%m/%Y")
            attendance_file.write(f"{normalized_name},{time_now},{date_now}\n")


def get_absent_students(class_names):
    """Return all students in the image folder that are absent from the attendance file."""
    if not ATTENDANCE_FILE.exists():
        return list(class_names)

    with ATTENDANCE_FILE.open("r", encoding="utf-8") as attendance_file:
        marked_names = {
            line.split(",", 1)[0].strip().upper()
            for line in attendance_file.readlines()
            if line.strip()
        }

    absent_students = [name for name in class_names if name not in marked_names]

    ABSENT_FILE.write_text("Absent Students\n" + "\n".join(absent_students) + ("\n" if absent_students else ""), encoding="utf-8")
    return absent_students


def send_email(absent_students, recipient_email=None):
    """Send the list of absent students by email if the app is configured to do so."""
    if not absent_students:
        print("No absent students to email.")
        return

    if not SENDER_EMAIL or not SENDER_PASSWORD:
        print("Email credentials not configured. Set ATTENDANCE_SENDER_EMAIL and ATTENDANCE_SENDER_PASSWORD to send emails.")
        return

    target_email = recipient_email or RECIPIENT_EMAIL
    email_body = "Absent Students:\n" + "\n".join(absent_students)

    msg = MIMEText(email_body)
    msg["Subject"] = "Absent Students List"
    msg["From"] = SENDER_EMAIL
    msg["To"] = target_email

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.send_message(msg)
        print(f"Email sent to {target_email}.")
    except Exception as exc:
        print(f"Could not send email: {exc}")


def prompt_for_time(prompt_text, default_value):
    user_choice = input(f"{prompt_text} [default: {default_value}]: ").strip()
    if not user_choice:
        return default_value
    return user_choice


def main():
    print("Loading known faces...")
    images, class_names = load_known_faces()
    encode_list_known = find_encodings(images)
    print("Encoding complete.")

    start_time_input = prompt_for_time("Enter start time (HH:MM:SS)", "09:00:00")
    end_time_input = prompt_for_time("Enter end time (HH:MM:SS)", "17:00:00")

    try:
        start_time = datetime.strptime(start_time_input, "%H:%M:%S").time()
        end_time = datetime.strptime(end_time_input, "%H:%M:%S").time()
    except ValueError as exc:
        raise ValueError("Time must be in HH:MM:SS format.") from exc

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check that your camera is connected and available.")

    print("Attendance started. Press 'q' or Enter to stop.")

    while True:
        current_time = datetime.now().time()

        if current_time < start_time:
            print("Attendance has not started yet.")
            cv2.waitKey(1000)
            continue

        if current_time > end_time:
            print("Attendance window is closed.")
            break

        success, img = cap.read()
        if not success:
            print("Failed to access the webcam frame.")
            break

        img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img_small_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(img_small_rgb)
        face_encodings = face_recognition.face_encodings(img_small_rgb, face_locations)

        for face_encoding, face_location in zip(face_encodings, face_locations):
            matches = face_recognition.compare_faces(encode_list_known, face_encoding)
            face_distances = face_recognition.face_distance(encode_list_known, face_encoding)

            if len(face_distances) == 0:
                continue

            match_index = int(np.argmin(face_distances))
            if matches[match_index]:
                name = class_names[match_index]
                y1, x2, y2, x1 = [value * 4 for value in face_location]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                mark_attendance(name)

        cv2.imshow("Webcam", img)

        key = cv2.waitKey(10) & 0xFF
        if key in (13, ord("q")):
            print("Stopping the system...")
            break

    cap.release()
    cv2.destroyAllWindows()

    absent_students = get_absent_students(class_names)
    print(f"Absent students: {absent_students}")
    send_email(absent_students)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Error: {exc}")
        raise
