import face_recognition
import cv2
import numpy as np
import os
import time

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)

# Define the path to the folder containing the images
known_faces_folder = "static/profile_pics"

# Create lists for known face encodings and names
known_face_encodings = []
known_face_names = []

# Load known face encodings and names from the folder
for file in os.listdir(known_faces_folder):
    if file.endswith(".jpg") or file.endswith(".png"):
        # Load the image file
        image_path = os.path.join(known_faces_folder, file)
        known_face_image = face_recognition.load_image_file(image_path)

        # Compute the face encoding and add it to the list
        face_encoding = face_recognition.face_encodings(known_face_image)
        if len(face_encoding) > 0:
            known_face_encodings.append(face_encoding[0])
            # Use the file name (without the extension) as the known face name
            known_face_names.append(os.path.splitext(file)[0])

while True:
    # Capture an image from the live video feed after 5 seconds
    time.sleep(0)  # Wait for 5 seconds
    ret, frame = video_capture.read()
    captured_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find face locations and encodings in the captured image
    face_locations = face_recognition.face_locations(captured_image)
    face_encodings = face_recognition.face_encodings(captured_image, face_locations)

    # Match the captured face with known faces
    face_names = []
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]
        face_names.append(name)

    # Display the results on the captured image
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # Display the captured image with face recognition results
    cv2.imshow('Captured Image', frame)

    # Press 'q' on the keyboard to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close the display window
video_capture.release()
cv2.destroyAllWindows()
