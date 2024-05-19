from flask import Flask, request, jsonify
import face_recognition
import cv2
import numpy as np
import base64

app = Flask(__name__)

# Load known images and their encodings
known_encodings = []
known_names = []

# Load the known image and encoding
known_image = face_recognition.load_image_file('known_face.jpg')
known_face_encoding = face_recognition.face_encodings(known_image)[0]
known_encodings.append(known_face_encoding)
known_names.append("Known Face")

# Define the API endpoint for facial recognition
@app.route('/recognize-face', methods=['POST'])
def recognize_face():
    # Receive image from frontend
    image_data = request.files['image'].read()
    nparr = np.frombuffer(image_data, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert the image from BGR color to RGB color
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Find all the faces and face encodings in the current frame
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    # Perform facial recognition
    face_names = []
    for face_encoding in face_encodings:
        # Compare the face with the known faces
        matches = face_recognition.compare_faces(known_encodings, face_encoding)
        name = "Unknown"
        if True in matches:
            name = known_names[matches.index(True)]
        face_names.append(name)

    # Return the recognition result
    return jsonify({"message": face_names[0]})

if __name__ == '__main__':
    app.run(debug=True)
