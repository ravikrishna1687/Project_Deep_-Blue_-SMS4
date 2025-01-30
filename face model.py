import cv2
import numpy as np

# Load pre-trained DNN model for age estimation
AGE_MODEL = "age_deploy.prototxt"
AGE_PROTO = "age_net.caffemodel"
AGE_CLASSES = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(21-24)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
age_net = cv2.dnn.readNet(AGE_PROTO, AGE_MODEL)

# Real-time height estimation
def estimate_height(frame, reference_height, reference_pixel_height):
    """
    Estimate height of a person or object in real-time.
    :param frame: Input video frame
    :param reference_height: Real-world height of the reference object (e.g., in cm)
    :param reference_pixel_height: Height of the reference object in pixels
    :return: Estimated height in real-world units (e.g., cm)
    """
    person_pixel_height = detect_person_height(frame)
    if person_pixel_height == 0:
        return None
    scale = reference_height / reference_pixel_height
    return person_pixel_height * scale

def detect_person_height(frame):
    """
    Detect the height of a person in the video frame (bounding box height in pixels).
    Requires a pre-trained person detector (e.g., YOLO or Haar cascades).
    """
    # Placeholder for person detection logic
    # Implement person detection using OpenCV or YOLO
    return 200  # Example: bounding box height in pixels

# Age Estimation
def estimate_age(face_image):
    """
    Estimate age from a face image using a pre-trained DNN model.
    :param face_image: Cropped face image
    :return: Estimated age group
    """
    blob = cv2.dnn.blobFromImage(face_image, 1.0, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_net.setInput(blob)
    age_preds = age_net.forward()
    age = AGE_CLASSES[age_preds[0].argmax()]
    return age

# Weight Estimation (Placeholder)
def estimate_weight(body_measurements):
    """
    Estimate weight using body measurements.
    :param body_measurements: Array of body measurements (e.g., height, chest, waist)
    :return: Estimated weight in kg
    """
    # Placeholder for ML model prediction
    return 70  # Example: estimated weight in kg

# Main function for real-time processing
def main():
    # Access webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the camera.")
        return

    reference_height = 30  # cm (height of a reference object)
    reference_pixel_height = 100  # pixels (height of the reference object in the frame)

    print("Press 'q' to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from camera.")
            break

        # Height estimation
        estimated_height = estimate_height(frame, reference_height, reference_pixel_height)
        height_text = f"Height: {estimated_height:.2f} cm" if estimated_height else "Height: N/A"

        # Age estimation (using face detection)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            age_group = estimate_age(cv2.resize(face, (227, 227)))
            age_text = f"Age: {age_group}"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            cv2.putText(frame, age_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # Overlay text
        cv2.putText(frame, height_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display video frame
        cv2.imshow("Real-Time Height and Age Estimation", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
