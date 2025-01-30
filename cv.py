import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe pose model
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2)

def calculate_distance(point1, point2):
    """Calculate Euclidean distance between two points."""
    return np.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)

def estimate_height(frame, focal_length=800):
    """
    Estimate height of a person in a frame.
    Assumes a fixed focal length for the webcam.
    """
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if not results.pose_landmarks:
        return None, None, None
    
    landmarks = results.pose_landmarks.landmark
    nose = [landmarks[mp_pose.PoseLandmark.NOSE].x, landmarks[mp_pose.PoseLandmark.NOSE].y]
    ankle_left = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].x, landmarks[mp_pose.PoseLandmark.LEFT_ANKLE].y]
    ankle_right = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].x, landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE].y]
    
    # Calculate height in pixels
    ankle_avg = [(ankle_left[0] + ankle_right[0]) / 2, (ankle_left[1] + ankle_right[1]) / 2]
    height_pixels = calculate_distance(nose, ankle_avg)
    
    # Draw bounding box
    min_x = int(min([nose[0], ankle_left[0], ankle_right[0]]) * frame.shape[1])
    max_x = int(max([nose[0], ankle_left[0], ankle_right[0]]) * frame.shape[1])
    min_y = int(min([nose[1], ankle_left[1], ankle_right[1]]) * frame.shape[0])
    max_y = int(max([nose[1], ankle_left[1], ankle_right[1]]) * frame.shape[0])
    
    # Estimate real-world height (in cm) using a reference distance
    scaling_factor = 170 / height_pixels  # Assuming average human height as 170 cm
    real_height = height_pixels * scaling_factor
    
    return real_height, (min_x, min_y, max_x, max_y), results.pose_landmarks

def estimate_weight(height_cm):
    """
    Estimate weight based on height using BMI approximation.
    Assumes a normal BMI range for approximation.
    """
    avg_bmi = 22  # Average BMI
    height_m = height_cm / 100
    weight_kg = avg_bmi * (height_m ** 2)
    return weight_kg

# Real-time webcam processing
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)  # Access the default webcam
    if not cap.isOpened():
        print("Error: Could not access the webcam.")
        exit()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame. Exiting...")
            break
        
        frame = cv2.flip(frame, 1)  # Flip the frame horizontally for natural interaction

        # Estimate human height and weight
        height, bounding_box, landmarks = estimate_height(frame)
        if height:
            weight = estimate_weight(height)
            
            # Draw bounding box
            min_x, min_y, max_x, max_y = bounding_box
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)
            
            # Overlay text with height and weight
            cv2.putText(frame, f"Height: {height:.2f} cm", (min_x, min_y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Weight: {weight:.2f} kg", (min_x, min_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow("Real-Time Measurement", frame)
        
        # Press 'q' to quit the application
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
