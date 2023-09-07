import cv2
import mediapipe as mp
import random
import time

# Initialize Mediapipe face detection module
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection()

# Load the apple image
apple_img = cv2.imread('apple.png', cv2.IMREAD_UNCHANGED)

# Initialize webcam capture
cap = cv2.VideoCapture(0)

# Set up random circle position
circle_radius = 40
score = 0

# Set up countdown timer
start_time = time.time()
duration = 20  # seconds

reset_game = False

# Generate initial random circle position
circle_position = (random.randint(circle_radius, 640 - circle_radius), random.randint(circle_radius, 480 - circle_radius))

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        continue
    
    if reset_game:
        # Reset game parameters
        start_time = time.time()
        score = 0
        reset_game = False
    
    # Flip the frame horizontally
    frame = cv2.flip(frame, 1)  # 1 indicates horizontal flip
 
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame to detect faces
    results = face_detection.process(rgb_frame)
    
    # Calculate time remaining
    elapsed_time = time.time() - start_time
    time_remaining = max(0, duration - int(elapsed_time))
    
    if time_remaining == 0:
        # Display "Game Over" text and final score
        cv2.putText(frame, "Game Over", (200, 250), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
        cv2.putText(frame, f"Final Score: {score}", (150, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    else:
        if results.detections:
            for detection in results.detections:
                landmarks = detection.location_data.relative_keypoints
                mouth_landmarks = landmarks[mp_face_detection.FaceKeyPoint.MOUTH_CENTER]
                ih, iw, _ = frame.shape
                mouth_x, mouth_y = int(mouth_landmarks.x * iw), int(mouth_landmarks.y * ih)
                
                # Check if mouth touches the circle
                circle_center_x, circle_center_y = circle_position
                distance = ((mouth_x - circle_center_x) ** 2 + (mouth_y - circle_center_y) ** 2) ** 0.5
                if distance < circle_radius:
                    # Mouth touches the circle, update the score
                    score += 1
                    # Generate a new random circle position
                    circle_position = (random.randint(circle_radius, 640 - circle_radius), random.randint(circle_radius, 480 - circle_radius))
                
                # Draw a circle around the mouth
                cv2.circle(frame, (mouth_x, mouth_y), circle_radius, (255, 0, 0), 2)
    
    # Display the score
    cv2.putText(frame, f"Score: {score}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display the time remaining
    cv2.putText(frame, f"Time: {time_remaining} s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # Display reset instructions
    cv2.putText(frame, "Press 'r' to reset", (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Draw the random circle
    cv2.circle(frame, circle_position, 20, (0, 255, 0), -1)

    # Resize the apple image to match the circle's dimensions
    apple_resized = cv2.resize(apple_img, (circle_radius * 2, circle_radius * 2))
    
    # Calculate the position for placing the apple image on top of the circle
    apple_x = circle_position[0] - circle_radius
    apple_y = circle_position[1] - circle_radius
    
    # Overlay the apple image with transparency on the frame
    for y in range(apple_resized.shape[0]):
        for x in range(apple_resized.shape[1]):
            if apple_resized[y, x, 3] != 0:  # Check the alpha channel for transparency
                frame[apple_y + y, apple_x + x] = apple_resized[y, x, 0:3]

    
    cv2.imshow('Face and Mouth Detection', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        reset_game = True

cap.release()
cv2.destroyAllWindows()
