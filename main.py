import cv2
import mediapipe as mp
import pyautogui
import os
import shutil


def calculate_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

def main():
    # Initialize MediaPipe Hand detection
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)
    mp_draw = mp.solutions.drawing_utils

    # Initialize OpenCV video capture
    cap = cv2.VideoCapture(0)

    screen_width, screen_height = pyautogui.size()
    prev_click = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Flip and process the frame
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Extract landmark positions
                landmarks = hand_landmarks.landmark
                index_finger_tip = [landmarks[8].x, landmarks[8].y]
                thumb_tip = [landmarks[4].x, landmarks[4].y]

                # Convert normalized coordinates to screen coordinates
                index_screen_x = int(index_finger_tip[0] * screen_width)
                index_screen_y = int(index_finger_tip[1] * screen_height)

                # Move mouse pointer to the index finger tip position
                pyautogui.moveTo(index_screen_x, index_screen_y)

                # Detect pinch gesture (thumb and index finger close)
                distance = calculate_distance(index_finger_tip, thumb_tip)
                if distance < 0.05:  # Adjust threshold as needed
                    if not prev_click:
                        pyautogui.click()
                        prev_click = True
                else:
                    prev_click = False

                # Detect "OK" gesture for copy operation
                middle_finger_tip = [landmarks[12].x, landmarks[12].y]
                ring_finger_tip = [landmarks[16].x, landmarks[16].y]

                ok_distance = calculate_distance(index_finger_tip, thumb_tip)
                ring_middle_distance = calculate_distance(middle_finger_tip, ring_finger_tip)

                if ok_distance < 0.05 and ring_middle_distance < 0.05:
                    print("Copy operation detected!")
                    # Define file paths for the copy operation
                    src_file = "source_file_path_here"
                    dest_file = "destination_path_here"
                    try:
                        shutil.copy(src_file, dest_file)
                        print("File copied successfully!")
                    except Exception as e:
                        print(f"Error copying file: {e}")

        # Display the frame
        cv2.imshow("Hand Gesture Mouse", frame)

        # Exit loop on pressing 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
