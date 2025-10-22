import numpy as np
import cv2
import mediapipe as mp

MULTIPLIER_FACTOR = 3
video_feed = cv2.VideoCapture(0)

mediapipe_hands = mp.solutions.hands
drawing_utilities = mp.solutions.drawing_utils
hand_detector = mediapipe_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

while video_feed.isOpened():
    success, current_frame = video_feed.read()
    if not success:
        break

    rgb_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
    detection_results = hand_detector.process(rgb_frame)

    if detection_results.multi_hand_landmarks:
        frame_height, frame_width, _ = current_frame.shape
        
        for detected_hand in detection_results.multi_hand_landmarks:
            drawing_utilities.draw_landmarks(current_frame, detected_hand, mediapipe_hands.HAND_CONNECTIONS)
            
            thumb_landmark = detected_hand.landmark[4]
            index_landmark = detected_hand.landmark[8]
            
            thumb_x, thumb_y = int(thumb_landmark.x * frame_width), int(thumb_landmark.y * frame_height)
            index_x, index_y = int(index_landmark.x * frame_width), int(index_landmark.y * frame_height)

            difference_x, difference_y = thumb_x - index_x, thumb_y - index_y
            pixel_distance = int(np.hypot(difference_x, difference_y))

            center_x, center_y = int((thumb_x + index_x) / 2), int((thumb_y + index_y) / 2)

            cv2.circle(current_frame, (thumb_x, thumb_y), 6, (0, 0, 255), -1)
            cv2.circle(current_frame, (index_x, index_y), 6, (0, 255, 0), -1)
            cv2.line(current_frame, (thumb_x, thumb_y), (index_x, index_y), (255, 0, 0), 2)

            rectangle_size = max(10, int(pixel_distance * MULTIPLIER_FACTOR))
            radius = rectangle_size // 2
            
            top_left_x, top_left_y = max(0, center_x - radius), max(0, center_y - radius)
            bottom_right_x, bottom_right_y = min(frame_width - 1, center_x + radius), min(frame_height - 1, center_y + radius)
            
            cv2.rectangle(current_frame, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 165, 255), 2)
            cv2.putText(current_frame, f"{rectangle_size}px", (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2, cv2.LINE_AA)

    cv2.imshow("Reconocimiento de Letras", current_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_feed.release()
cv2.destroyAllWindows()