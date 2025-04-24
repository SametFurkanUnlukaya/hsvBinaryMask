
import cv2
import numpy as np


cap = cv2.VideoCapture(0)
clicked_color = None
lower_bound = None
upper_bound = None


def click_event(event, x, y, flags, param):
    global clicked_color, lower_bound, upper_bound
    if event == cv2.EVENT_LBUTTONDOWN and param is not None:
        frame = param
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        clicked_color = hsv[y, x]
        print("Tıklanan HSV renk:", clicked_color)

      
        lower_bound = np.clip(clicked_color - np.array([10, 50, 50]), [0, 0, 0], [179, 255, 255])
        upper_bound = np.clip(clicked_color + np.array([10, 50, 50]), [0, 0, 0], [179, 255, 255])

cv2.namedWindow("Camera")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Her karede en güncel frame'i param olarak gönder
    cv2.setMouseCallback("Camera", click_event, param=frame)

    if clicked_color is not None:
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500:  # Küçük gürültüleri yok say
                cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        cv2.imshow("Mask", mask)

    cv2.imshow("Camera", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
