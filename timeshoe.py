from ultralytics import YOLO
import cv2
import math 

# start webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("last (2).pt")

# object classes
classNames = ["Shoe","People"]

# detection threshold
threshold_frames = 150  # 5 seconds at 30 fps

# counter for consecutive frames
consecutive_frames = 0

while True:
    success, img = cap.read()
    results = model(img, stream=True)

    # Check if results list is not empty
    if results:
        # coordinates
        for r in results:
            boxes = r.boxes
            shoe_detected = False  # flag to track if shoe is detected

            for box in boxes:
                # class name
                cls = int(box.cls[0])

                # Check if the class index is within the valid range
                if 0 <= cls < len(classNames):
                    # filter by class name
                    if classNames[cls] == "Shoe":
                        # increment consecutive frames counter
                        consecutive_frames += 1

                        # check if shoe detected for threshold_frames frames
                        if consecutive_frames >= threshold_frames:
                            # bounding box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                            # put box in cam
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

                            # confidence
                            confidence = math.ceil((box.conf[0]*100))/100
                            print("Confidence --->",confidence)

                            # object details
                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2

                            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

                            # reset consecutive frames counter
                            consecutive_frames = 0
                            shoe_detected = True

            # if shoe is not detected in current frame, reset the counter
            if not shoe_detected:
                consecutive_frames = 0

    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
