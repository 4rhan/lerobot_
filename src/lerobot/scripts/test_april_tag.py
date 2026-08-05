import cv2
import numpy as np

def test_webcam_apriltag():
    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Load the dictionary
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    
    # --- NEW OPENCV 4.7+ API ---
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    print("Looking for AprilTag... Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect the tag using the new detector object
        corners, ids, rejected = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            # Draw a green box around the tag if found!
            cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0, 255, 0))
            cv2.putText(frame, f"TAG FOUND! ID: {ids[0][0]}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "No Tag Seen", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Live Tag Debugger", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_webcam_apriltag()