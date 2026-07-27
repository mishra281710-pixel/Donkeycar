import cv2
import time

class ArucoDetector:

    def __init__(self):

        # Dictionary containing 50 markers (IDs 0–49)
        self.dictionary = cv2.aruco.getPredefinedDictionary(
            cv2.aruco.DICT_4X4_50
        )

        # Create detector
        self.detector = cv2.aruco.ArucoDetector(self.dictionary)

    def run(self, img_arr):

        if img_arr is None:
            return None, img_arr

        # Convert to grayscale
        gray = cv2.cvtColor(img_arr, cv2.COLOR_RGB2GRAY)

        # Detect markers
        corners, ids, rejected = self.detector.detectMarkers(gray)

        marker_id = None

        if ids is not None:

            # Draw markers
            cv2.aruco.drawDetectedMarkers(img_arr, corners, ids)

            # Read every detected marker
            for marker in ids.flatten():

                marker_id = int(marker)

                print("Detected Marker:", marker_id)

                # Example actions
                if marker_id == 1:
                    print("TURN LEFT")

                elif marker_id == 2:
                    print("TURN RIGHT")

                elif marker_id == 3:
                    print("STOP")

                elif marker_id == 4:
                    print("U-TURN")

        return marker_id, img_arr


####################################################
# Test using webcam
####################################################

cap = cv2.VideoCapture(0)

aruco = ArucoDetector()

while True:

    ret, frame = cap.read()

    if not ret:
        break

    marker, output = aruco.run(frame)

    cv2.imshow("ArUco Detector", output)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
