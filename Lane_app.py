import cv2
import numpy as np

def main():
    video_path = "C:/Users/lgspa/Downloads/lane/Ytube-rld-main/road_car_view.mp4"
    video = cv2.VideoCapture(video_path)

    while True:
        ret, frame = video.read()

        if not ret:
            break

        blurred_frame = cv2.GaussianBlur(frame, (5, 5), 0)
        hsv = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2HSV)

        lower_y = np.array([18, 94, 140])
        upper_y = np.array([18, 255, 255])

        mask = cv2.inRange(hsv, lower_y, upper_y)
        edges = cv2.Canny(mask, 74, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, maxLineGap=50)

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Display the processed frame (with lane detection)
        cv2.imshow("Lane Detection", frame)

        key = cv2.waitKey(25)
        if key == 27:
            break

    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
