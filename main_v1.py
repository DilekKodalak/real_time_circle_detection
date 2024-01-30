import cv2
import numpy as np
import time

CAMERA_ID = 0
IMAGE_WIDTH = 300
IMAGE_HEIGHT = 300
fps = 50

def set_camera_properties(cap, width, height):
    cap.set(3, width)
    cap.set(4, height)

def capture_frame(cap):
    ret, frame = cap.read()
    if not ret:
        raise ValueError("Failed to read a frame from the camera")
    return frame

def detect_circles(gray_frame):
    circles = cv2.HoughCircles(
        gray_frame,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=50,
        param1=60,
        param2=55,
        minRadius=3,
        maxRadius=50
    )
    return circles

def processing_frame(frame):
    frame = cv2.blur(frame, (3, 3))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kernel = np.ones((5, 5), np.uint8)
    gray = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, kernel)

    return frame, gray

def draw_circles(frame, circles):
    output = frame.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 200, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return output

def visual_fps(image, fps: float) -> cv2.UMat:
    if len(np.shape(image)) < 3:
        text_color = (255, 255, 255)
    else:
        text_color = (0, 255, 0)

    row_size = 25
    font_size = 1
    font_thickness = 1
    left_margin = 20
    text_fps = 'FPS: {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    image_umat = cv2.UMat(image)
    cv2.putText(image_umat, text_fps, text_location, cv2.FONT_HERSHEY_PLAIN, font_size, text_color, font_thickness)
    return image_umat

def main():
    try:
        cap = cv2.VideoCapture(CAMERA_ID)
        if not cap.isOpened():
            raise ValueError("Could not open the camera")
        set_camera_properties(cap, IMAGE_WIDTH, IMAGE_HEIGHT)
        print("Press 'ESC' to exit ")

        while True:
            start_time = time.time()
            frame = capture_frame(cap)
            frame, gray = processing_frame(frame)
            circles = detect_circles(gray)

            if circles is not None:
                output = draw_circles(frame, circles)
            else:
                output = frame.copy()

            end_time = time.time()
            seconds = end_time - start_time
            fps = 1.0 / seconds

            output_umat = visual_fps(output, fps)
            out_np = cv2.UMat.get(output_umat)

            cv2.imshow("frame", np.hstack((out_np,)))

            if cv2.waitKey(33) == 27:
                break

    except Exception as exp:
        print(exp)

    finally:
        cv2.destroyAllWindows()
        cap.release()

if __name__ == '__main__':
    main()