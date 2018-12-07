import cv2
import sys
from extraction.extractor import ObjectExtractor

video_capture = cv2.VideoCapture(0)
image_font = cv2.FONT_HERSHEY_SIMPLEX


def print_(message, image, y_off):
    """
    print message to multiple consoles
    :param message: message to print
    :param image: image to print on
    :param y_off: vertical location in image (y offset)
    :return: None
    """
    print(message)  # to STDOUT
    cv2.putText(image, message, (50, y_off), image_font, 0.8, (0, 0, 0))  # to screen


def draw_rectangle(image, coordinates, color=(255, 0, 0), thickness=2):
    """
    draw a rectangle at given coordinates on an image
    :param image: image to draw rectangle on
    :param coordinates: where to draw rectangle
    :param color: color of rectangle
    :param thickness: thickness of rectangle
    :return: None
    """
    x, y, w, h = coordinates
    cv2.rectangle(image, (x, y), (x+w, y+h), color, thickness)


def exit_():
    """
    custom exit function
    :return: None
    """
    video_capture.release()
    cv2.destroyAllWindows()
    sys.exit()


def main():
    """
    main function
    :return: None
    """
    face_extractor = ObjectExtractor(
        ObjectExtractor.FACE_CASCADE_PATH,
        ObjectExtractor.max_size_selector,)

    while True:
        ret, frame = video_capture.read()
        if not ret:
            continue
        face = face_extractor.detect_object(frame)
        if face is not None:
            draw_rectangle(frame, face)
        cv2.imshow('live', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            exit_()


if __name__ == "__main__":
    main()
