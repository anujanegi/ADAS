import cv2
import numpy as np


class ObjectExtractor:

    """
    object is assumed to be in the same position;
    for a maximum of MAX_PERSISTENCE_COUNTER frames;
    """
    MAX_PERSISTENCE_COUNTER = 10

    """
    jitter is removed using moving average;
    the window is given by MOVING_AVERAGE_WINDOW;
    """
    MOVING_AVERAGE_WINDOW = 10

    """
    minimum area deviation to update the frame
    """
    MIN_AREA_CHANGE = 2500
    """
    maximum permissible area deviation allowed for updating the frame
    """
    MAX_AREA_CHANGE = 5000

    """
    maximum permissible distance deviation allowed for updating the frame
    """
    MAX_DISTANCE_CHANGE = 400

    """ cascade locations """
    FACE_CASCADE_PATH = "models/haarcascade_frontalface_default.xml"
    EYE_CASCADE_PATH = "models/haarcascade_eye.xml"

    @staticmethod
    def max_size_selector(items):
        """
        returns largest item (area wise)
        :param items: list of items
        :return: largest item
        """
        items.sort(reverse=True, key=lambda x: x[2] * x[3])
        return items[0]

    def __init__(self, classifier_path, object_selector, scale_factor=1.1, min_size=(50, 50), verbose=False):
        """
        initialize an object-extractor
        :param classifier_path: path of classifier file
        :param object_selector: function for selecting best object from a list of objects
        :param scale_factor: scale factor while finding objects
        :param min_size: minimum size of objects
        :param verbose: log data?
        """
        self.moving_average_window = []
        self.classifier = cv2.CascadeClassifier(classifier_path)
        self.object_selector = object_selector
        self.scale_factor = scale_factor
        self.min_size = min_size
        self.persistence_counter = 0
        self.curr_object = None
        self.verbose = verbose

    def detect_object(self, frame):
        """
        detect the object in a frame
        :param frame: object will be searched in this
        :return: object coordinates in frame
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        objects = list(self.classifier.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minSize=self.min_size))

        """ if any object is found """
        if objects and len(objects) > 0:
            selected_object = self.object_selector(objects)
            self.moving_average_window.append(selected_object)
            """ if moving average window is full """
            if len(self.moving_average_window) > self.MOVING_AVERAGE_WINDOW:
                self.moving_average_window.pop(0)
                average = np.mean(self.moving_average_window, axis=0)
                area_difference = abs(np.prod(average[2:]) - np.prod(selected_object[2:]))
                euclidean_distance = np.linalg.norm(average-selected_object)
                if self.verbose:
                    print("average", average)
                    print("area_wise", area_difference)
                    print("euclidean", euclidean_distance)
                    print()
                """ check thresholds """
                if self.MIN_AREA_CHANGE < area_difference < self.MAX_AREA_CHANGE and \
                        euclidean_distance < self.MAX_DISTANCE_CHANGE:
                    self.curr_object = selected_object
            """ reset the counter """
            self.persistence_counter = 0
        else:
            if self.persistence_counter < self.MAX_PERSISTENCE_COUNTER:
                """ show for some more frames """
                self.persistence_counter += 1
            else:
                """ reset """
                self.curr_object = None
                self.persistence_counter = 0
        return self.curr_object
