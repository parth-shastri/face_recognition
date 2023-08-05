import dlib
from PIL import Image
import numpy as np
import config
import cv2

face_detector = dlib.get_frontal_face_detector()

landmark_predictor = dlib.shape_predictor(config.FACE_LANDMARK_MODEL)

face_recognition_model = dlib.face_recognition_model_v1(config.FACE_ENCODER_MODEL)


def read_image(filename, mode="RGB"):
    image = Image.open(filename)
    if mode:
        image = image.convert(mode)
    
    image = np.array(image)
    return image


def _rect_to_css(rect):
    """
    Convert a dlib 'rect' object to a plain tuple in (top, right, bottom, left) order

    :param rect: a dlib 'rect' object
    :return: a plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return rect.top(), rect.right(), rect.bottom(), rect.left()


def _css_to_rect(css):
    """
    Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :return: a dlib `rect` object
    """
    return dlib.rectangle(css[3], css[0], css[1], css[2])


def _trim_css_to_bounds(css, image_shape):
    """
    Make sure a tuple in (top, right, bottom, left) order is within the bounds of the image.

    :param css:  plain tuple representation of the rect in (top, right, bottom, left) order
    :param image_shape: numpy shape of the image array
    :return: a trimmed plain tuple representation of the rect in (top, right, bottom, left) order
    """
    return (
        max(css[0], 0),
        min(css[1], image_shape[1]),
        min(css[2], image_shape[0]),
        max(css[3], 0),
    )


def face_distance(face_encodings, face_to_compare):
    """
    Given a list of face encodings, compare them to a known face encoding and get a euclidean distance
    for each comparison face. The distance tells you how similar the faces are.

    :param face_encodings: List of face encodings to compare
    :param face_to_compare: A face encoding to compare against
    :return: A numpy ndarray with the distance for each face in the same order as the 'faces' array
    """
    if len(face_encodings) == 0:
        return np.empty((0))

    return np.linalg.norm(face_encodings - face_to_compare, axis=1)


def get_face_locations(image, number_of_times_to_upsample):
    """
    Given the image, return the face locations using a hog model.
    """
    detections = face_detector(image, number_of_times_to_upsample)
    face_locations = [
        _trim_css_to_bounds(_rect_to_css(face), image.shape) for face in detections
    ]
    return face_locations


def get_face_landmarks(image, face_locations, model=None):
    """
    Given the image and the face locations get the face landmarks.
    """
    if model:
        landmark_predictor = dlib.shape_predictor(model)

    face_locations = [_css_to_rect(face_location) for face_location in face_locations]
    landmarks = [
        landmark_predictor(image, face_location) for face_location in face_locations
    ]
    return landmarks


def get_face_encodings(image, known_face_landmarks, jitter=1, model=None):
    """
    Given the image containing one or more faces, the known face landmarks to get the face encodings.
    """
    if model:
        face_recognition_model = dlib.face_recognition_model_v1(model)

    face_encodings = [
        np.array(
            face_recognition_model.compute_face_descriptor(image, raw_landmark, jitter)
        )
        for raw_landmark in known_face_landmarks
    ]
    return face_encodings


def compare_faces(known_face_encodings, face_encodings, threshold=0.6):
    """
    Given encodings of an unknown image compare them with a known image and return a truth value if similar
    """
    face_encodings_array = np.array(face_encodings)
    known_face_encodings_array = np.array(known_face_encodings)
    face_distances = face_distance(face_encodings_array, known_face_encodings_array)
    return list(face_distances <= threshold), face_distances


if __name__ == "__main__":
    image_path = "lfw/Bill_Gates/Bill_Gates_0001.jpg"
    image = Image.open(image_path)
    
    image = np.array(image)

    face_locations = get_face_locations(image, 1)
    face_landmarks = get_face_landmarks(image, face_locations)
    face_encodings = get_face_encodings(image, face_landmarks, jitter=1)

    print(np.array(face_encodings).shape)
    for location in face_locations:
        # [top, right, bottom, left]
        top, right, bottom, left = location
        cv2.rectangle(image, [top, left], [bottom, right], (255, 0, 0), 2)
    
    # Image.fromarray(image).show()

    comp_image_path = "lfw/Bill_Doba/Bill_Doba_0001.jpg"
    comp_image = Image.open(comp_image_path)
    
    comp_image = np.array(comp_image)

    comp_face_locations = get_face_locations(comp_image, 1)
    comp_face_landmarks = get_face_landmarks(comp_image, comp_face_locations)
    comp_face_encodings = get_face_encodings(comp_image, comp_face_landmarks, jitter=1)

    print(compare_faces(face_encodings, comp_face_encodings))