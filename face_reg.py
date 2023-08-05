import numpy as np
import glob
import os
from utils import get_face_encodings, get_face_landmarks, get_face_locations, read_image
from utils import compare_faces



class FaceRec:
    """
    Helper class to get the encdodings and manipulate parameters
    """
    def __init__(self, jitter=1, number_of_times_to_upsample=1, landmark_model=None, encoder_model=None):
        self.jitter = jitter   # used to compute the face_encodings
        self.number_of_times_to_upsample = number_of_times_to_upsample  # used for face_locations
        self.landmark_model = landmark_model
        self.encoder_model = encoder_model

    def detect_faces(self, image):
        return get_face_locations(image, self.number_of_times_to_upsample)

    def calculate_encodings(self, image, face_locations=None):
        # image = self._read_image(filename)
        if face_locations is None:
            face_locations = get_face_locations(image, self.number_of_times_to_upsample)
        
        face_landmarks = get_face_landmarks(image, face_locations, self.landmark_model)
        face_encodings = get_face_encodings(image, face_landmarks, self.jitter, self.encoder_model)
        return face_encodings
    
    def get_similarity(self, known_encodings, encodings):
        """
        :param known_encodings: a list/array of encodings for each id shape --> (N_id, emb_dim)
        :param encodings: a single encoding of the face to compare shape--> (N_emb,)
        """
        # unsqueeze
        encoding = np.expand_dims(encodings, axis=0)
        mask, dist = compare_faces(known_encodings, encoding)
        if sum(mask) > 1:
            mask = [dis == min(dist) for dis in dist]
        return mask, dist

    def generate_encodings(self, train_dir):
        identities = [name for name in os.listdir(train_dir) if name != "Unknown"]
        print(identities)
        self.encoding_dict = {}
        for id in identities:
            # compute the face encoding for the db entries
            face_images = list(
                map(read_image, glob.glob(os.path.join(self.db_path, id, "*.png")))
            )
            face_encodings = list(map(self.calculate_encodings, face_images))
            print(glob.glob(os.path.join(self.db_path, id, "*.png")))
            print(np.array(face_images).shape)
            print(np.array(face_encodings).shape)

            self.encoding_dict[id] = np.mean(face_encodings, axis=0)

        return self.encoding_dict