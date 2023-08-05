import cv2
import os
from time import sleep
import datetime
import config
import numpy as np
from argparse import ArgumentParser
from face_reg import FaceRec

face_reg = FaceRec(
        jitter=config.JITTER,
        number_of_times_to_upsample=config.NO_OF_TIMES_TO_UPS,
        landmark_model=config.FACE_LANDMARK_MODEL,
        encoder_model=config.FACE_ENCODER_MODEL
        )

class VideoReader:
    def __init__(
        self,
        source,
        save_every=5,
        detect_faces=False,
        identify_faces=False,
        save_path="./output"
    ):
        self.source = source
        # self.fps = fps
        self.save_every = save_every  # in seconds
        self.detect_faces = detect_faces
        self.identify_faces = identify_faces
        self.save_path = save_path

        if self.identify_faces and self.detect_faces:
            # initialize the universal encoding buffer
            # a dict mapping from "id": [encoding]
            self.identity_encodings = {}
            self.count_id_enc = {}  # a count of each of the embeddings saved
            # make the directory for the video
            name = (
                "{}_{}".format(self.source, datetime.datetime.now().strftime("%H%M%S"))
                if source == "webcam"
                else "{}".format(self.source.split("/")[-1])
            )
            self.video_dir = os.path.join(self.save_path, name)
            try:
                os.mkdir(self.video_dir)
            except FileExistsError:
                print("The Directory already exists!")

        else:
            raise ValueError("Cannot Identify faces without detecting them. Please set detect_faces to True")

    def display(
        self, on_streamlit=False, stop_button=False, frame_display=None
    ):
        if self.source == "webcam":
            cap = cv2.VideoCapture(0)
        elif os.path.isfile(self.source):
            cap = cv2.VideoCapture(self.source)
        else:
            raise ValueError("Unexpected Source {}, either provide a file or 'webcam'")

        # cap.set(cv2.CAP_PROP_FPS, self.fps)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        print("Video FPS : {}".format(cap.get(cv2.CAP_PROP_FPS)))

        if not cap.isOpened():
            print("Cannot Open source")
            exit()

        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if self.detect_faces:
                # perform face detection and draw the bbox on the frame
                try:
                    face_locations = face_reg.detect_faces(gray)
                except Exception as e:
                    print("Something wrong in the face_reg module ! Code exited with an exception' {}'".format(e))
                    exit()

                face_crops = []

                for location in face_locations:
                    top, right, bottom, left = location
                    cv2.rectangle(frame, [left, top], [right, bottom], (255, 0, 0), 2)
                    face_crops.append(frame[top:bottom, left:right, ...])

            if self.identify_faces:

                try:
                    encodings = face_reg.calculate_encodings(frame, face_locations)
                except Exception as e:
                    print("Something wrong in the face_reg module ! Code exited with an exception '{}'".format(e))
                    exit()

                # if more than one faces is detected in a single frame then categorize them into different ids
                if len(encodings) > 0:
                    if len(self.identity_encodings) == 0:
                        # make the dir corresponding to each identity
                        for i in range(len(encodings)):
                            name = "ID_{}".format(i)
                            dirname = os.path.join(self.video_dir, name)
                            try:
                                os.mkdir(dirname)
                            except:
                                print("{} already exists !".format(dirname))

                            if name not in self.identity_encodings:
                                self.identity_encodings[name] = encodings[i]
                                self.count_id_enc[name] = 1

                                # print(encodings[i].shape) (128,)

                            # save the detected faces in their respective dirs
                            cv2.imwrite(
                                os.path.join(
                                    dirname,
                                    "{}_{}.png".format(
                                        name, datetime.datetime.now().strftime("%H%M%S")
                                    ),
                                ),
                                img=face_crops[i],
                            )

                    elif len(self.identity_encodings) > 0:
                        # if there are previous ids in the id encodings
                        # compare the face encodings from the current frame using similarity and threshold
                        # if there is a match put it in the respective dir
                        # else put that into a new directory and append it into the id encodings
                        id_encodings = np.array(list(self.identity_encodings.values()))
                        ids = list(self.identity_encodings.keys())
                        for i, cur_encoding in enumerate(encodings):
                            
                            try:
                                sim_mask, dist = face_reg.get_similarity(id_encodings, cur_encoding)
                            except Exception as e:
                                print("Something wrong in the face_reg module ! Code exited with an exception '{}'".format(e))
                                exit()


                            selected_id = np.array(ids)[sim_mask]
                            print(selected_id, dist)

                            if len(selected_id) == 0:
                                # if the selected id empty
                                # create a new entry in the dir and the identity encodings
                                print("Found a new identity...")
                                name = "ID_{}".format(len(ids))
                                dirname = os.path.join(self.video_dir, name)
                                try:
                                    os.mkdir(dirname)
                                except:
                                    print("{} already exists !".format(dirname))

                                if name not in self.identity_encodings:
                                    print("Adding to global encoding database...")
                                    self.identity_encodings[name] = encodings[i]
                                    self.count_id_enc[name] = 1

                                # print(encodings[i].shape) (128,)

                                # save the detected faces in their respective dirs
                                cv2.imwrite(
                                    os.path.join(
                                        dirname,
                                        "{}_{}.png".format(
                                            name,
                                            datetime.datetime.now().strftime("%H%M%S"),
                                        ),
                                    ),
                                    img=face_crops[i],
                                )

                            else:
                                # write the image in that dir
                                name = selected_id[0]
                                dirname = os.path.join(self.video_dir, name)
                                cv2.imwrite(
                                    os.path.join(
                                        dirname,
                                        "{}_{}.png".format(
                                            name,
                                            datetime.datetime.now().strftime("%H%M%S"),
                                        ),
                                    ),
                                    img=face_crops[i],
                                )
                                # update the count and keep the Running Mean of the id-encoding
                                self.identity_encodings[name] *= self.count_id_enc[name]
                                self.identity_encodings[name] += cur_encoding
                                self.identity_encodings[name] /= (self.count_id_enc[name] + 1 )
                                self.count_id_enc[name] += 1

            # if self.save_frames and (frame_count % (self.save_every * self.fps) == 0):
            #     print("Saving at frame number: {}".format(frame_count))
            #     for i, crop in enumerate(face_crops):
            #         name = "person_{}_{}.png".format(
            #             frame_count, datetime.datetime.now().strftime("%H%M%S")
            #         )
            #         cv2.imwrite(
            #             os.path.join(
            #                 config.IMAGES_FROM_VIDEO,
            #                 name,
            #             ),
            #             img=crop,
            #         )

            if not on_streamlit:
                cv2.imshow("frame", frame)
            else:
                frame_display.image(frame, channels="BGR")

            frame_count += 1

            if cv2.waitKey(1) == ord("q") or (stop_button):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    # arg parser to run the standalone script
    parser = ArgumentParser()
    parser.add_argument("--source", type=str, default=config.SOURCE)
    parser.add_argument("--detect_faces", action="store_true", default=config.DETECT_FACES)
    parser.add_argument("--identify_faces", action="store_true", default=config.DETECT_FACES)
    parser.add_argument("--landmark_model", type=str, default=config.FACE_LANDMARK_MODEL)
    parser.add_argument("--encoder_model", type=str, default=config.FACE_ENCODER_MODEL)
    parser.add_argument("--save_path", type=str, default=config.IMAGES_SEG)
    parser.add_argument("--encoder_jitter", type=int, default=config.JITTER)
    parser.add_argument("--detector_upsample", type=int, default=config.NO_OF_TIMES_TO_UPS)

    args = parser.parse_args()

    # init the FaceRec

    face_reg = FaceRec(
        jitter=args.encoder_jitter,
        number_of_times_to_upsample=args.detector_upsample,
        landmark_model=args.landmark_model,
        encoder_model=args.encoder_model
        )
    
 
    video_reader = VideoReader(
        source=args.source, save_every=0.5, identify_faces=args.identify_faces, detect_faces=args.detect_faces, save_path=args.save_path
    )
    video_reader.display()
