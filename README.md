# Face Detection and Identification App

This project demonstrates face detection and identification using Streamlit. It includes two main components: a command-line script "video_detector.py" for face detection and separation from videos, and a Streamlit app "main.py" for visualizing the results.

## Installation:

### 1. Clone the repository:

    git clone https://github.com/your-username/face-detection-app.git
    cd face-detection-app

### 2. Create a virtual environment (optional but recommended):
    python -m venv venv
    source venv/bin/activate      # On Windows: 
    venv\Scripts\activate

### 3. Install the required dependencies:
    pip install -r requirements.txt

## Usage: Video Reader (Face Detection and Separation)

The "video_detector.py" script allows you to perform face detection on a video and segregate the detected faces into separate folders under a directory with the same name as the video file.

The flags --detect_faces and --identify_faces can be unset to display pure stream, and only --detect_faces can be used to just detect the faces in real-time

Usage:

    python video_detector.py --source /path/to/video.mp4 --detect_faces --identify_faces \
    --landmark_model /path/to/landmark_model.pth \
    --encoder_model /path/to/encoder_model.pth \
    --save_path /path/to/save_results \
    --encoder_jitter 5 --detector_upsample 1

-   `--source`: Path to the input video file.
-   `--detect_faces`: Flag to enable face detection.
-   `--identify_faces`: Flag to enable face identification.
-   `--landmark_model`: Path to the face landmark model.
-   `--encoder_model`: Path to the face encoder model.
-   `--save_path`: Path to the directory where results will be saved.
-   `--encoder_jitter`: Number of times to apply jitter for face encoding.
-   `--detector_upsample`: Number of times to upsample input image during detection.

## Usage: Streamlit App

The "main.py" script is a Streamlit app that lets you interactively visualize the face detection and identification results.

Usage:

    streamlit run main.py

Open your web browser and navigate to the provided local URL (usually http://localhost:8501).

Project Structure:

.

├── output/

├── video_detector.py

├── face_rec.py

├── utils.py

├── main.py

├── requirements.txt

└── README.md


## Results:

TODO

## License:

This project is licensed under the [MIT License.](LICENSE)

