import utils.OAK_D_api as oak
import cv2
import torch
import numpy as np
import time
import threading
from utils.yolo_inference import yolo_inference
from models.common import DetectMultiBackend
from sort import Sort
from utils.feature_engineering import calculate_features
from ultralytics.utils.plotting import Annotator, colors
from joblib import load

   
class HandWaveClassifier():
    def __init__(self, height, width, fps, yolo_path, classifier_path, wave_length=40):
        self.height = height
        self.width = width
        self.fps = fps
        self.yolo_path = yolo_path
        self.classifier_path = classifier_path
        self._camera_setup()
        self._load_yolo()
        self._init_tracker()
        self._load_classifier()
        self.ids_list = []  # List to keep track of unique IDs
        self.tracks = {} # keeping x_c and y_c for each id
        self.wave_length = wave_length # should be same as in the training of the classifier

        # Read the overlay video
        self.overlay_video_path = 'videos/anime-waves.mp4'
        self.overlay_video = cv2.VideoCapture(self.overlay_video_path)
        self.overlay_duration = 0.7 # Duration of the overlay video in seconds 
        self.should_overlay = False # Flag to check if overlay should be displayed
        self.start_time = None # Start time of the overlay video  


    def run(self):
        # wave = 1  # 0 if no wave, 1 if wave detected
        cum_waves = 0  # counter for the number of waves

        # Your existing code to detect waves and annotate the frame
        frame, cum_waves = self._detect_waves_and_annotate()

        if cum_waves > 0:
            self.should_overlay = True
            self.start_time = time.time()
        if self.should_overlay and time.time() - self.start_time > self.overlay_duration:
            self.should_overlay = False

        # Overlay video if wave is detected
        if self.should_overlay:
            frame = self._overlay_video(frame)

        return frame, cum_waves
    
    def _overlay_video(self, frame):
        # Get the dimensions of the video
        overlay_width = int(self.overlay_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        overlay_height = int(self.overlay_video.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Calculate new dimensions based on the desired scale
        new_width = int(overlay_width * 0.4)
        new_height = int(overlay_height * 0.4)

        # Define the position for overlay (top right corner)
        frame_height, frame_width, _ = frame.shape
        x_offset = frame_width - new_width
        y_offset = 0

        # Loop through the overlay video frames
        ret, overlay_frame = self.overlay_video.read()
        if not ret:
            # If end of video is reached, rewind to beginning
            self.overlay_video.release()
            self.overlay_video = cv2.VideoCapture(self.overlay_video_path)
            ret, overlay_frame = self.overlay_video.read()

        if ret:
            # Resize the overlay frame based on the new dimensions
            overlay_frame = cv2.resize(overlay_frame, (new_width, new_height))

            # Overlay the frame onto the main frame
            frame[:new_height, x_offset:] = overlay_frame

        return frame


    '''Returns annotated image with bounding boxes and labels and the number of waves detected in the frame'''
    def _detect_waves_and_annotate(self):
        wave = 0 # 0 if no wave, 1 if wave detected
        cum_waves = 0 # counter for the number of waves

        # take frame from camera
        frame, camera_fps = self.oak_d.get_color_frame(show_fps=False)

        # Object detection
        img, bbox_coord_conf_cls = yolo_inference(frame=frame, classes=[0,1,2], model=self.yolo, device=self.device)
        annotator = Annotator(img, line_width=3, example=str(self.yolo.names))

        # Update tracker
        if len(bbox_coord_conf_cls) > 0: # If there are detections
            track_bbs_ids = self.tracker.update(bbox_coord_conf_cls)
        else: # If no detections update with empty list
            track_bbs_ids = self.tracker.update(np.empty((0, 5)))

        names = [self.yolo.names[int(cls)] for x1, y1, x2, y2, conf, cls in bbox_coord_conf_cls]
        confs = [conf for x1, y1, x2, y2, conf, cls in bbox_coord_conf_cls]

        # Draw frame rate on the frame
        cv2.putText(img, f'{camera_fps:.2f} fps', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 150), 2, cv2.LINE_AA)

        # Draw bounding boxes and labels for the tracked objects
        for i, bb_id in enumerate(track_bbs_ids):
            coords = bb_id[:4]
            x1, y1, x2, y2 = [int(i) for i in coords]

            # Get ID of the object
            if bb_id[8] not in self.ids_list:
                self.ids_list.append(bb_id[8]) # Add new ID to the list if not already present
        
            name_idx = self.ids_list.index(bb_id[8]) # Get the index of the ID in the list

            # Create label with class, confidence and ID
            label = names[i] + f' {confs[i]:.2f} ' + 'ID:{}'.format(str(name_idx))
            clr = int(bb_id[4])

            # add tracking point to dictionary
            self._add_point(name_idx, int((x1 + x2) / 2), int((y1 + y2) / 2))

            # only run the classificator if the object is a palm
            if names[i] == 'palm':
                # run the classificator
                wave = self._run_classificator(name_idx)
                cum_waves += wave
                if wave:
                    clr = 7 # green

            annotator.box_label([x1, y1, x2, y2], label, color=colors(clr, True))

        # show text of the number of waves detected
        cv2.putText(img, f'Waves detected: {cum_waves}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
 
        return img, cum_waves

    
    def _add_point(self, id, xc, yc):
        if id not in self.tracks:
            self.tracks[id] = {'xc': [], 'yc': []}
        self.tracks[id]['xc'].append(xc)
        self.tracks[id]['yc'].append(yc)


    def _run_classificator(self, id):
        # check whether we have datapoints for this id
        if id in self.tracks and len(self.tracks[id]['xc']) > self.wave_length:
            # get the last length points
            x_signal = self.tracks[id]['xc'][-self.wave_length:]
            y_signal = self.tracks[id]['yc'][-self.wave_length:]
            features = calculate_features(x_signal, y_signal, sampling_rate=30, width=1920, height=1080)
            x = np.array(list(features.values())).reshape(1, -1)
            
            # run the classifier
            return self.classifier.predict(x)[0]
        else:
            return 0


    def _camera_setup(self):
        self.oak_d = oak.OAK_D(fps=self.fps, width=self.width, height=self.height)


    def _load_yolo(self):
        # Loading pretrained model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.yolo = DetectMultiBackend(self.yolo_path, device=self.device, dnn=False, fp16=False)
        self.yolo.eval()


    def _init_tracker(self):
        # Initialize SORT tracker
        self.tracker = Sort(min_hits=5, max_age=20)


    def _load_classifier(self):
        # Loading pretrained classifier
        self.classifier = load(self.classifier_path)
        

def play_video(video_path, width, height):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Read and display frames until the end of the video
    while cap.isOpened():
        ret, frame = cap.read()

        # Check if frame is successfully read
        if not ret:
            break

        # resize the frame
        frame = cv2.resize(frame, (width, height))
        
        # show text
        cv2.putText(frame, "Please be patient while model loads...", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 3, (104, 188, 222), 4, cv2.LINE_AA)
        cv2.putText(frame, "You can't depend on your eyes when your imagination", (1000, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "is out of focus", (1000, 1000), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

        # Display the frame
        cv2.imshow('Loading model', frame)

        
        cv2.waitKey(50)

    # Release the video capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()



class DetectThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.detector = None
        self.event = threading.Event()

    def run(self):
        self.detector = HandWaveClassifier(height=height, width=width, fps=30,
                                    yolo_path='./runs/train/yolov5s_results3/weights/best.pt',
                                    classifier_path='models/classifier/wave_classifier.pkl')
    
        self.event.set()

    def get_detector(self):
        self.event.wait()
        return self.detector


if __name__ == '__main__':
    width, height = 1920, 1080
    initial_video_path = 'videos/rivian.mp4'

    # Initialize the detector in a separate thread to load the model
    init_thread = DetectThread()

    init_thread.start()

    # Play the video while the model is being loaded
    play_video(initial_video_path, width, height)
    
    detector = init_thread.get_detector()

    while True:
        # print("Running detector")
        frame, wave = detector.run()
        cv2.imshow("WaveDetector", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break