import utils.OAK_D_api as oak
import cv2
import torch
import numpy as np
import time
from utils.yolo_inference import yolo_inference
from models.common import DetectMultiBackend
from sort import Sort
from utils.feature_engineering import calculate_features
from ultralytics.utils.plotting import Annotator, colors
from joblib import load

   
class HandWaveClassifier():
    def __init__(self, height, width, fps, yolo_path, classifier_path):
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
        self.wave_length = 40 # should be same as in the training of the classifier

    def run(self):
        wave = 0 # 0 if no wave, 1 if wave detected
        
        # take frame from camera
        frame = self.oak_d.get_color_frame(show_fps=True)

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
                if wave:
                    clr = 7 # green

            annotator.box_label([x1, y1, x2, y2], label, color=colors(clr, True))
 
        # returns whether we have detected a wave or not
        # alongside with annotated frame
        return img, wave

    
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
            return self.classifier.predict(x)
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
        

if __name__ == '__main__':
    detector = HandWaveClassifier(height=1080, width=1920, fps=30,
                                yolo_path='./runs/train/yolov5s_results3/weights/best.pt',
                                classifier_path='models/classifier/wave_classifier.pkl')
    
    while True:
        frame, wave = detector.run()
        cv2.imshow("WaveDetector", frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break