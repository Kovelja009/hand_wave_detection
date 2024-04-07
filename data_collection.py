import utils.OAK_D_api as oak
import cv2
import torch
import numpy as np
import time
from utils.yolo_inference import yolo_inference
from models.common import DetectMultiBackend
from sort import Sort
from utils.video_utils import save_video_tracking_data
from ultralytics.utils.plotting import Annotator, colors

def run_object_tracking():
    
    # Load model
    weights_path = './runs/train/yolov5s_results3/weights/best.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights_path, device=device, dnn=False, fp16=False)
    model.eval()

    # camera setup
    oak_d = oak.OAK_D(fps=60, width=1920, height=1080)

    ############################################################
    ######################## SORT tracker ######################
    # Initialize SORT tracker
    mot_tracker = Sort(min_hits=5, max_age=20)

    ############################################################
    ##################### Logging data #########################
    # Initialize variables
    ids_list = []  # List to keep track of unique IDs
    frame_count = 0  # Counter to keep track of frame number
    file_name = 'data/tracking_data.csv'  # File name to save tracking data
    video_name = 'video3.mp4' # Video name to save tracking data
    elapsed_time = -3 # Time in seconds
    should_save = False 
    wave = 0 # indicator whether video should record wave or not
    # Initialize dictionary to store tracking information
    # {0: 'fist', 1: 'palm', 2: 'no_gesture'}
    saving_class = [model.names[1]]
    tracks = {
        "frame": [],
        "id": [],
        "x1": [],
        "y1": [],
        "x2": [],
        "y2": [],
        "xc": [],
        "yc": [],
        "class": [],
        "wave": [],
        "video": []
    }
    ############################################################
    start_time = time.time()

    while True:
        frame = oak_d.get_color_frame(show_fps=True)
        # Object detection
        img, bbox_coord_conf_cls = yolo_inference(frame=frame, classes=[0,1,2], model=model, device=device)
        annotator = Annotator(img, line_width=3, example=str(model.names))

        # Update tracker
        if len(bbox_coord_conf_cls) > 0: # If there are detections
            track_bbs_ids = mot_tracker.update(bbox_coord_conf_cls)
        else: # If no detections update with empty list
            track_bbs_ids = mot_tracker.update(np.empty((0, 5)))

        names = [model.names[int(cls)] for x1, y1, x2, y2, conf, cls in bbox_coord_conf_cls]
        confs = [conf for x1, y1, x2, y2, conf, cls in bbox_coord_conf_cls]

        # Draw bounding boxes and labels for tracking
        for i, bb_id in enumerate(track_bbs_ids):
            coords = bb_id[:4]
            x1, y1, x2, y2 = [int(i) for i in coords]

            # Get ID of the object
            if bb_id[8] not in ids_list:
                ids_list.append(bb_id[8]) # Add new ID to the list if not already present
        
            name_idx = ids_list.index(bb_id[8]) # Get the index of the ID in the list

            # Create label with class, confidence and ID
            label = names[i] + f' {confs[i]:.2f} ' + 'ID:{}'.format(str(name_idx))

            annotator.box_label([x1, y1, x2, y2], label, color=colors(int(bb_id[4]), True))

            # Save tracking information to dictionary only for the specified classes
            if names[i] in saving_class:
                tracks['frame'].append(frame_count)
                tracks['id'].append(name_idx)
                tracks['x1'].append(x1)
                tracks['y1'].append(y1)
                tracks['x2'].append(x2)
                tracks['y2'].append(y2)
                tracks['xc'].append(int((x1 + x2) / 2))
                tracks['yc'].append(int((y1 + y2) / 2))
                tracks['class'].append(names[i])
                tracks['wave'].append(wave)
                tracks['video'].append(video_name)
            
        frame_count += 1
        cv2.imshow("Levi", img)
        current_time = time.time()

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q') or (current_time - start_time > elapsed_time and elapsed_time > 0):
            cv2.destroyAllWindows()
            if should_save:
                # Save tracking data to CSV file
                save_video_tracking_data(tracks, file_name)
                print("Data appended to CSV file successfully!")
            break

if __name__ == '__main__':
    run_object_tracking()
    
