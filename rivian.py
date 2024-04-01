import OAK_D_api as oak
import cv2
import torch
from inference import run
from models.common import DetectMultiBackend


def run_object_detection():
    
    # Load model
    weights_path = './runs/train/yolov5s_results3/weights/best.pt'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DetectMultiBackend(weights_path, device=device, dnn=False, fp16=False)
    model.eval()

    # camera setup
    oak_d = oak.OAK_D(fps=60, width=1920, height=1080)

    while True:
        frame = oak_d.get_color_frame(show_fps=True)
        img, results_for_bounding_boxes = run(frame=frame, classes=[1], model=model)
        
        cv2.imshow("Levi", img)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

if __name__ == '__main__':
    run_object_detection()
    
