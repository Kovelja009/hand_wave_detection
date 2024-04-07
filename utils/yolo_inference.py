import torch
import numpy as np
from utils.dataloaders import OakDLoadImages
from utils.general import (
    Profile,
    check_img_size,
    non_max_suppression,
    scale_boxes,
)
from utils.torch_utils import smart_inference_mode


@smart_inference_mode()
def yolo_inference(
    frame=None,  # openCV image
    imgsz=(640, 640),  # inference size (height, width)
    conf_thres=0.25,  # confidence threshold
    iou_thres=0.45,  # NMS IOU threshold
    max_det=1000,  # maximum detections per image
    device="",  # cuda device, i.e. 0 or 0,1,2,3 or cpu
    classes=None,  # filter by class: --class 0, or --class 0 2 3
    agnostic_nms=False,  # class-agnostic NMS
    model=None
):
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    dataset = OakDLoadImages(frame, img_size= imgsz[0], stride=stride, auto=pt)
    
    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, _, dt = 0, [], (Profile(device=device), Profile(device=device), Profile(device=device))
    for im, im0 in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim
            if model.xml and im.shape[0] > 1:
                ims = torch.chunk(im, im.shape[0], 0)

        # Inference
        with dt[1]:
            if model.xml and im.shape[0] > 1:
                pred = None
                for image in ims:
                    if pred is None:
                        pred = model(image, augment=False, visualize=False).unsqueeze(0)
                    else:
                        pred = torch.cat((pred, model(image, augment=False, visualize=False).unsqueeze(0)), dim=0)
                pred = [pred, None]
            else:
                pred = model(im, augment=False, visualize=False)
        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        results_for_bounding_boxes = []
        original_image = im0.copy()
        # Process predictions
        for _, det in enumerate(pred):  # per image
            seen += 1
            # annotator = Annotator(im0, line_width=line_thickness, example=str(names))
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Get results
                for *xyxy, conf, cls in reversed(det):
                    lst_xyxy = [elem.cpu().item() for elem in xyxy]
                    lst_xyxy_conf_cls = lst_xyxy + [conf.cpu().item()] + [cls.cpu().item()]                    
                    results_for_bounding_boxes.append(np.array(lst_xyxy_conf_cls))
                
    return original_image, results_for_bounding_boxes
