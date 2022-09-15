import torch
import numpy as np
import cv2
from PIL import Image
# import onnxruntime as ort

import clip

import warnings
warnings.filterwarnings('ignore', message="User provided device_type of 'cuda'")

from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker


device = "cuda" if torch.cuda.is_available() else "cpu"
half = device != "cpu"


class ClipTracker:
    
    def __init__(self, 
            clip_model="ViT-B/16", 
            max_cosine_distance=0.4, 
            nn_budget=None, 
            nms_max_overlap=0.7, 
            conf_threshold=0.5,
            img_size=640, 
            auto_letterbox=True, 
            letterbox_stride=32,
            patch_dilate=1,
    ):
        self.img_size = img_size
        self.auto_letterbox = auto_letterbox
        self.letterbox_stride = letterbox_stride
        self.nms_max_overlap = nms_max_overlap
        self.patch_dilate = patch_dilate
        self.conf_threshold = conf_threshold

        # object bounding boxes
        self.yolo = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        self.yolo.eval()
        # self.yolo = ort.InferenceSession('yolov7-e6.onnx')
        # print([x.name for x in self.yolo.get_inputs()], [x.name for x in self.yolo.get_outputs()])
        # print([x.shape for x in self.yolo.get_inputs()], [x.shape for x in self.yolo.get_outputs()])

        # object features
        self.clip_model, self.clip_transform = clip.load(clip_model, device=device, jit=False)
        self.clip_model.eval()

        # instance tracker
        metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_cosine_distance, nn_budget)
        self.tracker = Tracker(metric)

    def update(self, img):
        # get detections
        bboxes, confs, class_nums, img2 = self.detect_objects(img)
        indices = nms(bboxes, confs, self.nms_max_overlap)
        bboxes, confs, class_nums = bboxes[indices], confs[indices], class_nums[indices]
        class_nums = class_nums.astype(int)

        features = self.get_features(img2, bboxes)
        detections = [
            Detection(bbox, conf, feature) 
            for bbox, conf, feature in 
            zip(bboxes, confs, features)
        ]
        for c, d in zip(class_nums, detections):
            d.class_num = c

        # update the tracker
        self.tracker.predict()
        self.tracker.update(detections)

    def detect_objects(self, im0):
        # Padded resize
        img, img2, gain, pad = letterbox(im0, self.img_size, stride=self.letterbox_stride, auto=self.auto_letterbox)
        # Convert
        img = np.ascontiguousarray(img.transpose((2, 0, 1))[::-1])  # HWC to CHW, BGR to RGB
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # run model!
        with torch.no_grad():
            det = self.yolo(img)[0]
        det = det.cpu().numpy()
        # det = self.yolo.run(None, {self.yolo.get_inputs()[0].name: img.numpy()})[self.yolo.get_outputs()[0].name]

        det = det[det[:, 4] > self.conf_threshold]

        # get boxes classes and confidences
        bboxes = det[:, :4]
        confs = det[:, 4]
        class_nums = det[:, -1]
        
        # undo letterbox
        bboxes[:, [0, 2]] = ((bboxes[:, [0, 2]] - pad[0]) / gain[0]).clip(0, im0.shape[0])  # x padding
        bboxes[:, [1, 3]] = ((bboxes[:, [1, 3]] - pad[1]) / gain[1]).clip(0, im0.shape[1])  # y padding
        bboxes[:, 2:] -= bboxes[:, :2]  # xyxy to tlwh

        return bboxes, confs, class_nums, img2


    def get_features(self, img, bboxes):
        with torch.no_grad():
            features = self.clip_model.encode_image(torch.stack([
                self.clip_transform(Image.fromarray(extract_image_patch(img, box))).to(device)
                for box in bboxes
            ])).cpu().numpy()
        return features

def extract_image_patch(image, bbox, dilate=None):
    x1, y1, w, h = np.asarray(bbox)
    xpad, ypad = (w*(dilate-1)/2, h*(dilate-1)/2) if dilate else (0, 0)
    x2 = x1+w
    y2 = y1+h
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = int(x1-xpad)
    y1 = int(y1-ypad)
    x2 = int(x1+w+xpad)
    y2 = int(y1+h+ypad)
    return image[y1:y2, x1:x2]


# https://github.com/ultralytics/yolov5/blob/92e47b85d952274480c8c5efa5900e686241a96b/utils/augmentations.py#L91
def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im_lb = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im_lb, im, ratio, (dw, dh)


def nms(boxes, scores, threshold):
    n_active = len(boxes)
    active = np.array([True]*len(boxes))
    idxs = np.argsort(scores)

    for i, ii in enumerate(idxs):
        if not n_active: break
        if not active[i]: continue
        for j, jj in enumerate(idxs[i+1:], i+1):
            if not active[j]: continue
            if IoU(boxes[ii], boxes[jj]) > threshold:
                active[j] = False
                n_active -= 1
    return idxs[active]

def IoU(boxA, boxB):
    ax, ay, aw, ah = boxA
    bx, by, bw, bh = boxB
    areaA = aw * ah
    areaB = bw * bh
    if areaA <= 0 or areaB <= 0: return 0
    intersectionArea = (
        max(min(ay + ah, by + bh) - max(ay, by), 0) * 
        max(min(ax + aw, bx + bw) - max(ax, bx), 0))
    return intersectionArea / (areaA + areaB - intersectionArea)




def draw_tracks(tracker, im0, names=None, **kw):
    tracks = [track for track in tracker.tracks if not track.is_confirmed() or track.time_since_update > 1]
    print([t.track_id for t in tracks])
    for track in tracks:
        label = str(track.track_id)#f'{names[track.class_num] if names else track.class_num} #{track.track_id}'
        plot_one_box(track.to_tlbr(), im0, label=label, **kw)



colors = ["#4892EA", "#00EEC3", "#FE4EF0", "#F4004E", "#FA7200", "#EEEE17", "#90FF00", "#78C1D2", "#8C29FF"]
def get_color_for(class_num):  # id > hex > rgb: https://stackoverflow.com/questions/29643352/converting-hex-to-rgb-value-in-python
    hex = colors[hash(class_num)%len(colors)]
    return tuple(int(hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or (get_color_for(label) if label else None) or [np.random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)




def _video_feed(src=0):
    cap = cv2.VideoCapture(src)
    while True:
        ret, im = cap.read()
        if not ret:
            break
        yield im

def main(src):
    ctrack = ClipTracker()
    from pyinstrument import Profiler
    try:
        with Profiler() as prof:
            for im in _video_feed(src):
                ctrack.update(im)
                
                draw_tracks(ctrack.tracker, im, names=classes)
                cv2.imshow('yay!', im)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration
    except KeyboardInterrupt:
        print("\nk bye! :)")
    finally:
        prof.print()

classes = [
    'person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant',
    'stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe',
    'backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat',
    'baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl',
    'banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant',
    'bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink',
    'refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush']


if __name__ == '__main__':
    import fire
    fire.Fire(main)