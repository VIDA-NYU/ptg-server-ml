import numpy as np
from collections import deque, Counter, defaultdict
import heapq
import cv2

SCORE_THRESHOLD = 1.5
UNSEEN_PENALTY = 1
LABEl_WINDOW_SIZE = 10

BOUNDARY_OFFSET = 20


class PredictionEntry:
    def __init__(self, pos, label, confidence):
        self.pos = pos
        self.label = label
        self.confidence = confidence


class MemoryEntry:
    def __init__(self, id, pos, label, confidence, timestamp):
        self.id = id
        self.pos = pos
        self.labels = deque([label])
        self.confidence = confidence
        self.last_seen = timestamp

    def update(self, pos, label, confidence, timestamp):
        self.pos = self.pos * (1-confidence) + pos * confidence
        self.labels.append(label)
        if len(self.labels) > LABEl_WINDOW_SIZE:
            self.labels.popleft()
        self.confidence += confidence
        self.last_seen = timestamp

    def get_label(self):
        c = Counter(self.labels)
        return c.most_common(1)[0][0]

    def __repr__(self):
        return "id: {}, pos: {}, labels: {}, conf: {}, last_seen: {}".format(
            self.id, self.pos, self.labels, self.confidence, self.last_seen)

    def json_dict(self):
        return {"xyz_center": self.pos, "confidence": self.confidence, "label": self.get_label(), "id": self.id}


class Memory:
    def __init__(self):
        self.objects = {}
        self.id = 0

    def update(self, detections, confidence, timestamp, intrinsics, world2pv_transform, img_shape):
        # E-step
        scores = []
        for idx, d in enumerate(detections):
            for k, o in self.objects.items():
                score = -getScore(d, o)
                if score < -SCORE_THRESHOLD:
                    scores.append((score, idx, k))

        # M-step
        heapq.heapify(scores)
        matching = {}
        matched_mem_key = set()
        while len(matching) < len(detections) and len(matched_mem_key) < len(self.objects) and scores:
            _, det_i, mem_key = heapq.heappop(scores)
            if det_i in matching or mem_key in matched_mem_key:
                continue
            matching[det_i] = mem_key
            matched_mem_key.add(mem_key)

        # update
        for det_i, mem_key in matching.items():
            d = detections[det_i]
            self.objects[mem_key].update(
                d.pos, d.label, confidence, timestamp)

        # unseen objects:
        to_remove = []
        for mem_k, mem_entry in self.objects.items():
            if mem_k not in matched_mem_key and checkInsideFOV(
                    mem_entry.pos, intrinsics, world2pv_transform, img_shape):
                mem_entry.confidence -= UNSEEN_PENALTY
                if mem_entry.confidence < 0:
                    to_remove.append(mem_k)
        for k in to_remove:
            del self.objects[k]

        # new objects:
        for det_i, d in enumerate(detections):
            if det_i not in matching and detections[det_i].confidence > 0.6:
                self.objects[self.id] = MemoryEntry(
                    self.id, d.pos, d.label, confidence, timestamp)
                self.id += 1

    def __repr__(self):
        strs = ["num objects: {}".format(len(self.objects))]
        for obj in self.objects.values():
            strs.append(str(obj))
        return '\n'.join(strs)

    def to_json(self):
        return [i.json_dict() for i in self.objects.values()]


rvec = np.zeros(3)
tvec = np.zeros(3)


def checkInsideFOV(pos, intrinsics, world2pv_transform, img_shape):
    p = world2pv_transform @ np.hstack((pos, [1]))
    if p[2] > 0:
        return False
    xy, _ = cv2.projectPoints(
        p[:3], rvec, tvec, intrinsics, None)
    xy = np.squeeze(xy)
    height, width = img_shape
    xy[0] = width - xy[0]
    return BOUNDARY_OFFSET <= xy[0] < width-BOUNDARY_OFFSET and BOUNDARY_OFFSET <= xy[1] < height-BOUNDARY_OFFSET


def getScore(pred: PredictionEntry, mem: MemoryEntry):
    return getPositionScore(pred, mem) + getLabelScore(pred, mem)


def getPositionScore(pred, mem):
    return min(2 * np.exp(5 * -np.linalg.norm(pred.pos - mem.pos)), 3)


def getLabelScore(pred, mem):
    return 1.6 * pred.confidence * sum(pred.label == i for i in mem.labels) / len(mem.labels) * (0.6 + min(0.4, mem.confidence / 10))


def nms(results, shape, threshold=0.4):
    if len(results) == 0:
        return []
    height, width = shape
    boxes = np.zeros((len(results), 4))
    class_to_ids = defaultdict(list)
    scores = np.zeros(len(results))
    for i, res in enumerate(results):
        class_to_ids[res["label"]].append(i)
        scores[i] = res["confidence"]
        xyxyn = res["xyxyn"]
        y1, y2, x1, x2 = int(
            xyxyn[1]*height), int(xyxyn[3]*height), int(xyxyn[0]*width), int(xyxyn[2]*width)
        boxes[i, :] = [x1, y1, x2, y2]

    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    # We add 1, because the pixel at the start as well as at the end counts
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    res = []
    for label, ids in class_to_ids.items():
        # The indices of all boxes at start. We will redundant indices one by one.
        if len(ids) == 1:
            res.append(ids[0])
            continue

        indices = np.array(sorted(ids, key=lambda i: scores[i]))
        while indices.size > 0:
            index = indices[-1]
            res.append(index)
            box = boxes[index, :]

            # Find out the coordinates of the intersection box
            xx1 = np.maximum(box[0], boxes[indices[:-1], 0])
            yy1 = np.maximum(box[1], boxes[indices[:-1], 1])
            xx2 = np.minimum(box[2], boxes[indices[:-1], 2])
            yy2 = np.minimum(box[3], boxes[indices[:-1], 3])
            # Find out the width and the height of the intersection box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            intersection = w * h

            # compute the ratio of overlap
            ratio = intersection / areas[indices[:-1]]
            # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index
            indices = indices[np.where(ratio < threshold)]

    # return only the boxes at the remaining indices
    return res
