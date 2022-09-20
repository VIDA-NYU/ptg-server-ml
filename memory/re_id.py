"""
Author Jianzhe Lin
May.2, 2020
"""
from dataclasses import dataclass
from collections import Counter
import numpy as np
import cv2
import matplotlib.pyplot as plt
    
@dataclass
class MemoryItem:
    xyz_center: np.ndarray
    label: str
    track_id: str
    seen_before: bool = False
    

class ReId:
    def __init__(self) -> None:
        self.memory = {}
        self.unseen_count = {}
        self.instance_count = Counter()
        
    def update_frame(self, objects):
        # get all objects whose seen_before = False
        unseens = set(self.unseen_count.keys())
        
        # update objects
        for obj in objects:
            track_id = self.update_object(obj)
            # remove it from unseens if we see the track_id
            unseens.discard(track_id)
        
        # update unseen frame count and change seen_before to True if unseen_count becomes 0
        for track_id in unseens:
            self.unseen_count[track_id] -= 1
            if self.unseen_count[track_id] == 0:
                self.memory[track_id].seen_before = True
                del self.unseen_count[track_id]
        
    def update_object(self, obj):
        xyz_center = np.asarray(obj['xyz_center'])
        label = obj['label']
        # check memory
        for item in self.memory.values():
            if label == item.track_id.rsplit('_', 1)[0] and self.memory_comparison(xyz_center, item.xyz_center):
                return item.track_id
        
        # unique name for multiple instances
        self.instance_count[label] += 1
        i = self.instance_count[label]
        track_id = f'{label}_{i}'
        
        # add new item to memory
        self.memory[track_id] = MemoryItem(xyz_center = xyz_center, label = label, track_id = track_id)
        self.unseen_count[track_id] = 10 # change seen_before to True if unseen for 10 frames
        return track_id

    def memory_comparison(self, seen, candidate):
        '''Compare a new instance to a previous one. Determine if they match.'''
        return np.linalg.norm(candidate - seen) < 0.4
    
    def dump_memory(self):
        return list(self.memory.values())


class DrawResults:
    def __init__(self, memory):
        self.location_memory = memory

    def draw_4panel(self, results, rgb):
        res = results.imgs[0]
        return np.vstack([
            np.hstack([
                self.draw_memory_yolo(results, rgb), 
                self.draw_basic_yolo(results),
            ]), 
            np.hstack([
                self.draw_message_board(res.shape), 
                self.draw_3d_space(res.shape),
            ]),
        ])

    def draw_memory_yolo(self, results, rgb):
#        img = results.imgs[0]
        for b, seen in zip(results.xywh[0], results.seen_before):
            rgb = draw_bbox(
                rgb.copy(), *b[:4], 
                color=(255, 0, 0) if seen else (0, 255, 0))
        draw_text_list(rgb, [
            f'hey I remember {name}'
            for name, seen in zip(results.track_ids, results.seen_before)
            if seen
        ])
        return rgb.copy()


    def draw_basic_yolo(self, results):
        return results.render()[0]

    def draw_message_board(self, shape):
        img = np.ones(shape, np.uint8) * 255
        img = draw_text_list(img, [
            f"{name}: [{', '.join(f'{x:.3f}' for x in loc)}]"
            for name, loc in self.location_memory.items()
        ])
        return img

    def draw_3d_space(self, shape):
        fig = plt.figure()
        ax = plt.axes(projection='3d')

        # ax.set_xlim((-2, 0))
        # ax.set_ylim((-2, 0))
        # ax.set_zlim((-2, 0))

        for name, loc in self.location_memory.items():    
            ax.scatter(*loc, marker='^')   
            ax.text(*loc, name, fontsize=6)

        fig.canvas.draw()
        src = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        src = src.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        src = cv2.resize(src, shape[:2][::-1])
        plt.close()
        return src

# drawing

def draw_text_list(img, texts):
    for i, txt in enumerate(texts):
        cv2.putText(img, txt, (20+400*(i//12), 40+30*(i%12)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 48, 48), 2)
    return img


def draw_bbox(img, xc, yc, w, h, *, color=(0, 255, 0)):
    img = cv2.rectangle(
        img, 
        (int(xc - w/2), int(yc - h/2)), 
        (int(xc + w/2), int(yc + h/2)), 
        color, 2)
    return img
