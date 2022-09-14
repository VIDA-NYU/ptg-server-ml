"""
Author Jianzhe Lin
May.2, 2020
"""
from collections import defaultdict
import numpy as np
import cv2
import matplotlib.pyplot as plt


class ReId:
    MIN_DEPTH_POINT_DISTANCE = 7

    def __init__(self) -> None:
        self.location_memory = {}
        self.instance_count = defaultdict(lambda: 0)

    def update_memory(self, xyz, label):
        # check memory
        for k, xyz_seen in self.location_memory.items():
            if label == k and self.memory_comparison(xyz_seen, xyz):
                return k, True
            elif label == k[:-2] and self.memory_comparison(xyz_seen, xyz):
                return k, True
        # unique name for multiple instances
        if label in self.location_memory:
            self.instance_count[label] += 1
            i = self.instance_count[label]
            label = f'{label}_{i}'
        
        # TODO: add other info
        self.location_memory[label] = xyz
        return label, False

    def memory_comparison(self, seen, candidate):
        '''Compare a new instance to a previous one. Determine if they match.'''
        return np.linalg.norm(candidate - seen) < 1


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
