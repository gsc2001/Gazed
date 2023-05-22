import numpy as np
import cv2


def generate_shot_names(actors):
    shot_names = []
    for i in range(1, 2 ** len(actors)):
        shot_name = ""
        for j in range(len(actors)):
            if i & (1 << j):
                shot_name += actors[j]
                shot_name += "-"

        if shot_name.count("-") > 1:
            shot_name += "fs"
        else:
            shot_name += "ms"

        shot_names.append(shot_name)
    return shot_names


def draw_rect(rect, frame, color, thickness=2):
    cv2.rectangle(
        frame,
        (int(rect[0]), int(rect[1])),
        (int(rect[2]), int(rect[3])),
        color,
        thickness,
    )
