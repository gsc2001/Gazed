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


def get_n_shots(shots, n):
    return [shot for shot in shots if shot.count("-") == n]


def draw_rect(rect, frame, color, thickness=2):
    cv2.rectangle(
        frame,
        (int(rect[0]), int(rect[1])),
        (int(rect[2]), int(rect[3])),
        color,
        thickness,
    )


def rect_centre(rectangle_coords):
    return np.array([rectangle_coords[0] + rectangle_coords[2],rectangle_coords[1] + rectangle_coords[3]])/ 2

def actors_in(shot_name:str):
    actors = set(shot_name.split('-')[:-1])
    return actors

def contained_actors(big_shot, small_shots):
    filtered_shots = []
    for shot in small_shots:
        if actors_in(big_shot).union(actors_in(shot)) == actors_in(big_shot):
            filtered_shots.append(shot)
    return filtered_shots

def join_unary_costs(cost_shot_a, cost_shot_b):
    return cost_shot_a + cost_shot_b - abs(cost_shot_b - cost_shot_a)

def get_rect_area(rect):
    return max(0, (rect[2] - rect[0])) * max(0,(rect[3] - rect[1]))

def crop_image(image, rect):
    return image[int(rect[1]):int(rect[3]), int(rect[0]):int(rect[2])]