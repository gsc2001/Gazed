import yaml
import numpy as np
import glob
import cv2
import os
from utils import *
from configparser import ConfigParser, ExtendedInterpolation


class Editor:
    def __init__(self, config_file: str) -> None:
        self.load_config(config_file)
        self.load_frames()
        self.load_gaze()
        self.load_actors_and_shots()

        self.timesteps = min([len(val) for val in self.shot_tracks.values()])
        self.timesteps = min(self.timesteps, len(self.frame_paths))

    def load_gaze(self):
        with open(self.gaze_file, "r") as f:
            gaze = yaml.safe_load(f)

        for key in gaze.keys():
            gaze[key] = np.array(gaze[key], dtype=np.float64)
        self.gazes = gaze

    def load_frames(self):
        frames = sorted(glob.glob(os.path.join(self.frames_dir, "*.png")))
        self.frame_paths = frames

    def load_actors_and_shots(self):
        self.actors = eval(self.config_reader.get("video", "actors"))
        self.shot_names = generate_shot_names(self.actors)
        shot_tracks = self.config_reader["shots"]
        self.shot_tracks = {}

        for key in shot_tracks.keys():
            self.shot_tracks[key] = np.loadtxt(
                shot_tracks[key], dtype=float, delimiter=" "
            )

    def load_config(self, config_file: str) -> None:
        config_parser = ConfigParser(interpolation=ExtendedInterpolation())
        config_parser.read(config_file)
        self.config_reader = config_parser

        self.frames_dir = self.config_reader.get("video", "frames")
        self.gaze_file = self.config_reader.get("video", "gaze")
        self.fps = self.config_reader.getint("video", "fps")

        self.gaze_t_offset = int(
            self.fps * self.config_reader.getfloat("parameters", "gazeTOffset")
        )
        self.gaze_x_offset = self.config_reader.getint("parameters", "gazeXOffset")
        self.gaze_y_offset = self.config_reader.getint("parameters", "gazeYOffset")

    def debug(self):
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0)]
        for i in range(self.timesteps):
            frame = cv2.imread(self.frame_paths[i])
            for key in self.gazes.keys():
                gaze = self.gazes[key][i + self.gaze_t_offset]
                cv2.circle(
                    frame,
                    (
                        int(gaze[0] + self.gaze_x_offset),
                        int(gaze[1] + self.gaze_y_offset),
                    ),
                    10,
                    (0, 0, 255),
                    -1,
                )
            for shot_i, key in enumerate(self.shot_tracks.keys()):
                shot_track = self.shot_tracks[key][i]
                draw_rect(shot_track, frame, color=colors[shot_i])
            cv2.imshow("frame", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()
        cv2.waitKey(1)
