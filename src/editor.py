import yaml
import numpy as np
import pickle as pkl
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
        self.load_cost_params()

        self.timesteps = min([len(val) for val in self.shot_tracks.values()])
        self.timesteps = min(self.timesteps, len(self.frame_paths))
        self.unary_costs = {shot: [0] * self.timesteps for shot in self.shot_names}
        self.shot_duration = {shot: [0] * self.timesteps for shot in self.shot_names}
        self.parent = {shot: [-1] * self.timesteps for shot in self.shot_names}
        self.track = [None] * self.timesteps

    def load_gaze(self):
        with open(self.gaze_file, "r") as f:
            gaze = yaml.safe_load(f)
        self.gazes = {}
        for key in gaze.keys():
            self.gazes[key] = np.array(gaze[key], dtype=np.float64)

    def load_frames(self):
        frames = sorted(glob.glob(os.path.join(self.frames_dir, "*.png")))
        self.frame_paths = frames

    def load_actors_and_shots(self):
        self.actors = eval(self.config_reader.get("video", "actors"))
        shot_tracks = self.config_reader["shots"]
        self.shot_tracks = {}

        for key in shot_tracks.keys():
            self.shot_tracks[key] = np.loadtxt(
                shot_tracks[key], dtype=float, delimiter=" "
            )
        self.shot_names = list(self.shot_tracks.keys())

    def load_config(self, config_file: str) -> None:
        config_parser = ConfigParser(interpolation=ExtendedInterpolation())
        config_parser.read(config_file)
        self.config_reader = config_parser

        self.frames_dir = self.config_reader.get("video", "frames")
        self.gaze_file = self.config_reader.get("video", "gaze")
        self.fps = self.config_reader.getint("video", "fps")
        self.height = self.config_reader.getint("video", "height")
        self.width = self.config_reader.getint("video", "width")
        self.start_time = int(self.fps * self.config_reader.getint("parameters", "start_time"))

        self.gaze_t_offset = int(
            self.fps * self.config_reader.getfloat("parameters", "gazeTOffset")
        )
        self.gaze_x_offset = self.config_reader.getint("parameters", "gazeXOffset")
        self.gaze_y_offset = self.config_reader.getint("parameters", "gazeYOffset")
    
    def load_cost_params(self):
        self.cost_params = {
            'transition_lambda': self.config_reader.getfloat("parameters", "transition_lambda"),
            'overlap_alpha': self.config_reader.getfloat("parameters", "overlap_alpha"),
            'overlap_beta': self.config_reader.getfloat("parameters", "overlap_beta"),
            'overlap_mu': self.config_reader.getfloat("parameters", "overlap_mu"),
            'overlap_nu': self.config_reader.getfloat("parameters", "overlap_nu"),
            'rhythm_lambda_1': self.config_reader.getfloat("parameters", "rhythm_lambda_1"),
            'rhythm_lambda_2': self.config_reader.getfloat("parameters", "rhythm_lambda_2"),
            'rhythm_l': int(self.fps * self.config_reader.getfloat("parameters", "rhythm_l")),
            'rhythm_m': int(self.fps * self.config_reader.getfloat("parameters", "rhythm_m"))
        }

    def edit(self):
        self.calc_unary_costs()
        self.costs = {shot: [0] * self.timesteps for shot in self.shot_names}
        # create dp array with backtracking
        for i in range(self.start_time, self.timesteps):
            for shot in self.shot_names:
                self.costs[shot][i] = self.unary_costs[shot][i]
                if i > 0:
                    min_cost = np.inf
                    min_cost_prev_shot = -1
                    for prev_shot in self.shot_names:
                        cost = self.costs[prev_shot][i - 1] + self.shift_cost(prev_shot, shot, i)

                        if cost < min_cost:
                            min_cost = cost
                            min_cost_prev_shot = prev_shot
                    assert self.costs[shot][i] != np.inf
                    self.costs[shot][i] += min_cost
                    self.parent[shot][i] = min_cost_prev_shot

                    if self.parent[shot][i] == shot: self.shot_duration[shot][i] = self.shot_duration[shot][i-1] + 1
                    else: self.shot_duration[shot][i] = 0


        # choose best path
        min_cost = np.inf
        min_cost_shot = -1
        for shot in self.shot_names:
            if self.costs[shot][self.timesteps - 1] < min_cost:
                min_cost = self.costs[shot][self.timesteps - 1]
                min_cost_shot = shot
        
        # backtrack
        for time in range(self.timesteps - 1, self.start_time - 1, -1):
            self.track[time] = min_cost_shot
            min_cost_shot = self.parent[min_cost_shot][time]
        
        for i in range(0, self.start_time): self.track[i] = self.track[self.start_time + 1]


        with open('cost_matrix.pkl', 'wb') as f:
            pkl.dump(self.costs, f)

        print('Done editing')

    

    def __find_unary_costs(self, timestep):
        one_shots = get_n_shots(self.shot_names, 1)
        costs = {}

        for shot in one_shots:
            centre = rect_centre(self.shot_tracks[shot][timestep])
            cost = 0.00001

            if centre[0] > 25 and centre[1] > 25:
                for key in sorted(self.gazes.keys()):
                    gaze = self.gazes[key][timestep + self.gaze_t_offset][:2] + np.array([self.gaze_x_offset, self.gaze_y_offset])
                    if gaze[0] > self.width or gaze[1] > self.height:
                        continue
                    cost += np.linalg.norm(centre - gaze)
                
                if cost != 0.00001:
                    costs[shot] = 1 / cost
                else:
                    costs[shot] = self.unary_costs[shot][timestep - 1]
            else:
                costs[shot] = cost
        # if timestep == 0: print(costs)
        
        sum_costs = sum(costs.values())

        for shot in one_shots: costs[shot] = costs[shot] / sum_costs

        for n in range(2, len(self.actors) + 1):
            n_shots = get_n_shots(self.shot_names, n)
            n_1_shots = get_n_shots(self.shot_names, n - 1)

            for shot in n_shots:
                filtered_shots = contained_actors(shot, n_1_shots)
                # sort on basis of x
                filtered_shots.sort(key= lambda shot: rect_centre(self.shot_tracks[shot][timestep])[0])
                # TODO: experiment with sorted one_shot_x

                left_one_shot = list((actors_in(shot) - actors_in(filtered_shots[1])))[0] + '-ms'
                right_one_shot = list((actors_in(shot) - actors_in(filtered_shots[0])))[0] + '-ms'

                left_cost = join_unary_costs(costs[left_one_shot], costs[filtered_shots[1]])
                right_cost = join_unary_costs(costs[filtered_shots[0]], costs[right_one_shot])

                costs[shot] = max(left_cost, right_cost)
        
        return costs
    
    def overlap_cost(self, prev_shot, cur_shot, timestep):
        frame1 = self.shot_tracks[prev_shot][timestep - 1]
        frame2 = self.shot_tracks[cur_shot][timestep]

        if get_rect_area(frame1) < 100 or get_rect_area(frame2) < 100:
            iou = 1
        else:
            # find intersection over union of these 2 rectangles
            min_x = max(frame1[0], frame2[0])
            min_y = max(frame1[1], frame2[1])
            max_x = min(frame1[2], frame2[2])
            max_y = min(frame1[3], frame2[3])

            intersection = get_rect_area([min_x, min_y, max_x, max_y])
            union = get_rect_area(frame1) + get_rect_area(frame2) - intersection

            iou = intersection / union
        
        # find cost
        if iou <= self.cost_params['overlap_alpha']:
            cost = 0
        elif iou >= self.cost_params['overlap_beta']:
            slope = self.cost_params['overlap_mu'] / (self.cost_params['overlap_beta'] -  self.cost_params['overlap_alpha'])
            cost = iou * slope - self.cost_params['overlap_alpha'] * slope
        else:
            cost = self.cost_params['overlap_nu'] 
        
        return cost



    def calc_unary_costs(self):
        for i in range(self.timesteps):
            costs = self.__find_unary_costs(i)
            for shot in self.shot_names:
                assert shot in costs
                self.unary_costs[shot][i] = -costs[shot]
    
    
    def shift_cost(self, prev_shot, cur_shot, timestep):
        shift_cost = 0

        # transition cost
        if prev_shot != cur_shot:
            transition_cost = self.cost_params['transition_lambda']
            # print('transition', transition_cost)
            shift_cost += transition_cost

        
        # overlap cost
        if prev_shot != cur_shot:
            overlap_cost = self.overlap_cost(prev_shot, cur_shot, timestep)
            # print('overlap', overlap_cost)
            shift_cost += overlap_cost
        
        # rythm cost
        if prev_shot == cur_shot:
            rhythm_cost = self.cost_params['rhythm_lambda_2'] * (1 - 1 / (1 + np.exp(self.shot_duration[prev_shot][timestep - 1] - self.cost_params['rhythm_m'])))
            # print('rhythm_same', rhythm_cost)
            shift_cost += rhythm_cost
        else:
            rhythm_cost = self.cost_params['rhythm_lambda_1'] * (1 - 1 / (1 + np.exp(-self.shot_duration[prev_shot][timestep-1] + self.cost_params['rhythm_l'])))
            # print('rhythm_diff', rhythm_cost)
            shift_cost += rhythm_cost
        
        return shift_cost


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
