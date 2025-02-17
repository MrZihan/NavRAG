import os
import sys
import argparse
import numpy as np
import json
import math
import h5py
from PIL import Image
import cv2
from progressbar import ProgressBar

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.multiprocessing import Pool, Process, set_start_method
try:
     set_start_method('spawn')
except RuntimeError:
    pass
sys.path.append('.')

import MatterSim
import habitat
from utils.habitat_utils import HabitatUtils
from scipy.spatial.transform import Rotation as R

import json
from habitat import get_config
from habitat.sims import make_sim
from habitat.utils.visualizations import maps
import matplotlib.pyplot as plt

import requests
import base64
import cv2

def encode_image(image):
    retval, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')


def chat_with_gpt(prompt,image):
    img_type = "image/jpeg"  
    img_b64_str = encode_image(image)
    openai_key = json.load(open("openai_key.json","r"))
    api_key = openai_key["api_key"]
    model = openai_key["model"]
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{img_type};base64,{img_b64_str}"},
                    }
                ]
            },
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            } 
        ]
    }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']
 
    else:
        return f"Error: {response.status_code}, {response.text}"
    

class HabitatUtils:
    def __init__(self, scene, level, hfov, h, w, housetype='hm3d'):
        # -- scene = data/hm3d/house/house.glb
        self.scene = scene
        self.level = level  # -- int
        self.house = scene.split('/')[-2]
        self.housetype = housetype

        #-- setup config
        self.config = get_config()
        self.config.defrost()
        self.config.SIMULATOR.SCENE = scene
        self.config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR",
                                                 "DEPTH_SENSOR",
                                                 "SEMANTIC_SENSOR"]
        self.config.SIMULATOR.RGB_SENSOR.HFOV = hfov
        self.config.SIMULATOR.RGB_SENSOR.HEIGHT = h
        self.config.SIMULATOR.RGB_SENSOR.WIDTH = w
        self.config.SIMULATOR.DEPTH_SENSOR.HFOV = hfov
        self.config.SIMULATOR.DEPTH_SENSOR.HEIGHT = h
        self.config.SIMULATOR.DEPTH_SENSOR.WIDTH = w
        self.config.SIMULATOR.SEMANTIC_SENSOR.HFOV = hfov
        self.config.SIMULATOR.SEMANTIC_SENSOR.HEIGHT = h
        self.config.SIMULATOR.SEMANTIC_SENSOR.WIDTH = w
        # self.config.SIMULATOR.AGENT_0.HEIGHT = 0

        # -- Original resolution
        self.config.SIMULATOR.FORWARD_STEP_SIZE = 0.1
        self.config.SIMULATOR.TURN_ANGLE = 9

        # -- fine resolution setps
        #self.config.SIMULATOR.FORWARD_STEP_SIZE = 0.05
        #self.config.SIMULATOR.TURN_ANGLE = 3

        # -- render High Rez images
        #self.config.SIMULATOR.RGB_SENSOR.HEIGHT = 720
        #self.config.SIMULATOR.RGB_SENSOR.WIDTH = 1280

        # -- LOOK DOWN
        #theta = 30 * np.pi / 180
        #self.config.SIMULATOR.RGB_SENSOR.ORIENTATION = [-theta, 0.0, 0.0]
        #self.config.SIMULATOR.DEPTH_SENSOR.ORIENTATION = [-theta, 0.0, 0.0]
        #self.config.SIMULATOR.SEMANTIC_SENSOR.ORIENTATION = [-theta, 0.0, 0.0]

        # -- OUTDATED (might be able to re-instantiate those in future commits)
        #self.config.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD",
        #                                     "TURN_LEFT", "TURN_RIGHT",
        #                                     "LOOK_UP", "LOOK_DOWN"]

        # -- ObjNav settings
        #self.config.SIMULATOR.FORWARD_STEP_SIZE = 0.25
        #self.config.SIMULATOR.TURN_ANGLE = 30


        self.config.freeze()

        self.agent_height = self.config.SIMULATOR.AGENT_0.HEIGHT

        self.sim = make_sim(id_sim=self.config.SIMULATOR.TYPE, config=self.config.SIMULATOR)

        self.semantic_annotations = self.sim.semantic_annotations()

        self.sim.reset()

        agent_state = self.get_agent_state()
        self.position = agent_state.position
        self.rotation = agent_state.rotation

        # -- get level dimensions
        # -- read it directly from the saved data from the .house files
        # Tries to set the agent on the given floor. It's actually quite hard..
        # if housetype == 'hm3d':
        #     env = '_'.join([self.house, str(self.level)])
        #     houses_dim = json.load(open('data/houses_dim.json', 'r'))
        #     self.center = np.array(houses_dim[env]['center'])
        #     self.sizes = np.array(houses_dim[env]['sizes'])
        #     self.start_height = self.center[1] - self.sizes[1]/2

        #     self.set_agent_on_level()
        # else:
        #     pass

        self.all_objects = self.get_objects_in_house()

    @property
    def position(self):
        return self._position


    @position.setter
    def position(self, p):
        self._position = p


    @property
    def rotation(self):
        return self._rotation


    @rotation.setter
    def rotation(self, r):
        self._rotation = r


    def set_agent_state(self):
        self.sim.set_agent_state(self._position,
                                 self._rotation)

    def get_agent_state(self):
        return self.sim.get_agent_state()


    def get_sensor_pos(self):
        ags = self.sim.get_agent_state()
        return ags.sensor_states['rgb'].position

    def get_sensor_ori(self):
        ags = self.sim.get_agent_state()
        return ags.sensor_states['rgb'].rotation



    def reset(self):
        self.sim.reset()
        agent_state = self.get_agent_state()
        self.position = agent_state.position
        self.rotation = agent_state.rotation

    def set_agent_on_level(self):
        """
        It is very hard to know exactly the z value of a level as levels can
        have stairs and difference in elevation etc..
        We use the level.aabb to get an idea of the z-value of the level but
        that isn't very robust (eg r1Q1Z4BcV1o_0 -> start_height of floor 0:
            -1.3 but all sampled point will have a z-value around 0.07, when
            manually naivagting in the env we can see a pth going downstairs..)
        """
        point = self.sample_navigable_point()
        self.position = point
        self.set_agent_state()

    def step(self, action):
        self.sim.step(action)


    def sample_navigable_point(self):
        """
        If house has only one level we sample directly a nav point
        Else we iter until we get a point on the right floor..
        """
        if len(self.semantic_annotations.levels) == 1:
            return self.sim.sample_navigable_point()
        else:
            for _ in range(1000):
                point = self.sim.sample_navigable_point()
                #return point
                if np.abs(self.start_height - point[1]) <= 1.5:
                #if np.all(((self.center-self.sizes/2)<=point) &
                #          ((self.center+self.sizes/2)>=point)):
                    return point
            print('No navigable point on this floor')
            return None


    def sample_rotation(self):
        theta = np.random.uniform(high=np.pi)
        quat = np.array([0, np.cos(theta/2), 0, np.sin(theta/2)])
        return quat



    def get_house_dimensions(self):
        return self.semantic_annotations.aabb



    def get_objects_in_scene(self):
        """

            returns dict with {int obj_id: #pixels in frame}

        """
        buf = self.sim.render(mode="semantic")
        unique, counts = np.unique(buf, return_counts=True)
        objects = {int(u): c for u, c in zip(unique, counts)}
        return objects


    def render(self, mode='rgb'):
        return self.sim.render(mode=mode)



    def render_semantic_mpcat40(self):
        buf = self.sim.render(mode="semantic")
        out = np.zeros(buf.shape, dtype=np.uint8) # class 0 -> void
        object_ids = np.unique(buf)
        for oid in object_ids:
            object = self.all_objects[oid]
            # -- mpcat40_name = object.category.name(mapping='mpcat40')
            mpcat40_index = object.category.index(mapping='mpcat40')
            # remap everything void/unlabeled/misc/etc .. to misc class 40
            # (void:0,  unlabeled: 41, misc=40)
            if mpcat40_index <= 0 or mpcat40_index > 40: mpcat40_index = 40 # remap -1 to misc
            out[buf==oid] = mpcat40_index
        return out





    def get_objects_in_level(self):
        # /!\ /!\ level IDs are noisy in HM3D
        # /!\ /!\

        if self.housetype == 'hm3d':

            assert self.level == int(self.semantic_annotations.levels[self.level].id)

            objects = {}
            for region in self.semantic_annotations.levels[self.level].regions:
                for object in region.objects:
                    objects[int(object.id.split('_')[-1])] = object
        else:
            objects = self.all_objects

        return objects


    def get_objects_in_house(self):
        objects = {int(o.id.split('_')[-1]): o for o in self.semantic_annotations.objects if o is not None}
        return objects


    def __del__(self):
        self.sim.close()


def load_viewpoint_ids(connectivity_dir):
    viewpoint_ids = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids.extend([(scan, x['image_id']) for x in data if x['included']])
    print('Loaded %d viewpoints' % len(viewpoint_ids))
    return viewpoint_ids


VIEWPOINT_SIZE = 12 # Number of discretized views from one viewpoint

WIDTH = 480
HEIGHT = 480
VFOV = 90
HFOV = 90


def build_simulator(connectivity_dir):
    sim = MatterSim.Simulator()
    sim.setNavGraphPath(connectivity_dir)
    sim.setRenderingEnabled(False)
    sim.setCameraResolution(WIDTH, HEIGHT)
    sim.setCameraVFOV(math.radians(VFOV))
    sim.setDiscretizedViewingAngles(True)
    sim.setDepthEnabled(False)
    sim.setPreloadingEnabled(False)
    sim.setBatchSize(1)
    sim.initialize()
    return sim

def build_habitat_sim(scan):
    sim = HabitatUtils(f'data/scene_datasets/hm3d/train/{scan}/'+scan.split('-')[-1]+'.basis.glb', int(0), int(math.degrees(HFOV)), HEIGHT, WIDTH)
    return sim


def process_features(proc_id, out_queue, scanvp_list, args, prompt, hm3d_annotation):
    print('start proc_id: %d' % proc_id)

    # Set up the simulator
    sim = build_simulator(args.connectivity_dir)
    habitat_sim = None
    pre_scan_id = None

    # Set up PyTorch CNN model
    torch.set_grad_enabled(False)

    for scan_id, viewpoint_id in scanvp_list:
        if scan_id in hm3d_annotation and viewpoint_id in hm3d_annotation[scan_id] and len(hm3d_annotation[scan_id][viewpoint_id])==6: # only store 6 views
            continue

        if scan_id != pre_scan_id:
            if habitat_sim != None:
                habitat_sim.sim.close()
            habitat_sim = build_habitat_sim(scan_id)
        pre_scan_id = scan_id

        # Loop all discretized views from this location
        images = []
        depths = []
        scene_annotation = {}
        for ix in range(VIEWPOINT_SIZE):
            if ix == 0:
                sim.newEpisode([scan_id], [viewpoint_id], [0], [0])
            else:
                sim.makeAction([0], [1.0], [0])
            state = sim.getState()[0]
            assert state.viewIndex == ix + 12

            # set habitat to the same position & rotation
            x, y, z, h, e = state.location.x, state.location.y, state.location.z, state.heading, state.elevation
            habitat_position = [x, z, -y]
            hm3d_h = np.array([0, 2*math.pi-h, 0]) # counter-clock heading
            hm3d_e = np.array([e, 0, 0])
            rotvec_h = R.from_rotvec(hm3d_h)
            rotvec_e = R.from_rotvec(hm3d_e)
            habitat_rotation = (rotvec_h * rotvec_e).as_quat()
            habitat_sim.sim.set_agent_state(habitat_position, habitat_rotation)

            image = np.array(habitat_sim.render('rgb'), copy=True)  # in RGB channel
            depth = np.array(habitat_sim.render('depth'), copy=True)  # in depth channel

            images.append(image)
            depths.append(depth)

            if ix % 2 == 0: # only store 6 views
                response = chat_with_gpt(prompt,image)
                #plt.imshow(image)
                #plt.show()
                error_count = 0
                while True:
                    try:
                        view_annotation = eval(response)
                        if scan_id not in scene_annotation:
                            scene_annotation[scan_id] = {}

                        if viewpoint_id not in scene_annotation[scan_id]:
                            scene_annotation[scan_id][viewpoint_id] = []

                        scene_annotation[scan_id][viewpoint_id].append(
                            {'habitat_position':habitat_position,'habitat_rotation':habitat_rotation.tolist(),'view_summary':view_annotation[0],
                                                                        'instance_description': view_annotation[1],'instance_affordance':view_annotation[2]})
                        break
                    except:
                        print("Some errors happen in response of GPT, try again...")
                        response = chat_with_gpt(prompt+' Please double-check the accuracy of the Python dictionary!',image)
                        error_count += 1
                        if error_count > 10:
                            print("Many errors happen in response of GPT, exit...")
                            print("GPT response is:")
                            print(response)
                            exit()


        out_queue.put((scene_annotation, images, depths))

    out_queue.put(None)


def build_output_file(args, prompt):
    
    if os.path.exists("hm3d_view_annotation.json"):
        hm3d_annotation = json.load(open("hm3d_view_annotation.json","r"))
    else:
        hm3d_annotation = {}
    scanvp_list = load_viewpoint_ids(args.connectivity_dir)

    num_workers = min(args.num_workers, len(scanvp_list))
    num_data_per_worker = len(scanvp_list) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scanvp_list[sidx: eidx], args, prompt, hm3d_annotation)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(scanvp_list))
    progress_bar.start()


    while num_finished_workers < num_workers:
        res = out_queue.get()
        if res is None:
            num_finished_workers += 1
        else:
            scene_annotation, images, depths = res
            for scene_id in scene_annotation:
                if scene_id not in hm3d_annotation:
                    hm3d_annotation[scene_id] = {}

                for viewpoint_id in scene_annotation[scene_id]:
                    hm3d_annotation[scene_id][viewpoint_id] = scene_annotation[scene_id][viewpoint_id]

            num_finished_vps += 1
            progress_bar.update(num_finished_vps)

        if num_finished_vps % 1000 == 0:
            with open("hm3d_view_annotation.json", "w") as file:
                json.dump(hm3d_annotation, file)


    progress_bar.finish()
    for process in processes:
        process.join()

    with open("hm3d_view_annotation.json", "w") as file:
        json.dump(hm3d_annotation, file)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', default='connectivity_hm3d')
    parser.add_argument('--num_workers', type=int, default=8) 
    args = parser.parse_args()
    with open("view_prompt.txt","r",encoding="utf-8") as f:
        prompt = f.read()

    build_output_file(args, prompt)