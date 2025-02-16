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


def chat_with_gpt(prompt):
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
    
    
# Input the a node set and the connectivity matrix, return the node set that connected to the given node set
def get_connected_nodes(current_node_set, connectivity_matrix):
    node_id_list = np.array([i for i in range(connectivity_matrix.shape[0])])
    connected_node_set = set(current_node_set)
    current_node_set = set(current_node_set)
    for node_id in current_node_set:
        node_connectivity = connectivity_matrix[node_id]
        connected_node_set = set(node_id_list[node_connectivity==1].tolist()) | connected_node_set

    connected_node_set = connected_node_set - current_node_set
    return connected_node_set


def zones_are_connected(zone_1,zone_2,connectivity_matrix):
    is_connected = False
    for node_id_1 in zone_1:
        for node_id_2 in zone_2:
            if connectivity_matrix[node_id_1,node_id_2] == 1:
                is_connected = True
    return is_connected


def load_connectivity_matrix(connectivity_dir):
    connectivity_annotation = []
    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json'%scan)) as f:
            data = json.load(f)
            viewpoint_ids = [x['image_id'] for x in data if x['included']]
            connectivity_bool_matrix = np.array([x['unobstructed'] for x in data if x['included']])
            not_included_ids = [i for i in range(len(data)) if not data[i]['included']]
            not_included_ids.reverse()
            for i in not_included_ids:
                connectivity_bool_matrix = np.concatenate([connectivity_bool_matrix[:,:i], connectivity_bool_matrix[:,i+1:]],axis=-1)
            connectivity_annotation.append([scan,viewpoint_ids,connectivity_bool_matrix])

    return connectivity_annotation



def process_features(proc_id, out_queue, connectivity_annotation, viewpoint_annotation, args, construct_prompt, summary_prompt):
    print('start proc_id: %d' % proc_id)
    
    zone_annotation = {}
    for scan_id, viewpoint_list, connectivity_bool_matrix in connectivity_annotation:

        zone_annotation[scan_id] = []

        connectivity_matrix = np.ones(connectivity_bool_matrix.shape)
        connectivity_matrix[connectivity_bool_matrix==False] = 0

        zone_list = []
        zone_node_set = set()
        waiting_node_set = set([node_id for node_id in range(connectivity_matrix.shape[0])])
        not_zone_node_set = set()

        # Loop for zones construction
        while True:
            # Calculate the connectivity of the nodes, search the node with max connectivity
            if len(zone_node_set) == 0:
                connectivity_value = connectivity_matrix.sum(-1)
                max_connectivity_node = connectivity_value.argmax(axis=0)

                zone_node_set = zone_node_set | set([max_connectivity_node])
                waiting_node_set = waiting_node_set - set([max_connectivity_node])

                not_zone_node_set = set()
            
            zone_node_ids = [node_id for node_id in zone_node_set]
            connected_node_set = list(get_connected_nodes(zone_node_set, connectivity_matrix) - not_zone_node_set)

            if len(connected_node_set) == 0: # number of nodes connected the zone is 0
                zone_list.append(zone_node_set)
                if len(waiting_node_set) == 0:
                    break

                zone_node_set = list(zone_node_set)
                zone_node_set.sort(reverse=True)
                for node_id in zone_node_set:
                    connectivity_matrix[:,node_id] = -1
                    connectivity_matrix[node_id,:] = -1                  

                zone_node_set = set()
                continue

            connected_node_set.sort()
            connected_node_ids = [node_id for node_id in connected_node_set]

            zone_node_annotations = {}
            for node_id in zone_node_ids:
                zone_node_annotations[node_id] = viewpoint_annotation[scan_id][viewpoint_list[node_id]].split('.')[0]+'.'

            connected_node_annotations = {}
            for node_id in connected_node_ids:
                connected_node_annotations[node_id] = viewpoint_annotation[scan_id][viewpoint_list[node_id]].split('.')[0]+'.'

            error_count = 0
            while True:
                try:
                    response = chat_with_gpt(construct_prompt+'\nThe Python dictionary of recognized viewpoints within the zone is as follows:\n'+str(zone_node_annotations)+'\nThe Python dictionary of unrecognized viewpoints is as follows:\n'+str(connected_node_annotations)+"\n Please carefully check the viewpoint_id.")
                    response = eval(response)
                    break
                except:
                    print("Some errors happen, try again...")
                    error_count += 1
                    if error_count > 5:
                        print("Many errors happen, exit...")
                        exit()

            for node_id in response['zone']:
                zone_node_set = zone_node_set | set([node_id])
                waiting_node_set = waiting_node_set - set([node_id])
            for node_id in response['not zone']:
                not_zone_node_set = not_zone_node_set | set([node_id])



        connectivity_matrix = np.ones(connectivity_bool_matrix.shape)
        connectivity_matrix[connectivity_bool_matrix==False] = 0
        zone_connectivity_matrix = np.zeros((len(zone_list),len(zone_list)),dtype=np.int32)
        for i in range(len(zone_list)):
            for j in range(len(zone_list)):
                if i != j and zones_are_connected(zone_list[i],zone_list[j],connectivity_matrix):
                    zone_connectivity_matrix[i,j] = 1

        
        zone_annotation[scan_id].append([zone_connectivity_matrix[i].tolist() for i in range(zone_connectivity_matrix.shape[0])])
        zone_annotation[scan_id].append({})

        zone_id = 0
        for zone in zone_list:
            zone = list(zone)
            zone.sort()
            viewpoints_in_zone = []
            for node_id in zone:
                viewpoints_in_zone.append(viewpoint_list[node_id])

            text_of_viewpoints = {}
            for viewpoint_id in viewpoints_in_zone:
                text_of_viewpoints[viewpoint_id] = viewpoint_annotation[scan_id][viewpoint_id]
            
            response = chat_with_gpt(summary_prompt+'\n'+str(text_of_viewpoints))
            if "\'" in response or "\"" in response:
                 response = response.replace("\'","").replace("\"","")

            zone_id += 1
            zone_annotation[scan_id][1]['zone_'+str(zone_id)] = [viewpoints_in_zone,response]

        out_queue.put(zone_annotation)

    out_queue.put(None)


def build_output_file(args, viewpoint_annotation, construct_prompt, summary_prompt):

    connectivity_annotation = load_connectivity_matrix(args.connectivity_dir)

    num_workers = min(args.num_workers, len(connectivity_annotation))
    num_data_per_worker = len(connectivity_annotation) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, connectivity_annotation[sidx: eidx], viewpoint_annotation, args, construct_prompt, summary_prompt)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(connectivity_annotation))
    progress_bar.start()

    zone_annotation = {}
    while num_finished_workers < num_workers:
        res = out_queue.get()
        if res is None:
            num_finished_workers += 1
        else:
            zone_data = res
            for scene_id in zone_data:
                zone_annotation[scene_id] = zone_data[scene_id]

            num_finished_vps += 1
            progress_bar.update(num_finished_vps)


    progress_bar.finish()
    for process in processes:
        process.join()

    with open("mp3d_zone_annotation.json", "w") as file:
        json.dump(zone_annotation, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', default='connectivity_mp3d')
    parser.add_argument('--num_workers', type=int, default=8) 
    args = parser.parse_args()
    with open("construct_zone_prompt.txt","r",encoding="utf-8") as f:
        construct_prompt = f.read()
    with open("zone_summary_prompt.txt","r",encoding="utf-8") as f:
        summary_prompt = f.read()

    viewpoint_annotation = json.load(open('mp3d_viewpoint_annotation.json','r'))

    build_output_file(args, viewpoint_annotation, construct_prompt, summary_prompt)
