import requests
import json
from tqdm import tqdm
import os
import sys
import argparse
import numpy as np
import json
import math
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


def chat_with_gpt(prompt=None, prompt_list=None):
    openai_key = json.load(open("openai_key.json","r"))
    api_key = openai_key["api_key"]
    model = openai_key["model"]
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    if prompt_list is None:
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
    else:
        messages = []
        for i in range(len(prompt_list)):
             messages.append(
                  {
                    "role": prompt_list[i][0],
                    "content": [
                        {
                            "type": "text",
                            "text": prompt_list[i][1]
                        }
                    ]
                }
             )

        data = {
            "model": model,
            "messages": messages
        }
    
    response = requests.post(url, headers=headers, json=data)
    
    if response.status_code == 200:
        response_data = response.json()
        return response_data['choices'][0]['message']['content']



def process_features(proc_id, out_queue, scene_ids, roles, instruction_prompt, house_annotation, zone_annotation, viewpoint_annotation, view_annotation, view_direction_keys, num_of_instructions_for_each_role):
    print('start proc_id: %d' % proc_id)

    for scene_id in scene_ids:
        role_id = -1
        for role in roles:
            role_id += 1
            dataset_annotations = {}

            if scene_id not in dataset_annotations:
                dataset_annotations[scene_id] = []

            prompt_text = ""
            prompt_text += instruction_prompt[0] + '\n'
            prompt_text += str(role) + '\n'
            prompt_text += instruction_prompt[1] + '\n'
            prompt_text += str(house_annotation[scene_id]) + '\n'
            prompt_text += instruction_prompt[2] + '\n'
            prompt_text += instruction_prompt[3] + '\n'
            response = chat_with_gpt(prompt_text)

            prompt_text = [["user",prompt_text],["system", response]] # !!!

            for instruction_id in range(num_of_instructions_for_each_role):

                try:
                    coarse_instruction_record = eval(response.replace("}}","}").replace("’","'").replace("}.","}").replace("'s"," s").replace("s'"," s").replace("'m"," am").replace("`","").replace("don't","do not").replace("''","'"))
                    target_zone = coarse_instruction_record['target zone']

                    viewpoints_of_target_zone = zone_annotation[scene_id][1][target_zone][0]
                    viewpoint_descriptions = {}
                    for viewpoint_index in range(len(viewpoints_of_target_zone)):
                        viewpoint_descriptions[viewpoint_index] = viewpoint_annotation[scene_id][viewpoints_of_target_zone[viewpoint_index]]

                    prompt_text.append(["user", instruction_prompt[5].replace("$ZONE",target_zone) + str(viewpoint_descriptions)]) # !!!

                    response = chat_with_gpt(prompt=None, prompt_list=prompt_text)
                    
                    if "\'" in response or "\"" in response:
                        response = response.replace("'","").replace("\"","").replace("{","").replace("}","")

                    if "_" in response:
                        response = response.split("_")[-1]
                    viewpoint_id = viewpoints_of_target_zone[int(response)]

                    viewpoint_direction = {}
                    for view_id in range(len(view_direction_keys)):
                        viewpoint_direction[view_direction_keys[view_id]] = view_annotation[scene_id][viewpoint_id][view_id]['view_summary']

                    prompt_text.append(["system",response]) # !!!
                    prompt_text.append(["user", instruction_prompt[6].replace("$VIEWPOINT",viewpoint_id) + str(viewpoint_direction)]) # !!!
                    response = chat_with_gpt(prompt=None, prompt_list=prompt_text)

                    if "\'" in response or "\"" in response:
                        response = response.replace("'","").replace("\"","").replace("{","").replace("}","")

                    direction_id = response
                    direction_id = view_direction_keys.index(direction_id)

                    prompt_text = prompt_text[:-3]
                    refine_prompt = instruction_prompt[7].replace("$VIEWPOINT",viewpoint_annotation[scene_id][viewpoint_id])
                    refine_prompt = refine_prompt.replace("$DIRECTION",view_annotation[scene_id][viewpoint_id][direction_id]['view_summary'])
                    refine_prompt = refine_prompt.replace("$INSTANCE",str(view_annotation[scene_id][viewpoint_id][direction_id]['instance_description']))
                    refine_prompt = refine_prompt.replace("$AFFORDANCE",str(view_annotation[scene_id][viewpoint_id][direction_id]['instance_affordance']))
                    prompt_text.append(["user",refine_prompt]) # !!!
                    prompt_text.append(["user",instruction_prompt[8]]) # !!!
                    response = chat_with_gpt(prompt=None, prompt_list=prompt_text).replace("}}","}").replace("’","'").replace("}.","}").replace("'s"," s").replace("s'"," s").replace("'m"," am").replace("`","").replace("don't","do not").replace("''","'")

                    instruction_annotation = eval(response)
                    
                    dataset_annotations[scene_id].append(
                        {
                            "role_id":role_id,
                            "target_viewpoint":viewpoint_id,
                            "habitat_position":view_annotation[scene_id][viewpoint_id][direction_id]['habitat_position'],
                            "habitat_rotation":view_annotation[scene_id][viewpoint_id][direction_id]['habitat_rotation'],
                            "instruction":instruction_annotation
                        }
                        )
                    
                    prompt_text = prompt_text[:-2]
                    prompt_text[-1][1] = str(instruction_annotation)
                    prompt_text.append(["user",instruction_prompt[4]])
                    response = chat_with_gpt(prompt=None, prompt_list=prompt_text)

                    
                except:
                    prompt_text = ""
                    prompt_text += instruction_prompt[0] + '\n'
                    prompt_text += str(role) + '\n'
                    prompt_text += instruction_prompt[1] + '\n'
                    prompt_text += str(house_annotation[scene_id]) + '\n'
                    prompt_text += instruction_prompt[2] + '\n'
                    prompt_text += instruction_prompt[3] + '\n'
                    response = chat_with_gpt(prompt_text)

                    if response == 'None':
                        break

                    prompt_text = [["user",prompt_text],["system", response]] # !!!  
                    
                    
            out_queue.put(dataset_annotations)

    out_queue.put(None)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=8) 

    args = parser.parse_args()
    
    with open("roles_prompt.txt","r",encoding="utf-8") as f:
        roles_prompt = f.read()

    roles = eval(roles_prompt)

    with open("instruction_prompt.txt","r",encoding="utf-8") as f:
            instruction_prompt = f.read()

    house_annotation = json.load(open("mp3d_house_annotation.json","r"))

    zone_annotation = json.load(open("mp3d_zone_annotation.json","r"))

    viewpoint_annotation = json.load(open("mp3d_viewpoint_annotation.json","r"))

    view_annotation = json.load(open("mp3d_view_annotation.json","r"))

    view_direction_keys = ['forward', 'front-right', 'back-right', 'backward', 'back-left', 'front-left']

    instruction_prompt = eval(instruction_prompt)

    dataset_annotations = {}
    if os.path.exists("mp3d_instruction_annotation.json"):
        dataset_annotations = json.load(open("mp3d_instruction_annotation.json", "r"))

    num_of_instructions_for_each_role = 20 ### !!!!!!!!!!!!!!!!!!!!!!!!!!

    scene_ids = []
    for scene_id in list(house_annotation.keys()):
        if scene_id not in dataset_annotations:
            scene_ids.append(scene_id)
        elif scene_id in dataset_annotations and len(dataset_annotations[scene_id]) < num_of_instructions_for_each_role*len(roles)//2:
            scene_ids.append(scene_id)

    num_workers = min(args.num_workers, len(scene_ids))
    num_data_per_worker = len(scene_ids) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, scene_ids[sidx: eidx], roles, instruction_prompt, house_annotation, zone_annotation, viewpoint_annotation, view_annotation, view_direction_keys, num_of_instructions_for_each_role)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_scenes = 0

    annotation_count = 0
    save_count = 0
    while num_finished_workers < num_workers:
        res = out_queue.get()
        if res is None:
            num_finished_workers += 1
        else:
            annotation_data = res
            for scene_id in annotation_data:
                if scene_id not in dataset_annotations:
                    dataset_annotations[scene_id] = []
                
                dataset_annotations[scene_id] = dataset_annotations[scene_id] + annotation_data[scene_id]
                annotation_count += len(annotation_data[scene_id])

            print("The number of annotated samples is", annotation_count)
            num_finished_scenes += 1

            if annotation_count // 1000 == save_count:
                save_count += 1
                with open("mp3d_instruction_annotation.json", "w") as file:
                    json.dump(dataset_annotations, file)

            
    for process in processes:
        process.join()

    with open("mp3d_instruction_annotation.json", "w") as file:
        json.dump(dataset_annotations, file)
