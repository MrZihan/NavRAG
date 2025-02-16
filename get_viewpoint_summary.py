import os
import sys
import argparse
import requests
import json
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


def chat_with_gpt(prompt, view_data):
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
                        "text": prompt+view_data
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
    
    
def process_features(proc_id, out_queue, view_annotation, args, prompt):
    print('start proc_id: %d' % proc_id)
    
    for view_data in view_annotation:
        scene_id, viewpoint_id, view_dict = view_data
        response = chat_with_gpt(prompt,str(view_dict))
        response = response[1:]
        response = response[:-1]
        viewpoint_data = response

        out_queue.put((scene_id, viewpoint_id, viewpoint_data))

    out_queue.put(None)


def build_output_file(args, view_annotation, prompt):
    
    input_data = []
    viewpoint_annotation = {}
    view_direction = ['forward', 'front-right', 'back-right', 'backward', 'back-left', 'front-left']
    for scan_id in view_annotation:
        for viewpoint_id in view_annotation[scan_id]:
            view_dict = {}
            for view_id in range(len(view_direction)):
                view_dict[view_direction[view_id]] = view_annotation[scan_id][viewpoint_id][view_id]

            input_data.append([scan_id,viewpoint_id,view_dict])

    num_workers = min(args.num_workers, len(input_data))
    num_data_per_worker = len(input_data) // num_workers

    out_queue = mp.Queue()
    processes = []
    for proc_id in range(num_workers):
        sidx = proc_id * num_data_per_worker
        eidx = None if proc_id == num_workers - 1 else sidx + num_data_per_worker

        process = mp.Process(
            target=process_features,
            args=(proc_id, out_queue, input_data[sidx: eidx], args, prompt)
        )
        process.start()
        processes.append(process)
    
    num_finished_workers = 0
    num_finished_vps = 0

    progress_bar = ProgressBar(max_value=len(input_data))
    progress_bar.start()


    while num_finished_workers < num_workers:
        res = out_queue.get()
        if res is None:
            num_finished_workers += 1
        else:
            scene_id, viewpoint_id, viewpoint_data = res
            
            if scene_id not in viewpoint_annotation:
                viewpoint_annotation[scene_id] = {}

            viewpoint_annotation[scene_id][viewpoint_id] = viewpoint_data

            num_finished_vps += 1
            progress_bar.update(num_finished_vps)


    progress_bar.finish()
    for process in processes:
        process.join()

    with open("mp3d_viewpoint_annotation.json", "w") as file:
        json.dump(viewpoint_annotation, file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=8) 
    args = parser.parse_args()
    with open("viewpoint_prompt.txt","r",encoding="utf-8") as f:
        prompt = f.read()
    view_annotation = json.load(open('mp3d_view_annotation.json','r'))

    build_output_file(args, view_annotation, prompt)