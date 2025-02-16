import os
import json
import networkx as nx
import argparse
import numpy as np
import random
import math
from tqdm import tqdm
import jsonlines

def get_tokenizer():
    from transformers import AutoTokenizer
    cfg_name = 'bert-base-uncased'
    tokenizer = AutoTokenizer.from_pretrained(cfg_name)
    return tokenizer

def create_nav_graphs(connectivity_dir):
    ''' Load connectivity graph for each scan '''

    def distance(pose1, pose2):
        ''' Euclidean distance between two graph poses '''
        return ((pose1['pose'][3]-pose2['pose'][3])**2\
          + (pose1['pose'][7]-pose2['pose'][7])**2\
          + (pose1['pose'][11]-pose2['pose'][11])**2)**0.5

    with open(os.path.join(connectivity_dir, 'scans.txt')) as f:
        scans = [x.strip() for x in f]

    graphs = {}
    for scan in scans:
        with open(os.path.join(connectivity_dir, '%s_connectivity.json' % scan)) as f:
            G = nx.Graph()
            positions = {}
            data = json.load(f)
            for i,item in enumerate(data):
                if item['included']:
                    for j,conn in enumerate(item['unobstructed']):
                        if conn and data[j]['included']:
                            positions[item['image_id']] = np.array([item['pose'][3],
                                    item['pose'][7], item['pose'][11]]);
                            assert data[j]['unobstructed'][i], 'Graph should be undirected'
                            G.add_edge(item['image_id'],data[j]['image_id'],weight=distance(item,data[j]))
            nx.set_node_attributes(G, values=positions, name='position')
            graphs[scan] = G
    return graphs


def load_nav_graphs(connectivity_dir):
        """
        load graph from scan,
        Store the graph {scan_id: graph} in graphs
        Store the shortest path {scan_id: {view_id_x: {view_id_y: [path]} } } in paths
        Store the distances in distances. (Structure see above)
        Load connectivity graph for each scan, useful for reasoning about shortest paths
        """
        graphs = create_nav_graphs(connectivity_dir)
        shortest_paths = {}
        for scan, G in graphs.items():  # compute all shortest paths
            shortest_paths[scan] = dict(nx.all_pairs_dijkstra_path(G))
        shortest_distances = {}
        for scan, G in graphs.items():  # compute all shortest paths
            shortest_distances[scan] = dict(nx.all_pairs_dijkstra_path_length(G))

        return shortest_paths, shortest_distances


if __name__ == '__main__':
    random.seed(0) # Fix the seed of random
    parser = argparse.ArgumentParser()
    parser.add_argument('--connectivity_dir', default='connectivity_mp3d')
    parser.add_argument('--num_of_traj_per_instruction', default=10)
    args = parser.parse_args()

    val_unseen_set = ['QUCTc6BB5sX', 'pLe4wQe7qrG', 'X7HyMhZNoso', 'x8F5xyUWy9e', 'oLBMNvg9in8', '8194nk5LbLH', '2azQ1b91cZZ', 'zsNo4HB9uLZ', 'Z6MFQCViBuw', 'TbHJrupSAjP', 'EU6Fwq7SyZv']
    test_unseen_set = ['Vt2qJdWjCF2', 'rqfALeAoiTq', 'WYY7iVyf5p8', 'jtcxE69GiFV', '5ZKStnWn8Zo', 'gYvKGZ5eRqb', 'pa4otMbVnkk', 'yqstnuAEVhm', 'RPmz2sHmrrY', 'gxdoqLR6rwA', 'ARNzJeq3xxb', 'wc2JMjhGNzB', 'fzynW3qQPVF', 'YVUC4YcDtcY', 'UwV83HsGsw3', 'YFuZgdQ5vWj', 'q9vSo1VnCiC', '2t7WUuJeko7']

    shortest_paths, shortest_distances = load_nav_graphs(args.connectivity_dir)
    distance_to_node = {}
    for scene_id in shortest_distances:
        distance_to_node[scene_id] = {}
        for viewpoint_id in shortest_distances[scene_id]:
            distance_to_node[scene_id][viewpoint_id] = []
            for navigation_node_id in shortest_distances[scene_id][viewpoint_id]:
                distance_to_node[scene_id][viewpoint_id].append((shortest_distances[scene_id][viewpoint_id][navigation_node_id],navigation_node_id))
            distance_to_node[scene_id][viewpoint_id].sort()

    mp3d_instruction_annotation = json.load(open("mp3d_instruction_annotation.json","r"))
    tokenizer = get_tokenizer()
    NavRAG_train_annotation = []
    NavRAG_val_unseen_annotation = []
    NavRAG_test_unseen_annotation = []

    NavRAG_train_annotation_for_pretrain = []
    NavRAG_val_unseen_annotation_for_pretrain = []
    NavRAG_test_unseen_annotation_for_pretrain = []

    path_id = -1
    for scene_id in tqdm(mp3d_instruction_annotation):
        for item in mp3d_instruction_annotation[scene_id]:
            try:
                target_viewpoint_id = item["target_viewpoint"]
                num_of_viewpoints = len(distance_to_node[scene_id][target_viewpoint_id])

                num_of_traj_per_instruction = min(args.num_of_traj_per_instruction,num_of_viewpoints//3-1)
                if scene_id in val_unseen_set or scene_id in test_unseen_set:
                    num_of_traj_per_instruction = 1

                sampled_start_viewpoints = random.choices(distance_to_node[scene_id][target_viewpoint_id][1:num_of_viewpoints//3],k=num_of_traj_per_instruction)
                for sampled_distance, sampled_viewpoint_id in sampled_start_viewpoints:
                    path_id += 1
                    heading = random.uniform(0,2*math.pi)
                    annotation_item = {
                    "distance": sampled_distance,
                    "scan": scene_id,
                    "path_id": path_id,
                    "path": shortest_paths[scene_id][sampled_viewpoint_id][target_viewpoint_id],
                    "heading": heading,
                    "instructions": [ item["instruction"]["instruction"]
                        ],
                    "instr_encodings": [ tokenizer.encode(item["instruction"]["instruction"])
                        ],
                    }
                    if scene_id in val_unseen_set:
                        NavRAG_val_unseen_annotation.append(annotation_item)
                    elif scene_id in test_unseen_set:
                        NavRAG_test_unseen_annotation.append(annotation_item)
                    else:
                        NavRAG_train_annotation.append(annotation_item)

                    pretrain_annotation_item = {
                                              "instr_id": annotation_item["path_id"],
                                              "scan": annotation_item["scan"],
                                              "path": annotation_item["path"],
                                              "heading": annotation_item["heading"], 
                                              "instr_encoding": annotation_item["instr_encodings"][0]
                                              }
                    if scene_id in val_unseen_set:
                        NavRAG_val_unseen_annotation_for_pretrain.append(pretrain_annotation_item)
                    elif scene_id in test_unseen_set:
                        NavRAG_test_unseen_annotation_for_pretrain.append(pretrain_annotation_item)
                    else:
                        NavRAG_train_annotation_for_pretrain.append(pretrain_annotation_item)
            except:
                print("Error happens, skip...")

    json.dump(NavRAG_train_annotation, open("NavRAG_mp3d_train_enc.json","w"))
    json.dump(NavRAG_val_unseen_annotation, open("NavRAG_mp3d_val_unseen_enc.json","w"))
    json.dump(NavRAG_test_unseen_annotation, open("NavRAG_mp3d_test_enc.json","w"))

    NavRAG_val_seen_annotation_for_pretrain = random.choices(NavRAG_train_annotation_for_pretrain,k=500)

    with jsonlines.open("NavRAG_mp3d_train_enc.jsonl", "w") as wfd:
        for item in NavRAG_train_annotation_for_pretrain:
            wfd.write(item)

    with jsonlines.open("NavRAG_mp3d_val_seen_enc.jsonl", "w") as wfd:
        for item in NavRAG_val_seen_annotation_for_pretrain:
            wfd.write(item)

    with jsonlines.open("NavRAG_mp3d_val_unseen_enc.jsonl", "w") as wfd:
        for item in NavRAG_val_unseen_annotation_for_pretrain:
            wfd.write(item)

    with jsonlines.open("NavRAG_mp3d_test_enc.jsonl", "w") as wfd:
        for item in NavRAG_test_unseen_annotation_for_pretrain:
            wfd.write(item)

