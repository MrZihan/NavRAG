import json
import requests
import numpy as np
    

zone_annotation = json.load(open('mp3d_zone_annotation.json','r'))
for scene_id in zone_annotation:
     scene_connectivity_summary = ""
     connectivity_matrix = np.array(zone_annotation[scene_id][0])
     for zone_id_1 in range(connectivity_matrix.shape[0]):
        zone_annotation[scene_id][1]["zone_"+str(zone_id_1+1)] = zone_annotation[scene_id][1]["zone_"+str(zone_id_1+1)][1]
        connectivity_summary = "zone_"+str(zone_id_1+1)+" is connected with "
        for zone_id_2 in range(connectivity_matrix.shape[1]):
                if connectivity_matrix[zone_id_1,zone_id_2] == 1:
                        connectivity_summary += "zone_"+str(zone_id_2+1)+", "
        if connectivity_summary != "zone_"+str(zone_id_1+1)+" is connected with ":
                connectivity_summary = connectivity_summary[:-2]+"\n "
        else:
                connectivity_summary = ''

        scene_connectivity_summary += connectivity_summary

     zone_annotation[scene_id][0] = scene_connectivity_summary
     

with open("mp3d_house_annotation.json", "w") as file:
        json.dump(zone_annotation, file)
