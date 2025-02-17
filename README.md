### NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM

#### Zihan Wang, Yaohui Zhu, Gim Hee Lee, Yachun Fan,


>Vision-and-Language Navigation (VLN) is an essential skill for embodied agents, allowing them to navigate in 3D environments following natural language instructions. High-performance navigation models require a large amount of training data, the high cost of manually annotating data has seriously hindered this field. Therefore, some previous methods translate trajectory videos into step-by-step instructions for expanding data, but such instructions do not match well with users' communication styles that briefly describe destinations or state specific needs. Moreover, local navigation trajectories overlook global context and high-level task planning. To address these issues, we propose NavRAG, a retrieval-augmented generation (RAG) framework that generates user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical scene description tree for 3D scene understanding from global layout to local details, then simulates various user roles with specific demands to retrieve from the scene tree, generating diverse instructions with LLM. We annotate over 2 million navigation instructions across 861 scenes and evaluate the data quality and navigation performance of trained models.

### TODOs

* [X] Release the instruction generation code for MP3D and HM3D.
* [X] Release the DUET code for the NavRAG dataset and the REVERIE dataset.
* [X] Release NavRAG dataset and preprocessed feature files.
* [X] Release the checkpoints.
* [X] Release annotations of scene description tree for MP3D.

### Requirements

1. Install the Matterport3D simulator for pre-training your model: follow the instructions [here](https://github.com/peteanderson80/Matterport3DSimulator).
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```
2. Download the NavRAG dataset, preprocessed feature files, and checkpoints from [TeraBox](https://1024terabox.com/s/1D5HEHsaW5AcWTjjIO15jpA).
3. (Optional) Install the Habitat simulator and download `Matterport3D scenes (MP3D)` to obtain the RGB-D images: follow instructions [here](https://github.com/jacobkrantz/VLN-CE).
4. (Optional) Download the `Habitat-Matterport 3D scenes (HM3D)` from [habitat-matterport-3dresearch](https://github.com/matterport/habitat-matterport-3dresearch).
   ```
   hm3d-train-habitat-v0.2.tar
   hm3d-val-habitat-v0.2.tar
   ```
5. (Optional) Input your OpenAI Key into `instruction_generator/openai_key.json`

### Pre-train the DUET model on the NavRAG dataset

```
cd VLN-DUET-NAVRAG/pretrain_src
bash run_rag_h14.sh "0,1" 2345
```

### Fine-tune or evaluate the DUET model on the NavRAG dataset

```
cd VLN-DUET-NAVRAG/map_nav_src
bash scripts/rag_h14_envedit_mix.sh "0,1" 2346
```

### Fine-tune or evaluate the DUET model on the REVERIE dataset

```
cd VLN-DUET-RVR/map_nav_src
bash scripts/reverie_h14_envedit_mix.sh "0,1" 2346
```

### (Optional) Generate the NavRAG dataset
```
python3 get_mp3d_image.py # For HM3D scenes, get_hm3d_image.py
python3 get_viewpoint_summary.py
python3 get_zones.py
python3 get_house_summary.py
python3 generate_instruction.py
python3 convert_to_dataset.py
```

### Acknowledgments
Our code is based on [DUET](https://github.com/cshizhe/VLN-DUET), some code and data are from [ScaleVLN](https://github.com/wz0919/ScaleVLN) and [BEVBert](https://github.com/MarSaKi/VLN-BEVBert). Thanks for their great works!
