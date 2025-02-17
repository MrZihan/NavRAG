### NavRAG: Generating User Demand Instructions for Embodied Navigation through Retrieval-Augmented LLM

#### Zihan Wang, Yaohui Zhu, Gim Hee Lee, Yachun Fan,


>Vision-and-Language Navigation (VLN) is an essential skill for embodied agents, allowing them to navigate in 3D environments following natural language instructions. High-performance navigation models require a large amount of training data, the high cost of manually annotating data has seriously hindered this field. Therefore, some previous methods translate trajectory videos into step-by-step instructions for expanding data, but such instructions do not match well with users' communication styles that briefly describe destinations or state specific needs. Moreover, local navigation trajectories overlook global context and high-level task planning. To address these issues, we propose NavRAG, a retrieval-augmented generation (RAG) framework that generates user demand instructions for VLN. NavRAG leverages LLM to build a hierarchical scene description tree for 3D scene understanding from global layout to local details, then simulates various user roles with specific demands to retrieve from the scene tree, generating diverse instructions with LLM. We annotate over 2 million navigation instructions across 861 scenes and evaluate the data quality and navigation performance of trained models.

## TODOs

* [X] Release the instruction generation code for MP3D.
* [ ] Release the instruction generation code for HM3D.
* [X] Release the DUET code for the NavRAG dataset and the REVERIE dataset.
* [ ] Release NavRAG dataset and preprocessed feature files.
* [ ] Release the checkpoints.

## Requirements

1. Install Matterport3D simulator for pre-training your model: follow instructions [here](https://github.com/peteanderson80/Matterport3DSimulator).
```
export PYTHONPATH=Matterport3DSimulator/build:$PYTHONPATH
```
2. (Optional) Install Habitat simulator for obtaining the RGB-D images from MP3D and HM3D scenes: follow instructions [here](https://github.com/jacobkrantz/VLN-CE).

3. 