# Motion-Artifact-Removal-in-Pixel-Frequency-Domain-via-Alternate-Masks-and-Diffusion-Model
This is the codebase for Motion-Artifact-Removal-in-Pixel-Frequency-Domain-via-Alternate-Masks-and-Diffusion-Model. 

This repository is based on [guided-diffusion](https://github.com/openai/guided-diffusion).
# Download pre-train Model
We provide pre-trained models for the HCP dataset, please save it to ```results/model/brain```. 

Here are the download link: 
https://drive.google.com/file/d/1Hh0wabKmW5CUXpUAS4GcEHZIoYeZq_v-/view?usp=sharing

# Sampling from the  pre-trained model
Please run:
```
python scripts/image_sample.py --conf_path ../conf/brain_sample_config.yml --img_dir brain --save_path motion_remove
```
Then, you can obtain the results of the example images after removing motion artefacts in ```results\motion_remove```
