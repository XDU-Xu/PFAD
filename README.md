# Motion-Artifact-Removal-in-Pixel-Frequency-Domain-via-Alternate-Masks-and-Diffusion-Model
This project is about artifact removal of medical images. The code reference to guided-diffusion(https://github.com/openai/guided-diffusion)

Please run:
```
python scripts/image_sample.py --conf_path ../conf/brain_sample_config.yml --img_dir brain --save_path motion_remove
```
Then, you can obtain the results of the example images after removing motion artefacts in ```results\motion_remove```
