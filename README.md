
# FmapSynth: Deep Learning-Based Fieldmap Synthesis for fMRI Distortion Correction

This project provides a deep learning model designed to estimate underlying fieldmaps for subsequent distortion correction of susceptibility artifacts in fMRI, specifically for unwarping distorted BOLD images using a tool like FSL FUGUE.

The model takes as input:
* A distorted BOLD image in native space
* A resampled T1-weighted structural image

And outputs:
* Estimated fieldmap that can be used for subsequent unwarping.

## Data
The model is trained on publically available data retrieved from OpenNeuro
[OpenNeuro](https://openneuro.org/)

| Case           | Dataset  | Version | Total Sub. | Sampled | Sessions | Seq. Len. | Vendor / Strength | Healthy | State | BOLD Dir. | Fmap. Dir. |
|----------------|----------|---------|------------|---------|----------|-----------|-------------------|---------|-------|-----------|------------|
| **Train & Val**| ds005038 | 1.0.3   | 58         | 50      | 2        | N/A       | S / 3             | Y       | R     | j-        | j-         |
| Total: 190     | ds002422 | 1.1.0   | 46         | 40      | 1        | N/A       | S / 1.5           | Y       | R     | j-        | -          |
| subjects       | ds003745 | 2.1.1   | 50         | 50      | 1        | N/A       | S / 3             | Y       | T     | j-        | j-         |
|                | ds004044 | 2.0.3   | 62         | 50      | 1        | N/A       | S / 3             | Y       | T     | j-        | i          |
| **Test**       | ds003835 | 1.0.2   | 24         | 10      | 1        | 360       | S / 3             | Y       | R/T   | j-        | j-         |
| Total: 90      | ds005165 | 1.0.4   | 10         | 10      | 5        | 212       | S / 3             | Y       | R     | j-        | j-         |
| Subjects       | ds000224 | 1.0.4   | 10         | 10      | 13       | 818       | S / 3             | Y       | R     | j-        | j-         |
|                | ds001454 | 1.3.1   | 24         | 10      | 2        | 195       | S / 3             | Y       | R     | j-        | j-         |
|                | ds004182 | 1.0.1   | 50         | 10      | 1        | 300       | S / 3             | Y       | R     | j         | j          |
|                | ds005263 | 1.0.0   | 68         | 10      | 1        | 488       | S / 3             | Y       | T     | j         | j-         |
|                | ds002898 | 1.4.2   | 27         | 10      | 1        | 242       | S / 3             | Y       | R     | i         | i          |
|                | ds004073 | 1.0.1   | 51         | 20      | 1        | 545       | P / 3             | Y       | T     | j         | j          |


## Docker

The best performing model is packed in a Docker image which can be pulled from https://hub.docker.com/r/maglindchr/fmapsynth

The model itself, and two example images can be retrieved [here](https://drive.google.com/drive/folders/1V2RDDLP2VzG8n5O6mMlezzlZXzB1QnT5?usp=drive_link) by running `docker pull maglindchr/fmapsynth:v1.0`. 

The model's performance on the example images has not been examined. 

The image performs a simple set of preprocessing steps (motion correction, coregistration, etc.), and have already the necessary tools installed.

Before running the container, prepare your data as follows:
* Create an input folder containing the distorted BOLD image as `BOLD.nii.gz` and the high resolution T1w images as `T1w.nii.gz`.
* Create an empty output folder to hold the estimated fieldmap `fieldmap.nii.gz` 

Mount and run the docker container by running the following

```bash
docker run --rm \
-v "path/to/input/folder":/data/input:ro \
-v "path/to/output/folder":/data/output \
maglindchr/fmapsynth:v1.0
```


## Authors

- [@Cyrkryta (Magnus Lindberg Christensen)](https://github.com/Cyrkryta/)


## Acknowledgements
A big thank you to Melanie Ganz-Benjaminsen from the University of Copenhagen, as well as Patrick Fisher and Cyril Pernet from NRU, for being great supervisors throughout the project. Also, a big thank you to Jan Tagscherer who is the initial Master student wokring on the project, affiliated with NRU.
