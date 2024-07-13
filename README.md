# MedSAM for multi-modality head and neck cancer GTV segmentation.

This is a work based on the LiteMedSAM model (https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM). 

Below is a demo for CT-only head and neck region tumor and organ segmentation with bounding box. 

<video width="320" height="240" controls>
  <source src="CT_demo2.mp4" type="video/mp4">
</video>

## Installation

The codebase is tested with: `Ubuntu 20.04` | Python `3.10` | `CUDA 11.8` | `Pytorch 2.1.2`
1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/Aarhus-RadOnc-AI/MedSAM_multimodality_HNC_GTV_segmentation`
4. Run `pip install -e .` to install.

