## Test-Time Degradation Adaptation for Open-Set Image Restoration (ICML 2024, Spotlight)

This work studied a challenging problem of Open-set Image Restoration (OIR), and revealed its essence from the perspective of distribution shift. In recent, test-time adaptation has emerged as an effective methodology to address this inherent disparity. As a consequence, this work presented a test-time degradation adaptation framework for addressing OIR.

#### Environments

The code is tested with Pytorch 2.0.1 and CUDA 11.7 on Ubuntu 20.04. Run the following command to install dependencies:

    pip install -r requirements.txt

#### Pretrained Model and Datasets

Please download the pretrained unconditional DDPM on ImageNet-256 (i.e., 256x256_diffusion_uncond.pt) from [this page](https://github.com/openai/guided-diffusion) and put it in folder `test_models`. This work adopts an unconditional pre-trained DDPM as foundation model for OIR due to the following considerations. First, it captures rich knowledges of generating various high-quality visual scenarios, which could be regarded as a generic pretraining for OIR targeting at producing clean images. Second, it is degradation-agnostic and any degradations in the test data could be considered as unforeseen.

This work adopts the synthetic dataset of HSTS from [RESIDE](https://sites.google.com/view/reside-dehaze-datasets/reside-standard?authuser=3D0) for image dehazing. The testing pairs from [LOL](https://daooshee.github.io/BMVC2018website/) are employed for low-light image enhancement. [Kodak24](https://github.com/MohamedBakrAli/Kodak-Lossless-True-Color-Image-Suite/tree/master) dataset is used for image denoising by adding the Gaussian noises with the noise level of $\sigma=30$ to clean images. Since the DDPM is pre-trained on the images of size $256\times 256$, we preprocess the images in the datasets by first center cropping them along the shorter edges, and then resizing them to match the image size.

We have provided the processed data and the method's results in folder `test_samples`.

#### TTA for OIR tasks

To explore the upper bound of the method on each type of degradation, the loss weights $\lambda_{1-3}$, $\gamma_{1-5}$ and guidance scale $s$ (Line 123-125, 160-172, 229 in `sample_xxx.py` files, respectively) are adjusted for different types of degradation. Specifically, several representative images of target degradation are selected first, and then used to adjust the parameters according to their quantitative or qualitative results. Finally, the obtained parameters are applied to all images of the target degradation. Here, the parameters for the degradations presented in the paper are provided.

Single Image Dehazing

    python sample_dehazing.py --sample_dir input_image_folder --result_dir output_image_folder

Low-light Image Enhancement

    python sample_lowlightE.py --sample_dir input_image_folder --result_dir output_image_folder

Single Image Denoising (Gaussian noises $\sigma=30$)

    python sample_denoising.py --sample_dir input_image_folder --result_dir output_image_folder

In addition, this repo provides a bash script for Ubuntu system to concurrently process multiple image folders through multiple GPUs. Remove the corresponding comments before running the script to handle the degradations. All experimental results in the paper are obtained through this script.

    bash tta_scripts.sh

To assess the performance, the metrics of PSNR and SSIM are employed which are calculated through
    
    python img_qua_ass/inference_iqa.py -m PSNR -i result_image_foler -r ground_truths_folder

    python img_qua_ass/inference_iqa.py -m SSIM -i result_image_foler -r ground_truths_folder

#### Citation

If this codebase is useful for your works, please cite the following paper:

    @inproceedings{gou2024tao,
        title={Test-Time Degradation Adaptation for Open-Set Image Restoration},
        author={Yuanbiao Gou and Haiyu Zhao and Boyun Li and Xinyan Xiao and Xi Peng},
        booktitle={Forty-first International Conference on Machine Learning},
        month={Jul.},
        year={2024}
    }

#### Acknowledgement

This repo is built upon the open-source repo of [GD](https://github.com/openai/guided-diffusion), [GDP](https://github.com/Fayeben/GenerativeDiffusionPrior) and [IQA](https://github.com/chaofengc/IQA-PyTorch), thanks for their excellent works.
