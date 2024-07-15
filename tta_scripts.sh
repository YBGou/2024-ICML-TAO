# !/bin/bash

############################################
# Single Image Dehazing
############################################

# declare -A pairs=( [A]=3 [B]=4 [C]=5 [D]=6 [E]=7 )

# for i in "${!pairs[@]}"; do
# {
#   j=${pairs[$i]}
#   echo CUDA=$j DATA=$i Processing...
#   CUDA_VISIBLE_DEVICES=$j python sample_dehazing.py \
#   --sample_dir test_samples/HSTS_256x256/sub_syn_$i > /dev/null
# }&
# done
# wait

# HQ=test_samples/HSTS_256x256/results;
# GT=test_samples/HSTS_256x256/original;
# python img_qua_ass/inference_iqa.py -m PSNR -i $HQ -r $GT;
# python img_qua_ass/inference_iqa.py -m SSIM -i $HQ -r $GT;

############################################
# Low-light Image Enhancement
############################################

# declare -A pairs=( [A]=3 [B]=4 [C]=5 [D]=6 [E]=7 )

# for i in "${!pairs[@]}"; do
# {
#   j=${pairs[$i]}
#   echo CUDA=$j DATA=$i Processing...
#   CUDA_VISIBLE_DEVICES=$j python sample_lowlightE.py \
#   --sample_dir test_samples/LOL_256x256/sub_lol_$i > /dev/null
# }&
# done
# wait

# HQ=test_samples/LOL_256x256/results;
# GT=test_samples/LOL_256x256/original;
# python img_qua_ass/inference_iqa.py -m PSNR -i $HQ -r $GT;
# python img_qua_ass/inference_iqa.py -m SSIM -i $HQ -r $GT;

############################################
# Single Image Denoising
############################################

# declare -A pairs=( [A]=4 [B]=5 [C]=6 [D]=7 )

# for i in "${!pairs[@]}"; do
# {
#   j=${pairs[$i]}
#   echo CUDA=$j DATA=$i Processing...
#   CUDA_VISIBLE_DEVICES=$j python sample_denoising.py \
#   --sample_dir test_samples/Kodak24_256x256/sigma30_$i > /dev/null
# }&
# done
# wait

# HQ=test_samples/Kodak24_256x256/results;
# GT=test_samples/Kodak24_256x256/original;
# python img_qua_ass/inference_iqa.py -m PSNR -i $HQ -r $GT;
# python img_qua_ass/inference_iqa.py -m SSIM -i $HQ -r $GT;