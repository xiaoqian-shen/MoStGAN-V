# MoStGAN: Video Generation with Temporal Motion Styles

Official PyTorch implementation of MoStGAN: Video Generation with Temporal Motion Styles

[[paper]]() [[project page]](https://xiaoqian-shen-projects.on.drv.tw/webpage/mostgan/)

<div style="text-align:center">
<img src="https://mostgan.s3.us-east-2.amazonaws.com/output.gif" alt='None'>
</div>

## Installation

```
conda env create -f environment.yaml
```

And also make sure [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements) is runnable. 

## System requirements

4 32GB V100s are required, training time is approximately 2 days

## Data

+ [CelebV-HQ](https://celebv-hq.github.io)
+ [FaceForensics](https://github.com/ondyari/FaceForensics)

+ [SkyTimelapse](https://github.com/weixiong-ur/mdgan)
+ [RainbowJelly](https://www.youtube.com/watch?v=P8Bit37hlsQ)
+ [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)

We follow the same procedure as [StyleGAN-V](https://github.com/universome/stylegan-v) to process all datasets

```
convert_videos_to_frames.py -s /path/to/source -t /path/to/target --video_ext mp4 --target_size 256
```

FaceForensics was preprocessed with `src/scripts/preprocess_ffs.py` to extract face crops, (result in a little bit unstable).

## Training

```
python src/infra/launch.py hydra.run.dir=. exp_suffix=my_experiment_name env=local dataset=ffs dataset.resolution=256 num_gpus=4
```

## Inference

+ evaluation

```
src/scripts/calc_metrics.py
```

+ generation

```
python src/scripts/generate.py --network_pkl /path/to/network-snapshot.pkl --num_videos 25 --as_grids true --save_as_mp4 true --fps 25 --video_len 128 --batch_size 25 --outdir /path/to/output/dir --truncation_psi 0.9
```

## Reference

This code is mainly built upon [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) and [StyleGAN-V](https://github.com/universome/stylegan-v) repositories.

Baseline codes are from [MoCoGAN-HD](https://github.com/snap-research/MoCoGAN-HD), [VideoGPT](https://github.com/wilson1yan/VideoGPT), [DIGAN](https://github.com/sihyun-yu/digan), [StyleGAN-V](https://github.com/universome/stylegan-v)

## Bibtex

