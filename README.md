# MoStGAN: Video Generation with Temporal Motion Styles

[CVPR 2023] Official pytorch implementation

[Webpage]() [Paper]()

## Installation

```
conda env create -f environment.yaml
```

And also make sure [StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch#requirements) is runnable. 

## System requirements

4 32GB V100 is required, training time is approximately 2 days

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

## Bibtex
