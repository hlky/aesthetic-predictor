# aesthetic-predictor
 
CLIP+MLP Aesthetic Score Predictor, using ViT-H, based on [improved-aesthetic-predictor](https://github.com/christophschuhmann/improved-aesthetic-predictor/)

# Want to help?

(Rate images for aesthetics)[https://tinybots.net/artbot/rate] and get (Stable Horde)[https://stablehorde.net/] Kudos

Datasets used:

* [AVA: A Large-Scale Database for Aesthetic Visual Analysis](https://academictorrents.com/details/71631f83b11d3d79d8f84efe0a7e12f0ac001460)
* [Simulacra Aesthetic Captions](https://github.com/JD-P/simulacra-aesthetic-captions)
* [LAION-logos](https://huggingface.co/datasets/ChristophSchuhmann/aesthetic-logo-ratings)


[precomputed ViT-H embeds for sac+ava+logos](https://dataset.sygil.dev/sac+ava+logos_vitH_embeds.tar)

# sacavalogo.py

prepares the data for training

[dataset ratings files](https://dataset.sygil.dev/sac+ava+logos_ratings.tar) (to use with precomputed embeds)

```
usage: sacavalogo.py [-h] --laion-logo-parquet LAION_LOGO_PARQUET --laion-logo-embeddings-dir LAION_LOGO_EMBEDDINGS_DIR --ava-txt AVA_TXT --ava-embeddings-dir AVA_EMBEDDINGS_DIR --sac-sqlite
                     SAC_SQLITE --sac-embeddings-dir SAC_EMBEDDINGS_DIR --output-dir OUTPUT_DIR [--x-only] [--y-only]
sacavalogo.py: error: the following arguments are required: --laion-logo-parquet, --laion-logo-embeddings-dir, --ava-txt, --ava-embeddings-dir, --sac-sqlite, --sac-embeddings-dir, --output-dir
```

use `--x-only` or `--y-only` to only prepare one part of the data, in case you want to try something different for the ratings only

# train_predictor.py

```
usage: train_predictor.py [-h] --x-npy X_NPY --y-npy Y_NPY [--output-dir OUTPUT_DIR] [--save-name SAVE_NAME] [--batch-size BATCH_SIZE] [--lr LR] [--optimizer {adam,adamw}]
                          [--val-percentage VAL_PERCENTAGE] [--epochs EPOCHS] [--val-count VAL_COUNT]
train_predictor.py: error: the following arguments are required: --x-npy, --y-npy
```

# app.py

gradio demo

```
usage: app.py [-h] --model-path MODEL_PATH [--device {cuda,cpu}] [--port PORT]
app.py: error: the following arguments are required: --model-path
```
