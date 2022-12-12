import argparse
import os
import threading
import time

import numpy as np
import open_clip
import torch
from loguru import logger
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument("--input-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--batch-size", type=int, default=64)
parser.add_argument("--num_preprocess_threads", type=int, default=6)
parser.add_argument("--gpu-id", type=int, default=0)
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

threads = []
lock = threading.Lock()

processed_images = []


def get_files(path, extensions):
    files = []
    for file in os.listdir(path):
        for extension in extensions:
            if file.endswith(extension):
                if len(file) > 255:
                    file = file[:200]
                if not os.path.exists(
                    f"{args.output_dir}/" + file.split(".")[0] + ".img.npy"
                ):
                    files.append(
                        {
                            "path": os.path.join(path, file),
                            "name": file.split(".")[0] + ".img.npy",
                        }
                    )
    return files


logger.info(f"Checking input directory: {args.input_dir}")
image_files = get_files(args.input_dir, [".jpg", ".png", ".jpeg"])
logger.info(f"Found {len(image_files)} images")

logger.info(f"Checking output directory: {args.output_dir}")
if not os.path.exists(args.output_dir):
    os.makedirs(args.output_dir)
    logger.info(f"Created output directory: {args.output_dir}")
else:
    logger.info(f"Output directory already exists: {args.output_dir}")

device = "cpu" if args.cpu else f"cuda:{args.gpu_id}"
offload = "cpu"

logger.info(f"Using device: {device}")

logger.info("Loading ViT-H-14")
model, _, preprocess = open_clip.create_model_and_transforms(
    "ViT-H-14", pretrained="laion2b_s32b_b79k"
)
model.to(device)
logger.info("Loaded ViT-H-14")


def preprocess_image(image_filename):
    return (
        preprocess(Image.open(image_filename["path"]))
        .unsqueeze(0)
        .to(offload, non_blocking=True)
    )


def image_embed(image_filename, outpath):
    image = preprocess_image(image_filename)
    embed_file = image_filename.split(".")[0] + ".img.npy"
    with torch.no_grad():
        image_features = model.encode_image(image)
    im_emb_arr = image_features.cpu().detach().numpy()
    np.save(f"{outpath}/{embed_file}", im_emb_arr)


def preprocess_worker():
    while len(image_files) > 0:
        lock.acquire()
        image_filename = image_files.pop()
        lock.release()
        try:
            preprocessed_image = preprocess_image(image_filename)
        except Exception as e:
            logger.error(f"Error preprocessing image {image_filename['path']}: {e}")
            continue
        lock.acquire()
        processed_images.append(
            {"image": preprocessed_image, "filename": image_filename["name"]}
        )
        lock.release()


last_batch_took = 0


def embed_worker():
    global processed_images, last_batch_took
    while len(processed_images) > 0:
        if len(processed_images) < args.batch_size:
            time.sleep(0.1)
        lock.acquire()
        tik = time.time()
        batch = processed_images[: args.batch_size]
        processed_images = processed_images[args.batch_size :]
        lock.release()
        with torch.no_grad():
            images = torch.cat([image["image"] for image in batch])
            image_features = model.encode_image(images.to(device, non_blocking=True))
        for i, image in enumerate(batch):
            im_emb_arr = image_features[i].cpu().detach().numpy()
            filename = image["filename"].split(".")[0] + ".img.npy"
            try:
                np.save(f"{args.output_dir}/{filename}", im_emb_arr)
            except Exception as e:
                logger.error(f"Error saving image {filename}: {e}")
                continue
        tok = time.time()
        lock.acquire()
        last_batch_took = tok - tik
        lock.release()


for i in range(args.num_preprocess_threads):
    logger.info(f"Starting preprocess thread {i}")
    t = threading.Thread(target=preprocess_worker, args=())
    t.start()
    threads.append(t)

time.sleep(5)

logger.info(f"Starting embed thread")
t = threading.Thread(target=embed_worker, args=())
t.start()
threads.append(t)

prev_image_files = len(image_files)
prev_processed_images = len(processed_images)


def log_queue():
    global image_files, processed_images, prev_image_files, prev_processed_images, last_batch_took
    while len(image_files) > 0 or len(processed_images) > 0:
        lock.acquire()
        images_processed = prev_image_files - len(image_files)
        per_second = images_processed / 10
        prev_image_files = len(image_files)
        lock.release()
        logger.info(
            f"Images remaining: {len(image_files)} | Added to queue: {images_processed} ({per_second}/s)"
        )
        logger.info(f"Images in queue: {len(processed_images)}")
        if last_batch_took > 0:
            logger.info(
                f"Last batch took: {last_batch_took}s ({args.batch_size / last_batch_took}/s)"
            )
        time.sleep(10)


logger.info(f"Starting log thread")
t = threading.Thread(target=log_queue, args=())
t.start()

for t in threads:
    t.join()
