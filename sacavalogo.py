import argparse
import os
import sqlite3
import statistics

import numpy as np
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--laion-logo-parquet", type=str, required=True)
parser.add_argument("--laion-logo-embeddings-dir", type=str, required=True)
parser.add_argument("--ava-txt", type=str, required=True)
parser.add_argument("--ava-embeddings-dir", type=str, required=True)
parser.add_argument("--sac-sqlite", type=str, required=True)
parser.add_argument("--sac-embeddings-dir", type=str, required=True)
parser.add_argument("--output-dir", type=str, required=True)
parser.add_argument("--x-only", action="store_true", default=False)
parser.add_argument("--y-only", action="store_true", default=False)
args = parser.parse_args()


def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel

    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)


x = []
y = []

df = pd.read_parquet(args.laion_logo_parquet)

for idx, row in tqdm(df.iterrows()):
    average_rating = float(row.proffesionalism_average)
    if average_rating < 1:
        continue
    try:
        im_emb_arr = np.load(f"{args.laion_logo_embeddings_dir}/{row.key}.img.npy")
    except:
        continue
    if args.x_only:
        x.append(normalized(im_emb_arr))
        continue
    if args.y_only:
        y_ = np.zeros((1, 1))
        y_[0][0] = average_rating
        y.append(y_)
        continue

    x.append(normalized(im_emb_arr))
    y_ = np.zeros((1, 1))
    y_[0][0] = average_rating
    y.append(y_)


f = open(args.ava_txt, "r")
data = f.readlines()
all_ratings = {}
for line in tqdm(data):
    line = line.strip()
    split = line.split(" ")
    image_id = split[1]
    ratings = []
    for i in range(int(split[2])):
        ratings.append(1)
    for i in range(int(split[3])):
        ratings.append(2)
    for i in range(int(split[4])):
        ratings.append(3)
    for i in range(int(split[5])):
        ratings.append(4)
    for i in range(int(split[6])):
        ratings.append(5)
    for i in range(int(split[7])):
        ratings.append(6)
    for i in range(int(split[8])):
        ratings.append(7)
    for i in range(int(split[9])):
        ratings.append(8)
    for i in range(int(split[10])):
        ratings.append(9)
    for i in range(int(split[11])):
        ratings.append(10)
    mean = statistics.mean(ratings)
    average = float(mean)
    all_ratings[image_id] = average

for line in tqdm(data):
    line = line.strip()
    split = line.split(" ")
    image_id = split[1]
    average = all_ratings[image_id]
    try:
        im_emb_arr = np.load(f"{args.ava_embeddings_dir}/{image_id}.img.npy")
    except:
        continue
    if args.x_only:
        x.append(normalized(im_emb_arr))
        continue
    if args.y_only:
        y_ = np.zeros((1, 1))
        y_[0][0] = average
        y.append(y_)
        continue
    x.append(normalized(im_emb_arr))
    y_ = np.zeros((1, 1))
    y_[0][0] = average
    y.append(y_)

conn = sqlite3.connect(args.sac_sqlite)
c = conn.cursor()
c.execute("select iid, path from paths")
rows = c.fetchall()
ratings = {}
c.execute("select iid, rating from ratings")
rows2 = c.fetchall()
for row in rows2:
    if row[0] not in ratings:
        ratings[row[0]] = []
    ratings[row[0]].append(float(row[1]))

for row in tqdm(rows):
    iid = row[0]
    path = os.path.splitext(row[1])[0]
    try:
        im_emb_arr = np.load(f"{args.sac_embeddings_dir}/{path}.img.npy")
    except:
        continue
    try:
        r = ratings[iid]
    except:
        continue
    if len(r) == 0:
        continue
    mean = statistics.mean(r)
    average = float(mean)
    if args.x_only:
        x.append(normalized(im_emb_arr))
        continue
    if args.y_only:
        y_ = np.zeros((1, 1))
        y_[0][0] = average
        y.append(y_)
        continue
    x.append(normalized(im_emb_arr))
    y_ = np.zeros((1, 1))
    y_[0][0] = average
    y.append(y_)
conn.close()

if args.x_only:
    x = np.vstack(x)
    print(x.shape)
    np.save(f"{args.output_dir}/sac+ava+laion-logo_x.npy", x)
    exit(0)
if args.y_only:
    y = np.vstack(y)
    print(y.shape)
    np.save(f"{args.output_dir}/sac+ava+laion-logo_y.npy", y)
    exit(0)

x = np.vstack(x)
y = np.vstack(y)
print(x.shape)
print(y.shape)
np.save(f"{args.output_dir}/sac+ava+laion-logo_x.npy", x)
np.save(f"{args.output_dir}/sac+ava+laion-logo_y.npy", y)
