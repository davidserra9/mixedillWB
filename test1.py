from glob import glob
import json
import shutil
from tqdm import tqdm

with open("/home/dserrano/Workspace/Transformer-WB/metadata/nikon_split.json", "r") as f:
    f = json.load(f)

train_places = []
valid_places = []
test_places = []

for key, place_list in f.items():
    if "train" in key:
        train_places += place_list
    elif "_val" in key:
        valid_places += place_list
    elif "test" in key:
        test_places += place_list

INPUT_PATH = "/home/dserrano/Documents/datasets/lsmi_mask_old/sony/input"
TARGET_PATH = "/home/dserrano/Documents/datasets/lsmi_mask_old/sony/target"

# for place in train_places:
#     input_place_paths = []
#     input_place_paths += glob(f"{INPUT_PATH}/{place}*_C_CS.png")
#     input_place_paths += glob(f"{INPUT_PATH}/{place}*_D_CS.png")
#     input_place_paths += glob(f"{INPUT_PATH}/{place}*_F_CS.png")
#     input_place_paths += glob(f"{INPUT_PATH}/{place}*_S_CS.png")
#     input_place_paths += glob(f"{INPUT_PATH}/{place}*_T_CS.png")
#
#     for path in input_place_paths:
#         shutil.copy(path, f"/media/david/media/lsmi_mask/sony/train/{path.split('/')[-1]}")
#
#     target_place_paths = glob(f"{OUTPUT_PATH}/{place}*")
#     for path in target_place_paths:
#         shutil.copy(path, f"/media/david/media/lsmi_mask/sony/train/{path.split('/')[-1].split('_')[0]}_{path.split('/')[-1].split('_')[1]}_G_AS.png")

for place in tqdm(valid_places):
    input_place_paths = []
    input_place_paths += glob(f"{INPUT_PATH}/{place}*_C_CS.png")
    input_place_paths += glob(f"{INPUT_PATH}/{place}*_D_CS.png")
    input_place_paths += glob(f"{INPUT_PATH}/{place}*_F_CS.png")
    input_place_paths += glob(f"{INPUT_PATH}/{place}*_S_CS.png")
    input_place_paths += glob(f"{INPUT_PATH}/{place}*_T_CS.png")

    for path in input_place_paths:
        shutil.copy(path, f"/home/dserrano/Documents/datasets/lsmi_mask/sony/valid/{path.split('/')[-1]}")

    target_place_paths = glob(f"{TARGET_PATH}/{place}*")
    for path in target_place_paths:
        shutil.copy(path, f"/home/dserrano/Documents/datasets/lsmi_mask/sony/valid/{path.split('/')[-1].replace('_png', '.png').replace('GT', 'G')}")

for place in tqdm(test_places):
    input_place_paths = []
    input_place_paths += glob(f"{INPUT_PATH}/{place}*_C_CS.png")
    input_place_paths += glob(f"{INPUT_PATH}/{place}*_D_CS.png")
    input_place_paths += glob(f"{INPUT_PATH}/{place}*_F_CS.png")
    input_place_paths += glob(f"{INPUT_PATH}/{place}*_S_CS.png")
    input_place_paths += glob(f"{INPUT_PATH}/{place}*_T_CS.png")

    for path in input_place_paths:
        shutil.copy(path, f"/home/dserrano/Documents/datasets/lsmi_mask/sony/test/{path.split('/')[-1]}")

    target_place_paths = glob(f"{TARGET_PATH}/{place}*")
    for path in target_place_paths:
        shutil.copy(path, f"/home/dserrano/Documents/datasets/lsmi_mask/sony/test/{path.split('/')[-1].replace('_png', '.png').replace('GT', 'G')}")







