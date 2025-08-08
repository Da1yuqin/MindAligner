import json
import os
import sys

import clip
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from my_utils.data_utils import get_train_dataloaders, load_nsd_images


def save_image(image, image_id, category_name, subj_id):
    output_dir = os.path.join(output_base_dir, category_name)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, f"subj{subj_id}_{image_id}.jpg")
    
    image.save(output_path)


image_dataset = load_nsd_images(".../your_data_path")  # TODO: change it to your file path
print("Loaded all 73k possible NSD images to cpu!", image_dataset.shape)

num_sessions = 1                   # TODO: number of sessions processed
output_base_dir = './v2subj1257/'  # TODO: results save to
os.makedirs(output_base_dir, exist_ok=True)

subj_list = [1, 2, 5, 7]
train_dls, voxels, num_trains = get_train_dataloaders(
    subj_ids=subj_list, num_sessions=num_sessions, return_num_train=True
)
print("**********num_trains:", num_trains)  # num_trains: [688, 688, 688, 688], uniq 536

device = 'cuda'
model, preprocess = clip.load("ViT-L/14", device=device)


category_names = [
    "person", "bicycle", "car", "motorcycle", "airplane", "train", "truck", "boat", "traffic",
    "fire hydrant", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball", "skateboard", "surfboard",
    "tennis racket", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "laptop", "keyboard",
    "cell phone", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush", "computer", "people", "A variety of fruits", "girl", "boy", "men", "woman", "computer", "surf"
]

for id, subj_id in enumerate(subj_list):
    category_image_idx = {cat: [] for cat in category_names}
    behav, _, _, _ = next(iter(train_dls[id]))
    image_idx = behav[:,0,0].cpu().long().numpy()
    voxel_idx = behav[:,0,5].cpu().long().numpy()
    
    image_id, image_sorted_idx = np.unique(image_idx, return_index=True)
    voxel_sorted_idx = voxel_idx[image_sorted_idx]
    voxel = voxels[f'subj0{subj_list[id]}'][voxel_sorted_idx]
    voxel = torch.Tensor(voxel).unsqueeze(1)

    image_np = image_id.reshape(-1)    
    image_data = np.array([image_dataset[idx] for idx in image_np])
    image_tensor = torch.tensor(image_data)

    for i, image_id in tqdm(enumerate(image_np), desc="processing"):
        img_tensor = image_tensor[i]
        img = transforms.ToPILImage()(img_tensor)
        image = img.convert('RGB')
        
        image_input = preprocess(image).unsqueeze(0).to(device)
        with torch.no_grad():
            image_features = model.encode_image(image_input)
        text_inputs = clip.tokenize(category_names).to(device)
        with torch.no_grad():
            text_features = model.encode_text(text_inputs)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        best_match_idx = similarity.argmax(dim=1).item()
        best_match_category = category_names[best_match_idx]
        
        category_image_idx[best_match_category].append(str(image_id) + "_" + str(i))
        
        save_image(image, image_id, best_match_category, subj_id)

    with open(os.path.join(output_base_dir, f'category_image_idx_subj{subj_id}.json'), 'w') as f:
        json.dump(category_image_idx, f, indent=4)

print("Classification completed, and the images have been saved to the specified folder.")
