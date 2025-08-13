from transformers import OneFormerForUniversalSegmentation, OneFormerProcessor
import torch
from Data_Process_6 import CustomDataset
from torch.utils.data import DataLoader
from transformers import (
    default_data_collator,
)

import json
with open("labels/id2label.json", "r") as f:
    id2label = json.load(f)
#id2label = {int(k):v for k,v in id2label.items()}
#label2id = {v: k for k, v in id2label.items()}
#num_labels = len(id2label)

"""orig_id2name = {
    0:  "Background",
    90:  "Road",
    75:  "Tree",
    113: "Low Vegetation",
    38: "Building",
    34: "Car",
    79: "Static Car",
    57: "Pedestrian"
}
"""
orig_id2name = {
    0:  "Background",
    19:  "Tree",
    8:  "Vegetation",
    15: "Person",
    13: "Fence",
    10: "Building",
    6: "Rock",
    3: "Grass"
}


# now build a new compact id2label for 0…5
new_id2label = {i: name for i, (_, name) in enumerate(orig_id2name.items())}
new_label2id = {v: k for k, v in new_id2label.items()}

orig2new = {orig_id: new_label2id[name]
            for orig_id, name in orig_id2name.items()}

processor = OneFormerProcessor.from_pretrained("shi-labs/oneformer_ade20k_swin_large")
model = OneFormerForUniversalSegmentation.from_pretrained(
    "shi-labs/oneformer_ade20k_swin_large",
    id2label=new_id2label,
    label2id=new_label2id, 
    num_labels = len(new_id2label),
    is_training=True,
    ignore_mismatched_sizes=True,
)
model.config.use_contrastive_loss = True
processor.image_processor.num_text = (model.config.num_queries - model.config.text_encoder_n_ctx)

dataset = CustomDataset(
    processor,
    #img_dir="uavid_train/seq1/Images",
    #mask_dir="uavid_train/seq1/Labels",
    img_dir="semantic_drone_dataset/original_images",
    mask_dir="semantic_drone_dataset/label_images_semantic",
    target_size=(512, 512),
    label2id=orig2new,        
)

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=default_data_collator)

batch = next(iter(dataloader))
for k, v in batch.items():
    if isinstance(v, torch.Tensor):
        print(k, v.shape)

# training
from torch.optim import AdamW

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
optimizer = AdamW(model.parameters(), lr=1e-5)
model.to(device)
model.train()
for epoch in range(5):
    for batch in dataloader:
        optimizer.zero_grad()
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        print("Loss:", loss.item())
        loss.backward()
        optimizer.step()
model.save_pretrained("model/model6")
processor.save_pretrained("model/model6")
