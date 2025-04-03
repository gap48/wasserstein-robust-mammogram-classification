import os
import sys
import argparse
import random
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader, random_split

import monai
from monai.transforms import (
    LoadImage,
    EnsureChannelFirst,
    ScaleIntensityRange,
    ToTensor,
    Compose,
    Resize,
    RandFlip,
    RandRotate,
    RandZoom,
    RandGaussianNoise
)

from tqdm import tqdm
from sklearn.metrics import classification_report
from typing import Optional, Tuple, Union

#########################################################
# 1) DATASET
#########################################################

class MammoMetadataDataset(Dataset):

    def __init__(self, metadata_df, transform=None):
        self.metadata_df = metadata_df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.metadata_df)

    def __getitem__(self, idx):
        row = self.metadata_df.iloc[idx]
        image_path = row['image_path']

        label_dict = {
            'mass_calc': row['label'],  
            'pathology': row['pathology'],
            'subtlety': row['subtlety'],
            'breast_density': row['breast_density'],
            'assessment': row['assessment'],
            'abnormality_type': row['abnormality_type']
        }

        if self.transform:
            img = self.transform(image_path)
        else:
            img = LoadImage(image_only=True)(image_path)
            img = torch.tensor(img[None, ...])

        return img, label_dict


def custom_collate(batch):
    images = torch.stack([item[0] for item in batch])
    labels = [item[1] for item in batch]

    collated_labels = {}
    if isinstance(labels[0], dict):
 
        for key in labels[0].keys():
            collated_labels[key] = [d[key] for d in labels]
        return images, collated_labels
    else:
   
        return images, labels

#########################################################
# 2) DATALOADER CREATION
#########################################################

def create_cls_dataloaders_full(metadata_df, batch_size=4, num_workers=2, seed=42):
    transform = Compose([
        LoadImage(image_only=True),
        EnsureChannelFirst(),
        ScaleIntensityRange(a_min=0, a_max=65535, b_min=0.0, b_max=1.0, clip=True),
        RandFlip(prob=0.5, spatial_axis=0),
        RandRotate(range_x=15, prob=0.5),
        RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.3),
        RandGaussianNoise(prob=0.2),
        Resize((224, 224)),
        ToTensor()
    ])

    dataset = MammoMetadataDataset(metadata_df, transform=transform)
    total_size = len(dataset)
    train_size = int(0.8 * total_size)
    val_size   = int(0.1 * total_size)
    test_size  = total_size - train_size - val_size

    train_ds, val_ds, test_ds = random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=custom_collate
    )
    return train_loader, val_loader, test_loader

#########################################################
# 3) WRM LOSS
#########################################################

class WRMLoss(nn.Module):

    def __init__(self, eps=0.3, steps=15):
        super().__init__()
        self.eps = eps
        self.steps = steps

    def forward(self, model, images, labels, criterion):
        x_adv = images.detach().clone().cuda()
        x_adv.requires_grad_(True)

      
        outputs = model(x_adv)
        loss = criterion(outputs, labels)
        grad = torch.autograd.grad(self.eps * loss, x_adv, create_graph=True)[0]
        x_adv = x_adv + grad.detach()

        for t in range(self.steps):
            x_adv = x_adv.detach().clone()
            x_adv.requires_grad_(True)

            outputs = model(x_adv)
            loss = criterion(outputs, labels)
            grad = torch.autograd.grad(self.eps * loss, x_adv, create_graph=True)[0]

            
            l2_loss = 0.5 * torch.sum((x_adv - images) ** 2)
            grad_reg = torch.autograd.grad(l2_loss, x_adv, create_graph=True)[0]

            grad = grad - grad_reg
            step_size = 1.0 / torch.sqrt(torch.tensor(t + 2.0, device=x_adv.device, dtype=torch.float))
            x_adv = x_adv + step_size * grad.detach()

    
        final_outputs = model(x_adv.detach())
        final_loss = criterion(final_outputs, labels)
        return final_loss, x_adv.detach()

#########################################################
# 4) UTILS: METADATA, CLASS WEIGHTS, LABEL CONVERSION
#########################################################
def find_full_mammogram(dcm_files, patient_folder):
    matching_files = []
    for dcm_path in dcm_files:
        if patient_folder in dcm_path and "full mammogram images" in dcm_path:
            matching_files.append(dcm_path)
    return matching_files[0] if matching_files else None

def process_dataframe(df, dcm_files, is_mass=True):
    prefix = "Mass-Training" if is_mass else "Calc-Training"
    density_col = 'breast_density' if is_mass else 'breast density'
    data = []

    for _, row in df.iterrows():
        patient_id = row['patient_id'].split('_')[1]
        breast = row['left or right breast']
        view = row['image view']

        folder_pattern = f"{prefix}_P_{patient_id}_{breast}_{view}"
        full_image_path = find_full_mammogram(dcm_files, folder_pattern)
        if full_image_path:
            item = {
                'image_path': full_image_path,
                'label': 1 if is_mass else 0,
                'pathology': row['pathology'],
                'subtlety': row['subtlety'],
                'breast_density': row[density_col],
                'assessment': row['assessment'],
                'abnormality_type': row['abnormality type'],
                'patient_id': row['patient_id'],
                'breast': breast,
                'view': view
            }
            if is_mass:
                item['mass_shape'] = row['mass shape']
                item['mass_margins'] = row['mass margins']
            else:
                item['calc_type'] = row['calc type']
                item['calc_distribution'] = row['calc distribution']
            data.append(item)
    return data

def parse_ddsm_metadata(root_dir, dcm_files):
    mass_train_csv = os.path.join(root_dir, 'mass_case_description_train_set.csv')
    mass_test_csv  = os.path.join(root_dir, 'mass_case_description_test_set.csv')
    calc_train_csv = os.path.join(root_dir, 'calc_case_description_train_set.csv')
    calc_test_csv  = os.path.join(root_dir, 'calc_case_description_test_set.csv')

    mass_train = pd.read_csv(mass_train_csv)
    mass_test  = pd.read_csv(mass_test_csv)
    calc_train = pd.read_csv(calc_train_csv)
    calc_test  = pd.read_csv(calc_test_csv)

    all_data = []
    all_data.extend(process_dataframe(mass_train, dcm_files, is_mass=True))
    all_data.extend(process_dataframe(mass_test,  dcm_files, is_mass=True))
    all_data.extend(process_dataframe(calc_train, dcm_files, is_mass=False))
    all_data.extend(process_dataframe(calc_test,  dcm_files, is_mass=False))

    metadata_df = pd.DataFrame(all_data)
    print(f"[INFO] After matching CSV and DICOM, found {len(metadata_df)} records.")
    return metadata_df

def convert_pathology_to_label(pathology_list):
    numeric_labels = []
    for p in pathology_list:
        if p.upper().startswith('MAL'):
            numeric_labels.append(1)
        else:
            numeric_labels.append(0)
    return torch.tensor(numeric_labels, dtype=torch.long)

def compute_class_weights(metadata_df):
    pathology_counts = metadata_df['pathology'].value_counts()
    benign_count = pathology_counts.get('BENIGN', 0) + pathology_counts.get('BENIGN_WITHOUT_CALLBACK', 0)
    malignant_count = pathology_counts.get('MALIGNANT', 0)

    benign_count = max(benign_count, 1)
    malignant_count = max(malignant_count, 1)
    total_samples = benign_count + malignant_count

    weight_benign    = total_samples / (2.0 * benign_count)
    weight_malignant = total_samples / (2.0 * malignant_count)
    return torch.tensor([weight_benign, weight_malignant], dtype=torch.float)

#########################################################
# 5) TRAIN-EVAL FUNCTION
#########################################################

def train_and_evaluate(
    model,
    train_loader,
    val_loader,
    test_loader,
    wrm_train=False,
    wrm_loss=None,
    class_weights=None,
    num_epochs=5,
    lr=1e-4,
    weight_decay=1e-5,
    output_dir='.',
    log_mode='console'
):
    if class_weights is None:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights).cuda()

    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    
    train_losses = []
    train_accs   = []

    val_clean_accs = []
    val_adv_accs   = []
    test_clean_accs= []
    test_adv_accs  = []

    best_val_clean_acc = 0.0

    for epoch in range(num_epochs):
        # ----------------------- TRAIN -----------------------
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        batch_pbar = tqdm(train_loader, 
                         desc=f"[Epoch {epoch+1}/{num_epochs}]", 
                         leave=False,
                         file=sys.stdout, 
                         mininterval=0.1)
        
        for imgs, label_dict in batch_pbar:
            imgs = imgs.cuda(non_blocking=True)
            labels = convert_pathology_to_label(label_dict['pathology']).cuda()

            optimizer.zero_grad()

            if wrm_train:
                loss, x_adv = wrm_loss(model, imgs, labels, criterion)
                with torch.no_grad():
                    preds = model(x_adv).argmax(dim=1)
            else:
                logits = model(imgs)
                loss = criterion(logits, labels)
                with torch.no_grad():
                    preds = logits.argmax(dim=1)

            loss.backward()
            optimizer.step()
            batch_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            epoch_loss += loss.item() * imgs.size(0)
            epoch_correct += (preds == labels).sum().item()
            epoch_total += imgs.size(0)

        avg_train_loss = epoch_loss / epoch_total if epoch_total else 0
        train_acc = epoch_correct / epoch_total if epoch_total else 0
        train_losses.append(avg_train_loss)
        train_accs.append(train_acc)

        # ----------------------- VALIDATION -----------------------
        model.eval()
   
        val_clean_correct = 0
        val_clean_total = 0
        with torch.no_grad():
            for imgs, ld in val_loader:
                imgs = imgs.cuda(non_blocking=True)
                lbls = convert_pathology_to_label(ld['pathology']).cuda()
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                val_clean_correct += (preds == lbls).sum().item()
                val_clean_total += imgs.size(0)
        val_clean_acc = val_clean_correct / val_clean_total if val_clean_total else 0
        val_clean_accs.append(val_clean_acc)

 
        val_adv_correct = 0
        val_adv_total = 0
        
        for imgs, ld in val_loader:
            imgs = imgs.cuda(non_blocking=True)
            lbls = convert_pathology_to_label(ld['pathology']).cuda()
            with torch.enable_grad():
                
                imgs.requires_grad_(True)
      
                _, x_adv = wrm_loss(model, imgs, lbls, criterion)
            with torch.no_grad():
                adv_preds = model(x_adv).argmax(dim=1)
            val_adv_correct += (adv_preds == lbls).sum().item()
            val_adv_total += imgs.size(0)
        val_adv_acc = val_adv_correct / val_adv_total if val_adv_total else 0
        val_adv_accs.append(val_adv_acc)

        # ----------------------- TEST -----------------------
        
        test_clean_correct = 0
        test_clean_total   = 0
        with torch.no_grad():
            for imgs, ld in test_loader:
                imgs = imgs.cuda(non_blocking=True)
                lbls = convert_pathology_to_label(ld['pathology']).cuda()
                logits = model(imgs)
                preds = logits.argmax(dim=1)
                test_clean_correct += (preds == lbls).sum().item()
                test_clean_total += imgs.size(0)
        test_clean_acc = test_clean_correct / test_clean_total if test_clean_total else 0
        test_clean_accs.append(test_clean_acc)

        test_adv_correct = 0
        test_adv_total   = 0
        for imgs, ld in test_loader:
            imgs = imgs.cuda(non_blocking=True)
            lbls = convert_pathology_to_label(ld['pathology']).cuda()
            with torch.enable_grad():
                imgs.requires_grad_(True)
                _, x_adv = wrm_loss(model, imgs, lbls, criterion)
            with torch.no_grad():
                adv_preds = model(x_adv).argmax(dim=1)
            test_adv_correct += (adv_preds == lbls).sum().item()
            test_adv_total += imgs.size(0)
        test_adv_acc = test_adv_correct / test_adv_total if test_adv_total else 0
        test_adv_accs.append(test_adv_acc)

        # ----------------------- LOG EPOCH -----------------------
        print(f"\n[Epoch {epoch+1}/{num_epochs}]"
              f" Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f} || "
              f"Val Clean Acc: {val_clean_acc:.4f}, Val Adv Acc: {val_adv_acc:.4f} || "
              f"Test Clean Acc: {test_clean_acc:.4f}, Test Adv Acc: {test_adv_acc:.4f}")

       
        if val_clean_acc > best_val_clean_acc:
            best_val_clean_acc = val_clean_acc
            best_ckpt = os.path.join(output_dir, "best_checkpoint_wrm.pth")
            torch.save(model.state_dict(), best_ckpt)
            print(f"  [*] New best Val Clean Acc: {best_val_clean_acc:.4f}")

    # ----------------------- PLOTS -----------------------
    epochs_range = range(1, num_epochs+1)

    plt.figure()
    plt.plot(epochs_range, val_clean_accs, label='Val Clean Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation (Clean) Accuracy')
    plt.legend()
    val_clean_path = os.path.join(output_dir, 'val_clean_accuracy.png')
    plt.savefig(val_clean_path)
    plt.show()

    plt.figure()
    plt.plot(epochs_range, val_adv_accs, label='Val Adv Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Validation (Adversarial) Accuracy')
    plt.legend()
    val_adv_path = os.path.join(output_dir, 'val_adv_accuracy.png')
    plt.savefig(val_adv_path)
    plt.show()

    plt.figure()
    plt.plot(epochs_range, test_clean_accs, label='Test Clean Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test (Clean) Accuracy')
    plt.legend()
    test_clean_path = os.path.join(output_dir, 'test_clean_accuracy.png')
    plt.savefig(test_clean_path)
    plt.show()

    plt.figure()
    plt.plot(epochs_range, test_adv_accs, label='Test Adv Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test (Adversarial) Accuracy')
    plt.legend()
    test_adv_path = os.path.join(output_dir, 'test_adv_accuracy.png')
    plt.savefig(test_adv_path)
    plt.show()


#########################################################
# 7) VISUALIZATION: ATTENTION & GRAD-CAM
#########################################################

def get_attention_block(model):
    if hasattr(model, 'layers'):  
        return model.layers[-1].blocks[-1]
    elif hasattr(model, 'encoder'):  
        return model.encoder.layers[-1].blocks[-1]
    return None

def process_attention_weights(attn_output, img_shape):
    attn = attn_output.detach().cpu()
    feature_magnitudes = torch.norm(attn[0], dim=-1)  # [7, 7]
    attention_map = (feature_magnitudes - feature_magnitudes.min()) / \
                    (feature_magnitudes.max() - feature_magnitudes.min() + 1e-8)
    attention_map = attention_map.numpy()
    attention_map = F.interpolate(
        torch.from_numpy(attention_map).unsqueeze(0).unsqueeze(0).float(),
        size=img_shape[-2:],
        mode='bicubic',
        align_corners=False
    ).squeeze().numpy()

    return attention_map

def get_random_sample(dataloader):
    dataset = dataloader.dataset
    random_idx = random.randint(0, len(dataset) - 1)
    image, labels = dataset[random_idx]
    if isinstance(labels, dict):
        labels = {k: [v] for k, v in labels.items()}
    else:
        labels = {'pathology': [labels]}

    return image.unsqueeze(0).cuda(), labels

def visualize_attention(model, dataloader, output_dir='.', log_mode='console'):
    model = model.cuda()

    img, labels = get_random_sample(dataloader)
    true_label = labels['pathology'][0]
    attn_block = get_attention_block(model)
    if attn_block is None:
        print("No attention block found - skipping attention visualization")
        return
    attention_values = []

    def hook_fn(module, input, output):
        attention_values.append(output)
    handle = attn_block.register_forward_hook(hook_fn)

    model.eval()
    with torch.no_grad():
        outputs = model(img)
        pred_class = outputs.argmax(dim=1).item()
    handle.remove()

    if not attention_values:
        print("No attention values captured")
        return
    attention_map = process_attention_weights(attention_values[0], img.shape)
    if attention_map is None:
        print("Could not process attention weights")
        return

    fig = plt.figure(figsize=(22, 7))
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    img_np = img.squeeze().cpu().numpy()
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title(f"Original Image\nLabel: {true_label}")
    ax1.axis('off')

    attention_display = ax2.imshow(attention_map, cmap='inferno')
    ax2.set_title("Raw Attention Map")
    ax2.axis('off')
    plt.colorbar(attention_display, ax=ax2, fraction=0.046, pad=0.04)

    ax3.imshow(img_np, cmap='gray')
    overlay = ax3.imshow(attention_map, cmap='inferno', alpha=0.5)
    ax3.set_title(f"Attention Overlay\nPredicted: {'Malignant' if pred_class else 'Benign'}")
    ax3.axis('off')
    plt.colorbar(overlay, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "attention_map.png")

    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.2)
    if log_mode == "console":
        plt.show()
    else:
        plt.close()


################### GradCAM ###################

def get_target_layer(model):
    target_layer = None
    if hasattr(model, 'layers'):  
        last_stage = model.layers[-2]
        if hasattr(last_stage, 'blocks'):
            target_layer = last_stage.blocks[-1].attn
        return target_layer
    elif hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):  
        last_stage = model.encoder.layers[-2]
        if hasattr(last_stage, 'blocks'):
            target_layer = last_stage.blocks[-1].attn
        return target_layer
    elif hasattr(model, 'blocks'):
        target_layer = model.blocks[-2].attn
        return target_layer

  
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            conv_layers.append(module)

    if conv_layers:
        target_layer = conv_layers[-2] if len(conv_layers) > 1 else conv_layers[-1]

    if target_layer is None:
        raise ValueError("Could not find suitable target layer")

    return target_layer

class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: Optional[torch.nn.Module] = None):
        self.model = model
        self.feature_maps = None
        self.gradient = None

        self.target_layer = target_layer if target_layer is not None else get_target_layer(model)
        if self.target_layer is None:
            raise ValueError("Could not find suitable target layer for GRAD-CAM")

        self.hooks = []
        self._register_hooks()

        self.is_transformer = self._check_if_transformer()

    def _check_if_transformer(self) -> bool:
        return any(
            hasattr(self.model, attr)
            for attr in ['blocks', 'encoder', 'transformer', 'attention']
        )

    def _register_hooks(self):
        def forward_hook(module, input, output):
            if isinstance(output, tuple):
                self.feature_maps = output[0]  
            else:
                self.feature_maps = output

        def full_backward_hook(module, grad_input, grad_output):
            self.gradient = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(full_backward_hook))

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()

    def _reshape_transform(self, tensor: torch.Tensor) -> torch.Tensor:

        if not self.is_transformer:
            return tensor

        if len(tensor.shape) == 4:  
            return tensor

        result = tensor
        if len(tensor.shape) == 3:
            num_patches = tensor.shape[1]
            h = w = int(np.sqrt(num_patches))
            result = tensor.reshape(tensor.shape[0], h, w, -1)
            result = result.permute(0, 3, 1, 2)

        return result

    def generate_cam(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        reshape_transform: bool = True
    ) -> Tuple[np.ndarray, int]:
  
        self.model.eval()
        input_tensor.requires_grad = True


        model_output = self.model(input_tensor)
        if isinstance(model_output, tuple):
            model_output = model_output[0]

        predicted_class = model_output.argmax(dim=1).item()
        target_class = predicted_class if target_class is None else target_class

        self.model.zero_grad()

        target_score = model_output[0][target_class]
        target_score.backward(retain_graph=True)

        gradients = self.gradient.detach()  
        attention_maps = self.feature_maps.detach()  

        print(f"Gradients shape: {gradients.shape}")
        print(f"Attention maps shape: {attention_maps.shape}")
        print(f"Gradient stats - mean: {gradients.mean().item()}, std: {gradients.std().item()}")
        print(f"Gradient range: {gradients.min().item()} to {gradients.max().item()}")

        gradients = 2 * (gradients - gradients.min()) / (gradients.max() - gradients.min()) - 1

        B, N, D = attention_maps.shape

        attention_scores = torch.matmul(attention_maps, attention_maps.transpose(-2, -1)) 
        attention_scores = F.softmax(attention_scores / torch.sqrt(torch.tensor(D).float()), dim=-1)
        attention_scores = attention_scores.mean(dim=0)  
        grad_weights = torch.norm(gradients, dim=2)  
        grad_weights = F.softmax(grad_weights.mean(dim=0), dim=0)  

        cam = attention_scores * grad_weights.view(-1, 1)  
        cam = cam.mean(dim=0) 
        size = int(np.sqrt(N))
        cam = cam.view(size, size)

        cam = F.relu(cam)  
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

            cam = torch.pow(cam, 0.5)

            cam = (cam - cam.min()) / (cam.max() - cam.min())
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),
            size=input_tensor.shape[-2:],
            mode='bicubic',
            align_corners=False
        ).squeeze()

        return cam.detach().cpu().numpy(), predicted_class

def process_gradcam(cam: np.ndarray, img_shape: Tuple[int, ...]) -> np.ndarray:
    cam_resized = F.interpolate(
        torch.from_numpy(cam).unsqueeze(0).unsqueeze(0).float(),
        size=img_shape[-2:],
        mode='bicubic',
        align_corners=False
    ).squeeze().numpy()

    return cam_resized

def visualize_gradcam(model, dataloader, output_dir='.', log_mode='console', target_layer=None):
    model = model.cuda()
    model.eval()

    img, labels = get_random_sample(dataloader)
    img_for_viz = img.clone().detach()
    true_label = labels['pathology'][0]
    grad_cam = GradCAM(model, target_layer)
    cam, pred_class = grad_cam.generate_cam(img)
    cam_processed = process_gradcam(cam, img.shape)

    
    fig = plt.figure(figsize=(22, 7))
    gs = plt.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.3)
    
    ax1 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[1])
    ax3 = plt.subplot(gs[2])

    img_np = img_for_viz.squeeze().cpu().numpy()
    ax1.imshow(img_np, cmap='gray')
    ax1.set_title(f"Original Image\nLabel: {true_label}")
    ax1.axis('off')

    gradcam_display = ax2.imshow(cam_processed, cmap='RdYlBu_r', vmin=0, vmax=1)
    ax2.set_title("GRAD-CAM Heatmap")
    ax2.axis('off')
    plt.colorbar(gradcam_display, ax=ax2, fraction=0.046, pad=0.04)

    ax3.imshow(img_np, cmap='gray')
    overlay = ax3.imshow(cam_processed, cmap='RdYlBu_r', alpha=0.7, vmin=0, vmax=1)
    ax3.set_title(f"GRAD-CAM Overlay\nPredicted: {'Malignant' if pred_class else 'Benign'}")
    ax3.axis('off')
    plt.colorbar(overlay, ax=ax3, fraction=0.046, pad=0.04)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "gradcam_map.png")
  
    plt.savefig(plot_path, bbox_inches='tight', pad_inches=0.2)
    if log_mode == "console":
        plt.show()
    else:
        plt.close()
    grad_cam.remove_hooks()

#########################################################
# 7) MAIN
#########################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, required=True)
    parser.add_argument('--dicom_path', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./outputs')
    parser.add_argument('--cls_epochs', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--train_cls', action='store_true')
    parser.add_argument('--WRM_train', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--log_mode', type=str, choices=['console','file'], default='console')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    if args.log_mode == 'file':
        log_path = os.path.join(args.output_dir, 'training_log.txt')
        sys.stdout = open(log_path, 'w', buffering=1)
        matplotlib.use('Agg')
        print(f"[INFO] Logging to file: {log_path}")

 
    if not os.path.isfile(args.dicom_path):
        print(f"[ERROR] Missing DICOM path file: {args.dicom_path}")
        sys.exit(1)
    with open(args.dicom_path, 'r') as f:
        dcm_files = [line.strip() for line in f]
    print(f"[INFO] Found {len(dcm_files)} DICOM files.")


    metadata_df = parse_ddsm_metadata(args.root_dir, dcm_files)
    if len(metadata_df) == 0:
        print("[ERROR] metadata_df is empty after parse. Exiting.")
        sys.exit(1)
    print(f"[INFO] Final matched records: {len(metadata_df)}")

    train_loader, val_loader, test_loader = create_cls_dataloaders_full(
        metadata_df,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )

 
    import timm
    model = timm.create_model(
        "swin_base_patch4_window7_224",
        pretrained=True,
        in_chans=1,
        num_classes=2
    ).cuda()

    class_w = compute_class_weights(metadata_df).cuda()
    wrm_criterion = WRMLoss(eps=0.3, steps=15)

    if args.train_cls:
        train_and_evaluate(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            wrm_train=args.WRM_train,
            wrm_loss=wrm_criterion,
            class_weights=class_w,
            num_epochs=args.cls_epochs,
            lr=1e-4,
            weight_decay=1e-5,
            output_dir=args.output_dir,
            log_mode=args.log_mode
        )
    else:
        best_ckpt = os.path.join(args.output_dir, 'best_checkpoint_wrm.pth')
        if os.path.exists(best_ckpt):
            model.load_state_dict(torch.load(best_ckpt))
            print(f"[INFO] Loaded checkpoint: {best_ckpt}")
        else:
            print("[INFO] No checkpoint found. Using pretrained model weights as-is.")

    if args.visualize:
        print("[INFO] Visualizing attention and Grad-CAM ...")
        visualize_attention(model, test_loader,output_dir=args.output_dir,log_mode=args.log_mode)
        visualize_gradcam(model, test_loader,output_dir=args.output_dir,log_mode=args.log_mode)

    if args.log_mode == 'file':
        print("[INFO] Finished all tasks.")
        sys.stdout.close()


if __name__ == "__main__":
    main()
