"""Resnet backbone with added linear layers"""
import os
import timm
import pandas as pd
from PIL import Image
# import matplotlib.image as mpimg
# from matplotlib import pyplot as plt

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchmetrics.regression import R2Score
from torch.optim.lr_scheduler import OneCycleLR

import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

# # Set a seed for PyTorch
# seed_value = 42
# torch.manual_seed(seed_value)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed(seed_value)

# # Set a seed for NumPy (if you're using NumPy alongside PyTorch)
# np.random.seed(seed_value)

class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    # model_name = 'maxvit_tiny_tf_384.in1k'  # Name of pretrained classifier
    model_name = 'resnet50.a1_in1k'  # Name of pretrained classifier
    # model_name = 'resnet50d'  # Name of pretrained classifier
    # model_name = 'efficientnet_b3.ra2_in1k'  # Name of pretrained classifier
    image_size = 384  # Input image size
    epochs = 12 # Training epochs
    batch_size = 8  # Batch size
    lr = 1e-4
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6 # Number of classes in the dataset
    num_folds = 5 # Number of folds to split the dataset
    fold = 0 # Which fold to set as validation data
    class_names = ['X4_mean', 'X11_mean', 'X18_mean',
                   'X26_mean', 'X50_mean', 'X3112_mean',]
    aux_class_names = list(map(lambda x: x.replace("mean","sd"), class_names))
    num_classes = len(class_names)
    aux_num_classes = len(aux_class_names)

BASE_PATH = '../planttraits2024/'

# print(f'{CFG.model_name}/sample_submission.csv')

# Train + Valid
df = pd.read_csv(f'{BASE_PATH}/train.csv')
df['image_path'] = f'{BASE_PATH}/train_images/'+df['id'].astype(str)+'.jpeg'
df.loc[:, CFG.aux_class_names] = df.loc[:, CFG.aux_class_names].fillna(-1)
print(df.head(2))

# Test
test_df = pd.read_csv(f'{BASE_PATH}/test.csv')
test_df['image_path'] = f'{BASE_PATH}/test_images/'+test_df['id'].astype(str)+'.jpeg'
FEATURE_COLS = test_df.columns[1:-1].tolist()
print(test_df.head(2))

class PlantDataset(Dataset):
    def __init__(self, paths, features, labels=None, aux_labels=None, transform=None, augment=False):
        self.paths = paths
        self.features = features
        self.labels = labels
        self.aux_labels = aux_labels
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        feature = self.features[idx]

        # Read and decode image
        image = self.decode_image(path)

        # Apply augmentations
        if self.augment:
            augmented = self.transform(image=image)
            image = augmented['image']            
        else:
            # Ensure channel dimension is the first one
            image = np.transpose(image, (2, 0, 1))
            image = torch.tensor(image, dtype=torch.float32)


        if self.labels is not None:
            label = torch.tensor(self.labels[idx])
            aux_label = torch.tensor(self.aux_labels[idx])
            return {'images': image, 'features': feature}, (label, aux_label)
        else:
            return {'images': image, 'features': feature}

    def decode_image(self, path):
        image = Image.open(path)
        image = image.resize((CFG.image_size,CFG.image_size))
        image = np.array(image)
        return image

def build_augmenter():
    # Define Albumentations augmentations
    transform = A.Compose([
        A.RandomBrightness(limit=0.1, always_apply=False, p=0.5),
        A.RandomContrast(limit=0.1, always_apply=False, p=0.5),
        A.HueSaturationValue(hue_shift_limit=0.1, sat_shift_limit=0.1, val_shift_limit=0.1, p=0.5),
        A.Cutout(num_holes=1, max_h_size=32, max_w_size=32, always_apply=False, p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=5, p=0.5),
        ToTensorV2(),
    ])

    return transform

def build_dataset(paths, features, labels=None, aux_labels=None, batch_size=32, cache=True, augment=True, repeat=True, shuffle=1024, cache_dir="", drop_remainder=False):
    dataset = PlantDataset(paths, features, labels, aux_labels, transform=build_augmenter(), augment=augment)

    if cache_dir != "" and cache:
        os.makedirs(cache_dir, exist_ok=True)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4, drop_last=drop_remainder, pin_memory=True)

    return dataloader

# Assuming df is your dataframe containing file paths, features, labels, and fold information
skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=42)

# Create separate bin for each trait
for i, trait in enumerate(CFG.class_names):
    bin_edges = np.percentile(df[trait], np.linspace(0, 100, CFG.num_folds + 1))
    df[f"bin_{i}"] = np.digitize(df[trait], bin_edges)

df["final_bin"] = df[[f"bin_{i}" for i in range(len(CFG.class_names))]].astype(str).agg("".join, axis=1)

df["fold"] = -1  # Initialize fold column

# Perform the stratified split using final bin
for fold, (train_idx, valid_idx) in enumerate(skf.split(df, df["final_bin"])):
    df.loc[valid_idx, "fold"] = fold

# Sample from full data
sample_df = df.copy()
train_df = sample_df[sample_df.fold != CFG.fold]
valid_df = sample_df[sample_df.fold == CFG.fold]
print(f"# Num Train: {len(train_df)} | Num Valid: {len(valid_df)}")

# Normalize features
scaler = StandardScaler()
train_features = scaler.fit_transform(train_df[FEATURE_COLS].values)
valid_features = scaler.transform(valid_df[FEATURE_COLS].values)

# Extract file paths, features, labels, and fold information for train and validation sets
train_paths = train_df.image_path.values
train_labels = train_df[CFG.class_names].values
train_aux_labels = train_df[CFG.aux_class_names].values

valid_paths = valid_df.image_path.values
valid_labels = valid_df[CFG.class_names].values
valid_aux_labels = valid_df[CFG.aux_class_names].values

# Build datasets
train_dataloader = build_dataset(train_paths, train_features, train_labels, train_aux_labels,
                         batch_size=CFG.batch_size,
                         repeat=True, shuffle=True, augment=True, cache=False)
valid_dataloader = build_dataset(valid_paths, valid_features, valid_labels, valid_aux_labels,
                         batch_size=CFG.batch_size,
                         repeat=False, shuffle=False, augment=False, cache=False)

class CustomModel(nn.Module):
    def __init__(self, num_classes, aux_num_classes, feature_cols, model_name):
        super(CustomModel, self).__init__()

        # Define input layers
        self.img_input = nn.Identity()
        self.feat_input = nn.Identity()

        # Load pre-trained model
        self.backbone = timm.create_model(model_name, 
                                          pretrained=True, 
                                        #   num_classes=num_classes, 
                                          global_pool='avg'
                                          )
        
        # Adapt the model to match the expected output size
        # self.backbone.global_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone.classifier = nn.Identity()

        self.dropout_img = nn.Dropout(0.2)

        # Branch for tabular/feature input
        self.dense1 = nn.Linear(len(feature_cols), 512)
        self.dense2 = nn.Linear(512, 256)
        self.dense3 = nn.Linear(256, 128)
        self.dense4 = nn.Linear(128, 64)
        self.dense5 = nn.Linear(64, 64)
        self.dropout_feat = nn.Dropout(0.1)

        # Output layer
        self.head1 = nn.Linear(1064, 128)
        self.head2 = nn.Linear(128, num_classes)
        self.aux_head1 = nn.Linear(1064, 128)
        self.aux_head2 = nn.Linear(128, aux_num_classes)

        print(f'Model Name: {model_name}')
        print(f'Model pooling: {self.backbone.global_pool}')
        print(f'Model classifier: {self.backbone.get_classifier()}')

    def forward(self, img, feat):
        """Forward pass"""
        # Image branch
        x1 = self.backbone(img)
        x1 = self.dropout_img(x1.flatten(1))

        # Feature branch
        x2 = F.selu(self.dense1(self.feat_input(feat)))
        x2 = F.selu(self.dense2(x2))
        x2 = self.dropout_feat(x2)

        x2 = F.selu(self.dense3(x2))
        x2 = F.selu(self.dense4(x2))
        x2 = self.dropout_feat(x2)

        x2 = F.selu(self.dense5(x2))
        x2 = self.dropout_feat(x2)

        # Concatenate both branches
        concat = torch.cat([x1, x2], dim=1)
        # Output layer
        out1 = F.relu(self.head1(concat))
        out1 = self.head2(out1)

        out2 = F.relu(self.aux_head1(concat))
        out2 = self.aux_head2(out2)

        return {'head': out1, 'aux_head': out2}


# Instantiate the model with the desired EfficientNetV2 model_name
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = CustomModel(CFG.num_classes, CFG.aux_num_classes, FEATURE_COLS, model_name=CFG.model_name)
model.to(device)
# Print the model summary
# print(model)

class R2Loss(nn.Module):
    def __init__(self, use_mask=False):
        super(R2Loss, self).__init__()
        self.use_mask = use_mask

    def forward(self, y_true, y_pred):
        if self.use_mask:
            mask = (y_true != -1)
            y_true = torch.where(mask, y_true, torch.zeros_like(y_true))
            y_pred = torch.where(mask, y_pred, torch.zeros_like(y_pred))

        SS_res = torch.sum((y_true - y_pred)**2, dim=0)  # (B, C) -> (C,)
        SS_tot = torch.sum((y_true - torch.mean(y_true, dim=0))**2, dim=0)  # (B, C) -> (C,)
        r2_loss = SS_res / (SS_tot + 1e-6)  # (C,)
        return torch.mean(r2_loss)  # ()

class R2Metric(nn.Module):
    def __init__(self, num_classes=6):
        super(R2Metric, self).__init__()
        self.num_classes = num_classes
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        self.SS_res = torch.zeros(self.num_classes, device=self.device)
        self.SS_tot = torch.zeros(self.num_classes, device=self.device)
        self.num_samples = torch.tensor(0, dtype=torch.float32, device=self.device)

    def forward(self, y_true, y_pred):
        y_true = y_true.to(self.device)
        y_pred = y_pred.to(self.device)

        SS_res = torch.sum((y_true - y_pred)**2, dim=0)
        SS_tot = torch.sum((y_true - torch.mean(y_true, dim=0))**2, dim=0)
        self.SS_res += SS_res
        self.SS_tot += SS_tot
        self.num_samples += y_true.size(0)

    def compute(self):
        r2 = 1 - self.SS_res / (self.SS_tot + 1e-6)
        return torch.mean(r2)

def get_lr_scheduler(optimizer, dataloader, epochs, max_lr, steps_per_epoch, pct_start=0.3, div_factor=25.0, final_div_factor=1e4):
    total_steps = steps_per_epoch * epochs
    scheduler = OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor)

    def lr_lambda(step):
        return scheduler.get_lr()

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Define loss functions and metrics
criterion_head = R2Loss(use_mask=False)
criterion_aux_head = R2Loss(use_mask=True)

# Loss weights
weight_head = 1.0
weight_aux_head = 0.3

# Model checkpoint
best_model_path = f'{CFG.model_name}_best_model2_2.pth'
best_r2_score = -float('inf')
optimizer = torch.optim.Adam(model.parameters(), lr=CFG.lr)

# Initialize the R2Score metric
metric_head = R2Score(num_outputs=6, multioutput='uniform_average').to(device)

# Training loop
for epoch in range(CFG.epochs):
    # Training
    model.train()
    total_train_r2 = 0.0
    train_batches = 0
    epoch_avg_list = []

    for batch_idx, (inputs_dict, (targets, aux_targets)) in enumerate(tqdm(train_dataloader)):
        optimizer.zero_grad()

        # Move data to GPU
        inputs_images = inputs_dict['images'].to(device, dtype=torch.float32)
        inputs_features = inputs_dict['features'].to(device, dtype=torch.float32)
        targets = targets.to(device, dtype=torch.float32)
        aux_targets = aux_targets.to(device, dtype=torch.float32)

        # Forward pass
        outputs = model(inputs_images, inputs_features)
        batch_avg = torch.mean(torch.abs(outputs['head'] - targets) / targets, dim=0)
        epoch_avg_list.append(batch_avg)

        # Compute losses
        loss_head = weight_head * criterion_head(outputs['head'], targets)
        loss_aux_head = weight_aux_head * criterion_aux_head(outputs['aux_head'], aux_targets)
        loss = loss_head + loss_aux_head

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update the R2 metric
        r2_value = metric_head(outputs['head'], targets)
        total_train_r2 += r2_value.item()
        train_batches += 1

    # Compute the average training R2 score
    avg_train_r2 = total_train_r2 / train_batches
    print(f"Epoch {epoch + 1}/{CFG.epochs}, Average Train R2 Score: {avg_train_r2}")
    
    # Print the averages along with trait names
    epoch_avg = torch.mean(torch.stack(epoch_avg_list, dim=0), dim=0)
    trait_names = ["X4", "X11", "X18", "X26", "X50", "X3112"]
    print("Trait accuracies for training")
    for i, avg in enumerate(epoch_avg):
        print(f"{trait_names[i]}: {avg.item()}")

    # Validation
    model.eval()
    total_val_r2 = 0.0
    val_batches = 0
    epoch_avg_list = []

    with torch.no_grad():
        for val_batch_idx, (val_inputs_dict, (val_targets, val_aux_targets)) in enumerate(tqdm(valid_dataloader)):
            val_inputs_images = val_inputs_dict['images'].to(device, dtype=torch.float32)
            val_inputs_features = val_inputs_dict['features'].to(device, dtype=torch.float32)
            val_targets = val_targets.to(device, dtype=torch.float32)
            val_aux_targets = val_aux_targets.to(device, dtype=torch.float32)

            val_outputs = model(val_inputs_images, val_inputs_features)
            batch_avg = torch.mean(torch.abs(val_outputs['head'] - val_targets) / val_targets, dim=0)
            epoch_avg_list.append(batch_avg)

            # Compute the R2 metric for validation
            r2_value = metric_head(val_outputs['head'], val_targets)
            total_val_r2 += r2_value.item()
            val_batches += 1

    # Compute the average validation R2 score
    avg_val_r2 = total_val_r2 / val_batches
    print(f"Epoch {epoch + 1}/{CFG.epochs}, Average Val R2 Score: {avg_val_r2}")

    # Print the averages along with trait names
    epoch_avg = torch.mean(torch.stack(epoch_avg_list, dim=0), dim=0)
    trait_names = ["X4", "X11", "X18", "X26", "X50", "X3112"]
    print("Trait accuracies for training")
    for i, avg in enumerate(epoch_avg):
        print(f"{trait_names[i]}: {avg.item()}")

    # Save the best model based on validation R2 score
    if avg_val_r2 > best_r2_score:
        best_r2_score = avg_val_r2
        torch.save(model.state_dict(), best_model_path)

model.load_state_dict(torch.load(best_model_path))

# Test
test_paths = test_df.image_path.values
test_features = scaler.transform(test_df[FEATURE_COLS].values) 
test_ds = build_dataset(test_paths, test_features, batch_size=CFG.batch_size,
                         repeat=False, shuffle=False, augment=False, cache=False)

model.eval()  # Set the model to evaluation mode

# List to store predictions
all_predictions = []

# Iterate over batches in the test data loader
for batch_idx, inputs_dict in enumerate(tqdm(test_ds, desc='Testing')):
    # Extract images and features from the inputs_dict
    inputs_images = inputs_dict['images'].to(device, dtype=torch.float32)  # Assuming 'device' is the target device
    inputs_features = inputs_dict['features'].to(device, dtype=torch.float32)

    # Forward pass
    with torch.no_grad():
        outputs = model(inputs_images, inputs_features)

    # Get predictions
    predictions = outputs['head'].cpu().numpy()  # Assuming 'head' is the main task output

    # Append predictions to the list
    all_predictions.append(predictions)

# Concatenate predictions for all batches
all_predictions = np.concatenate(all_predictions, axis=0)

pred_df = test_df[["id"]].copy()
target_cols = [x.replace("_mean","") for x in CFG.class_names]
pred_df[target_cols] = all_predictions.tolist()

sub_df = pd.read_csv(f'{BASE_PATH}/sample_submission.csv')
sub_df = sub_df[["id"]].copy()
sub_df = sub_df.merge(pred_df, on="id", how="left")
sub_df.to_csv(f"{CFG.model_name}.csv", index=False)
sub_df.head()