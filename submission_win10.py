"""Resnet backbone with added linear layers"""
import os
import timm
import pandas as pd
from PIL import Image

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import warnings
import albumentations as A
from albumentations.pytorch import ToTensorV2

warnings.filterwarnings("ignore")

class CFG:
    verbose = 1  # Verbosity
    seed = 42  # Random seed
    model_name = 'resnet50.a1_in1k'  # Name of pretrained classifier
    image_size = 384  # Input image size
    epochs = 12 # Training epochs
    batch_size = 8  # Batch size
    lr = 1e-4
    drop_remainder = True  # Drop incomplete batches
    num_classes = 6 # Number of classes in the dataset
    num_folds = 5 # Number of folds to split the dataset
    fold = 0 # Which fold to set as validation data
    class_names = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean',]
    aux_class_names = list(map(lambda x: x.replace("mean","sd"), class_names))
    num_classes = len(class_names)
    aux_num_classes = len(aux_class_names)

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

# Original base
class CustomModel(nn.Module):
    def __init__(self, num_classes, aux_num_classes, feature_cols, model_name, lin_lay = 1600):
        super(CustomModel, self).__init__()

        # Define input layers
        self.img_input = nn.Identity()
        self.feat_input = nn.Identity()

        # Load pre-trained model
        self.backbone = timm.create_model(model_name, 
                                          pretrained=True, 
                                          global_pool='avg'
                                          )
        
        # Adapt the model to match the expected output size
        # self.backbone.global_pool = nn.AdaptiveAvgPool2d(1)
        self.backbone.classifier = nn.Identity()

        self.dropout_img = nn.Dropout(0.2)

        # Branch for tabular/feature input
        self.dense1 = nn.Linear(len(feature_cols), 326)
        self.dense2 = nn.Linear(326, 64)
        self.dropout_feat = nn.Dropout(0.1)

        # Output layer
        self.head = nn.Linear(lin_lay, num_classes)
        self.aux_head = nn.Linear(lin_lay, aux_num_classes)

        print(f'Model Name: {model_name}')
        print(f'Model pooling: {self.backbone.global_pool}')
        print(f'Model classifier: {self.backbone.get_classifier()}')

    def forward(self, img, feat):
        # Image branch
        x1 = self.backbone(img)
        x1 = self.dropout_img(x1.flatten(1))

        # Feature branch
        x2 = F.selu(self.dense1(self.feat_input(feat)))
        x2 = F.selu(self.dense2(x2))
        x2 = self.dropout_feat(x2)

        # Concatenate both branches
        concat = torch.cat([x1, x2], dim=1)
        # Output layer
        out1 = self.head(concat)
        out2 = F.relu(self.aux_head(concat))

        return {'head': out1, 'aux_head': out2}
  
# resnet_v2
class CustomModel2(nn.Module):
    def __init__(self, num_classes, aux_num_classes, feature_cols, model_name):
        super(CustomModel2, self).__init__()

        # Define input layers
        self.img_input = nn.Identity()
        self.feat_input = nn.Identity()

        # Load pre-trained model
        self.backbone = timm.create_model(model_name, pretrained=True, global_pool='avg')
        
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

        # print(f'Model Name: {model_name}')
        # print(f'Model pooling: {self.backbone.global_pool}')
        # print(f'Model classifier: {self.backbone.get_classifier()}')

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

  
if __name__ == '__main__':

    BASE_PATH = 'planttraits2024/'

    # Test
    test_df = pd.read_csv(f'{BASE_PATH}/test.csv')
    test_df['image_path'] = f'{BASE_PATH}/test_images/'+test_df['id'].astype(str)+'.jpeg'
    FEATURE_COLS = test_df.columns[1:-1].tolist()
    print(test_df.head(2))  

    # Assuming df is your dataframe containing file paths, features, labels, and fold information
    skf = StratifiedKFold(n_splits=CFG.num_folds, shuffle=True, random_state=42)

    # Instantiate the model with the desired EfficientNetV2 model_name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # resnet50.a1_in1k IMPROVED
    model = CustomModel2(CFG.num_classes, CFG.aux_num_classes, FEATURE_COLS, model_name='resnet50.a1_in1k')
    out_name = "resnet_V2.csv"
    model_weights = 'resnet50.a1_in1k_best_model2_2.pth'

    # resnet50.a1_in1k
    # model = CustomModel(CFG.num_classes, CFG.aux_num_classes, FEATURE_COLS, model_name='resnet50.a1_in1k')
    # out_name = "resnet.csv"
    # model_weights = 'resnet50.a1_in1k_best_model2V2.pth'

    # efficientnet_b3.ra2_in1k
    # model = CustomModel(CFG.num_classes, CFG.aux_num_classes, FEATURE_COLS, model_name='efficientnet_b3.ra2_in1k')
    # out_name = "efficientnet.csv"
    # model_weights = 'efficientnet_b3.ra2_in1k_best_model2.pth'

    # maxvit_tiny_tf_384.in1k
    # model = CustomModel(CFG.num_classes, CFG.aux_num_classes, FEATURE_COLS, model_name='maxvit_tiny_tf_384.in1k', lin_lay=1064)
    # out_name = "maxvit.csv"
    # model_weights = 'maxvit_tiny_tf_384.in1k_best_model.pth'
    
    model.to(device)
    model.load_state_dict(torch.load(model_weights, map_location=device))

    # Test
    test_paths = test_df.image_path.values
    scaler = StandardScaler()
    test_features = scaler.fit_transform(test_df[FEATURE_COLS].values) 
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

    sub_df = pd.read_csv(f'sample_sub.csv')
    sub_df = sub_df[["id"]].copy()
    sub_df = sub_df.merge(pred_df, on="id", how="left")
    sub_df.to_csv(out_name, index=False)
    sub_df