# Vision Transformer-Based BMR Detection - Thesis Project

This repository contains the complete implementation of my thesis project titled:

**“A Vision Transformer-Based Assistive System for Dermatological Diagnosis in Systemic Lupus Erythematosus”**

The goal of this work is to classify facial images as either **Butterfly Malar Rash (BMR)** or **non-BMR rashes**, supporting early diagnosis of SLE using Vision Transformers (ViTs) and CNNs.

# Models Directory

This directory contains the organized structure of all trained deep learning models evaluated in the research project. The models are categorized based on their architecture (CNN or Vision Transformer) and training strategies (baseline, after augmentation, and with CutMix).

## Folder Structure

```
Models/
├── CNNs_Before_Augmentation/
│   ├── DensNet121_before_aug
│   ├── MobileNet_V2_before_aug
│   ├── ResNet_50_before_aug
│   ├── VGG_16_before_aug
│   └── Xception_before_aug
├── CNNs_After_Augmentation/
│   ├── DensNet121_after_aug
│   ├── MobileNet_V2_after_aug
│   ├── ResNet_50_after_aug
│   ├── VGG_16_after_aug
│   └── Xception_after_aug
├── CNN_with_CutMix/
│   └── MobileNet_V2_with_cutmix
├── ViTs_Before_Augmentation/
│   ├── DeiT_without_aug
│   ├── TnT_before_aug
│   └── ViT_base_before_aug
├── ViTs_After_Augmentation/
│   ├── DeiT_after augmentation
│   ├── TNT_After_Augmentation
│   └── ViT_Base_after_Aug
├── ViTs_with_cutmix/
│   ├── DeiT_small_with_cutmix
│   ├── TNT_small_with_cutmix
│   ├── ViT_base_with_cutmix
│   └── ViT_samll_with_cutmix_final
```

### Description

- **CNNs_Before_Augmentation**: Baseline CNN models trained without any data augmentation.
- **CNNs_After_Augmentation**: Same CNN models trained with augmentation strategies.
- **CNN_with_CutMix**: MobileNetV2 model trained using CutMix augmentation technique.
- **ViTs_Before_Augmentation**: Vision Transformer models evaluated without any augmentation.
- **ViTs_After_Augmentation**: Vision Transformers trained with standard augmentation techniques.
- **ViTs_with_cutmix**: Transformer-based models trained using CutMix, including the final ViT-Small model.
--- 
### How to Run the Code


1. **Requirements**:  
   - Python 3.8+
   - PyTorch
   - `timm` (for pretrained ViT models)
   - Matplotlib, Pandas, tqdm
2. **To Train a Model**:
   - Open any notebook (e.g., `ViT_Small_with_CutMix.ipynb`)
   - Run all cells in order
   - Results (loss/accuracy plots, metrics) will be saved in the `results/` folder.

3. **Data**:
   - The dataset is organized into `train`, `val`, and `test` folders.
   - Each contains two subfolders: `BMR/` and `RASH/` with corresponding labeled images.
--- 
### Output

- Training curves: Accuracy & Loss 
- CSV file with per-epoch metrics
- Final test accuracy and loss printed in each notebook

---
### Load The Best Model for Prediction
```python
import torch
from timm import create_model
from torchvision import transforms
from PIL import Image

# Step 1: Load the Vision Transformer model architecture
model_name = "vit_small_patch16_224"
num_classes = 2  # We have two classes: BMR and RASH
model = create_model(model_name, pretrained=False)

# Define the same custom head used during training
model.head = nn.Sequential(
    nn.Linear(model.head.in_features, num_classes)
)

# Now load the weights
model.load_state_dict(torch.load("Suggested_Best_Model.pth", map_location='cpu'))
model.eval()  # Set the model to evaluation mode

# Step 3: Define the preprocessing steps for input image
# Resize to 224x224, convert to tensor, and normalize with mean and std
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # Normalize to [-1, 1] range
])

# Step 4: Load and prepare the image
image_path = "test_image.jpg" # Replace with actual image path
image = Image.open(image_path).convert("RGB")  # Make sure it's in RGB format
image_tensor = transform(image)  # Apply the transformation

# Step 5: Make a prediction
output = model(image_tensor.unsqueeze(0))  # Add batch dimension
predicted = output.argmax(dim=1).item()  # Get the predicted class index

# Step 6: Print the result
print("Predicted class:", "BMR" if predicted == 0 else "RASH") 
```
### Notes

- To use the best model, ensure the model weights file `Suggested_Best_Model.pth` is in your working directory.
-  Models are trained and validated using stratified datasets.
- CutMix was applied to improve generalization in small medical datasets.
- Early stopping and learning rate scheduling were 
used to prevent overfitting.

---