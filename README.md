# Project Title
Implementation of Printed Circuit Board (PCB) defect detection through modern Machine Learning (ML) algorithms.

# Dataset Overview

The dataset  used for this project was obtained from Kaggle  and contains various components such as:
Annotations (XML files )
Images (JPG format )
Rotation 
PCB used
First, I determined the directory structure of the dataset and extracted relevant details from the XML and JPG  files. These details were compiled into a DataFrame  containing the following attributes:
xmin, xmax, ymin, ymax (bounding box coordinates)
class ( defect type)
file ( image file name)
width, height ( image dimensions)
The dataset was then split into training and testing  subsets. The six defect classes were mapped to numerical labels 1 to 6. Using plot_image, the PCBs  were visualized with their bounding boxes 

# Data Preprocessing

A custom PCBData dataset was created to:
1. Load mages and extract bounding box annotations .
2. Normalize and apply necessary transformation to the images Store target data as a dictionary containing:
3. boxes
4. labels
5. image_id
6.  is_crowd
Bounding boxes were visualized to confirm accurate labeling.
 
# Model Preparation
The dataset was split into train_df , valid_df , and test_df . Data was loaded using:
1. num_workers = 6
2. batch_size = 1
3. custom  collate_fn function
A pretrained Faster R-CNN model (FasterRCNN_ResNet50_FPN_Weights.DEFAULT) was used. The final ROI head  layer was replaced with a custom layer:

# Training and Optimization
The model was trained using Adam optimizer , a learning rate scheduler , and ODNN loss computation . The following hyperparameters were used:
1. Learning Rate: 0.0001
2. Weight Decay: 0.0005
3.  StepLR Scheduler: step_size=3, gamma=0.1
4. Epochs: 25
The training loop  followed these steps:
1. Iterate through the training datase
 Compute loss using:

