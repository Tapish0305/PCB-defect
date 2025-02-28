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
The dataset was then split into training and testing  subsets. The six defect classes were mapped to numerical labels 1 to 6. Using plot_image, the PCBs  were visualized with their bounding boxes.

# Data Preprocessing

A custom PCBData dataset was created to:
1. Load mages and extract bounding box annotations .
2. Normalize and apply necessary transformation to the images Store target data as a dictionary containing:
3. boxes
4. labels
5. image_id
6.  is_crowd
Bounding boxes were visualized to confirm accurate labeling.

# Model Architecture
![model](https://github.com/user-attachments/assets/8c1aff7c-96f4-437d-a3be-1c45bed397a2)

# Model Preparation
The dataset was split into train_df , valid_df , and test_df . Data was loaded using:
1. num_workers = 6
2. batch_size = 1
3. custom  collate_fn function
A pretrained Faster R-CNN model (FasterRCNN_ResNet50_FPN_Weights.DEFAULT) was used. The final ROI head  layer was replaced with a custom layer.



# Training and Optimization
The model was trained using Adam optimizer , a learning rate scheduler , and ODNN loss computation . The following hyperparameters were used:
1. Learning Rate: 0.0001
2. Weight Decay: 0.0005
3.  StepLR Scheduler: step_size=3, gamma=0.1
4. Epochs: 25
   
The training loop  followed these steps:
1. Iterate through the training dataset and compute loss


 2. Perform gradient backpropagation and then update the weights



Adjust the learning rate scheduler .
Print and log the training loss .
The model was evaluated on the validation set by calculating the Intersection over Union (IoU)  metric
After training, the models  and corresponding loss/IoU curves  were stored for comparison. Then, I plotted  the Loss Measure vs Number of Epochs and IoU Curve vs Number of Epochs. Finally, I used the trained model to predict  bounding boxes on test images  and saved the model path . Later, I developed a refined model  that enhances prediction accuracy  on testing results.

## Result - 
![results 1](https://github.com/user-attachments/assets/cdfec633-f1e5-4529-8b54-d2a00e68b53a)
Represents the average IoU value over all the validation data
![Uploading correlation value.png…]()

This shows the reduction of correlation value among network filters

# Conclusion
This project successfully implemented PCB defect detection  using ODNN  and a Faster R-CNN model . The dataset was processed efficiently , and the model was trained with effective loss computation  and optimization techniques . Future improvements can include:
Experimenting with different backbone architectures.
Fine-tuning hyperparameters for better performance.
Implementing real-time defect detection using edge devices.


# Dependencies
 Python 3.x
 PyTorch
 torchvision
 NumPy
 OpenCV
 Pandas
 Matplotlib
 tqdm
 
# Acknowledgments
Special thanks to Kaggle  for providing the dataset  and to the open-source community  for valuable tools and frameworks  that made this project possible!



