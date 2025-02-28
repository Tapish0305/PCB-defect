# PCB Defect Detection using ODNN 
**Dataset Overview**/n
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
The dataset was then split into training and testing  subsets. The six defect classes were mapped to numerical labels 1 to 6. Using plot_image, the PCBs  were visualized with their bounding boxes .
