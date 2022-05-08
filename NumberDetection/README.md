### Project 2: House Number Detection and Classification with CNN Models
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Two neural networks were trained on the Street View House Number (SVHN) dataset. Two different methods were compared to complete this task. The first required that the detection (i.e. Where is the number?) and the classification (i.e. Which number is it?) stages were seperate. The second method utilizes a type of single-shot detection method to perform the detection and classification in one stage.

## Method 1: Detection with MSER and Classification with VGG16
&nbsp;&nbsp;1) Detection with Maximal Stable Extremal Region Extractor (MSER)[https://docs.opencv.org/3.4/d3/d28/classcv_1_1MSER.html]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MSER returns stable regions from an image (e.g. shapes, letters, numbers). This is good for the detection of numbers but may not work for the detection of other more-complex objects (e.g. vehicles, people, etc.).  
&nbsp;&nbsp;2) Classification with Pytorch's VGG16 implemention with untrained weights and pre-trained weights  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The VGG16 network was trained with two methods 1) training the weights from scratch and 2) training from a baseline of pre-trained weights. As expected, the network utilizing the pre-trained weights performed better than without; in this case it performed 4.5% higher after 20 epochs of training.

<img src="https://user-images.githubusercontent.com/29446797/167309158-949b24bc-2e22-47fc-a688-d06bbb262538.png"> <img src="https://user-images.githubusercontent.com/29446797/167309161-c2e9381e-4ef2-46fc-8ca0-6a3af97f615e.png">  
&nbsp;&nbsp;3) These graphs represent the accuracy at each epoch for the network trained from scratch (left) and the network trained with pre-trained weights (right).

![vgg](https://user-images.githubusercontent.com/29446797/167310676-9744a36d-e80d-4262-abf4-cf41248f6cc6.gif) [Full-Res Video](https://github.com/hsilvaga/ComputerVisionProjects/blob/master/NumberDetection/data/vgg16_output.avi])


## Method 2: Detection and Classification performed by YOLO (You Only Look Once)
&nbsp;&nbsp;1) Yolo is able to perform detection and classification simultaneously. In this case, Pytorch's Yolov5-small implementation with pre-trained weights was used. Even with these short videos, YOLO performed noticeably faster.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NOTE: The bounding boxes from YOLO are not as "fitting" compared to method 1's detection boxes.
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;NOTE: Detection is poor on small objects.  
![yolo_output](https://user-images.githubusercontent.com/29446797/167315576-e57daf80-8f8d-4c62-847a-fa2332b1a6a1.gif) [Full-Res](https://github.com/hsilvaga/ComputerVisionProjects/blob/master/NumberDetection/data/yolo_mile_output.avi)   
![yolo_paper_output](https://user-images.githubusercontent.com/29446797/167315579-28ae63ce-3211-4a69-b909-5c026b1fbf80.gif) [Full-Res](https://github.com/hsilvaga/ComputerVisionProjects/blob/master/NumberDetection/data/yolo_paper_output.avi)  
