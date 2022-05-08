### Project 2: House Number Detection and Classification with CNN Models
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Two neural networks were trained on the Street View House Number (SVHN) dataset. Two different methods were compared to complete this task. The first required that the detection (i.e. Where is the number?) and the classification (i.e. Which number is it?) stages were seperate. The second method utilizes a type of single-shot detection method to perform the detection and classification in one stage.

## Method 1: Detection with MSER and Classification with VGG16
&nbsp;&nbsp;1) Detection with Maximal Stable Extremal Region Extractor (MSER)[https://docs.opencv.org/3.4/d3/d28/classcv_1_1MSER.html]  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;MSER returns stable regions from an image (e.g. shapes, letters, numbers). This is good for the detection of numbers but may not work for the detection of other more-complex objects (e.g. vehicles, people, etc.).  
&nbsp;&nbsp;2) Classification with Pytorch's VGG16 implemention with untrained weights and pre-trained weights  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The VGG16 network was trained with two methods 1) training the weights from scratch and 2) training from a baseline of pre-trained weights.  

<img src="https://user-images.githubusercontent.com/29446797/167309158-949b24bc-2e22-47fc-a688-d06bbb262538.png" title="title">

<img src="https://user-images.githubusercontent.com/29446797/167309161-c2e9381e-4ef2-46fc-8ca0-6a3af97f615e.png"  alt="things">


![acc_scratch](https://user-images.githubusercontent.com/29446797/167309661-00dd71ac-4fdf-4577-a140-132ff70e899c.png "image2")  

![acc_pretrained](https://user-images.githubusercontent.com/29446797/167309648-5c02c433-e679-485e-9f9a-9ea509863b99.png "Image1")



## Method 2: Detection and Classification performed by YOLO (You Only Look Once)
&nbsp;&nbsp;1)
