# Paper Implementation of Object Removal by Exemplar-Based Image Inpainting
# Objective: Implement Region-Filling [Paper](https://www.irisa.fr/vista/Papers/2004_ip_criminisi.pdf) and Obtain Similar Results 

# Walkthrough of Implemented Algorithm:
&nbsp;&nbsp;This algorithm takes in three inputs  
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;1: A color-image | 2: A mask of the object to be removed | 3: A window size which determines the size of the infill region  
&nbsp;&nbsp;1) Compute Patch Priorities of Fill-Front  
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This is a calculation of which point, and its window, will be filled in first.  
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This calculation ensures that the algorithm fills in 1) the pixels from outer-towards-inner. And that 2) pixels where the image has the strongest edges fill the image first; this ensures that lines going behind the object are continued through as the object is filled in.
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The priority calculations are the main novelty of the algorithm. 
-  
&nbsp;&nbsp;2) Perform Inpainting on Highest Priority Point on Fill-Front  
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This portion of the algorithm simply selects the most similar window-region from the image to fill-in the object in question.
- &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;This was implemented using OpenCV's matchTemplate() to find the most similar region.
# Results:
### From left-to-right: 1: Input image, 2: masked region of object to be removed, 3: output with removed object
## Cone Removed 
<img src="https://user-images.githubusercontent.com/29446797/157146736-85275658-0eed-42f4-bd07-6b4021fd1f94.png" height="487" width="274"> <img src="https://user-images.githubusercontent.com/29446797/157146869-ab0be8c4-6b73-4b87-8d50-44881357e2ab.png" height="487" width="274">
<img src="https://user-images.githubusercontent.com/29446797/157146944-34414296-86de-456d-8d07-b2dd504ed175.png" height="487" width="274">
## Person Removed
<img src="https://user-images.githubusercontent.com/29446797/157149418-ec987978-de2a-4b5d-84fa-7624303ba3db.png" height="274" width="487"> <img src="https://user-images.githubusercontent.com/29446797/157149490-f8f2636f-cccf-4b04-9486-8ef4ce9c5820.png" height="274" width="487">
<img src="https://user-images.githubusercontent.com/29446797/157149557-ac138fef-afc4-4add-b1c9-3bcd8d99e564.png" height="247" width="487">

