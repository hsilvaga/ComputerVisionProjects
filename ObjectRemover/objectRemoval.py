import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from segmentationNetwork import MyModel

import torch
from torchvision import datasets, models, transforms

"""
@description: Aligns images based off of first input image
@param: Array of images
@returns: Array of aligned images
"""
def alignImages(imgs):
    warpedImgs = []

    baseImg = imgs[0]  # Align all images off of first image
    imgs.pop(0)
    warpedImgs.append(baseImg)

    for i, currImg in enumerate(imgs):

        ORB = cv2.ORB_create(nfeatures=8000,
                             scaleFactor=1.8,
                             nlevels=5,
                             edgeThreshold=90,
                             firstLevel=1,
                             WTA_K=4,
                             patchSize=31,
                             fastThreshold=30)

        keypoints1, des1 = ORB.detectAndCompute(baseImg, None)
        keypoints2, des2 = ORB.detectAndCompute(currImg, None)

        # Create BF matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)  # cv2.FlannBasedMatcher(flann_params, {}) #
        matches = bf.knnMatch(des1, des2, k=2)

        # Compare descriptors of current image with the following image
        matches_cleaned = []

        for m, n in matches:
            if m.distance < 0.75 * n.distance:
                matches_cleaned.append(m)

        # sort matches by closest distance
        matches_cleaned = sorted(matches_cleaned, key=lambda x: x.distance)

        # Sample keypoints
        matching_keypoints_1 = []
        matching_keypoints_2 = []

        for match in matches_cleaned:
            matching_keypoints_1.append(keypoints1[match.queryIdx])
            matching_keypoints_2.append(keypoints2[match.trainIdx])

        p1 = []
        p2 = []

        for i in range(len(matching_keypoints_1)):
            p1.append([matching_keypoints_1[i].pt[0], matching_keypoints_1[i].pt[1]])
            p2.append([matching_keypoints_2[i].pt[0], matching_keypoints_2[i].pt[1]])

        f, inliers = cv2.findHomography(srcPoints=np.array(p2),
                                        dstPoints=np.array(p1),
                                        method=cv2.RANSAC,
                                        ransacReprojThreshold=3.8)

        warpedImg = cv2.warpPerspective(currImg, f, (currImg.shape[1], currImg.shape[0]))

        warpedImgs.append(warpedImg)

    return warpedImgs

"""
@description: Reads in images with opencv
@param: Array of paths for each image
@returns: Array of images (numpy-format)
"""
def readImages(imagePath):
    imgs = []
    for x in imagePath:
        imgs.append(cv2.cvtColor(cv2.imread(x), cv2.COLOR_BGR2RGB))

    return imgs

"""
@description: Predicts segmentation mask from image
@param: accepts pytorch model, image in numpy-format, ground-truth mask for comparison (optional)
@returns: @TODO#################################################
"""
def predict_image(model, image, gt_mask=None, showImages=True):
    model.eval()
    if gt_mask is None:
        gt_mask = np.ones((1, 1))
    with torch.no_grad():
        image = image.astype("float32") / 255.0 if gt_mask.sum() == 1 else image.astype("float32")
        image = cv2.resize(image, (256, 256))

        orig_image = image.copy()

        # gt_mask = cv2.resize(mask, (256, 256)) if gt_mask.sum() > 0 else None

        image = np.transpose(image, (2, 0, 1))  # Move channel axis to front
        image = np.expand_dims(image, 0)

        # Make prediction from image input
        image = torch.from_numpy(image).to(DEVICE)
        pred_mask = model(image)[0].squeeze()

        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = pred_mask.cpu().numpy()

        f, axarr = plt.subplots(1, 3, figsize=(15, 15)) if gt_mask.sum()==0 else plt.subplots(1, 2, figsize=(15, 15))

        if showImages:
            axarr[0].title.set_text('Original Image')
            axarr[1].title.set_text('Predicted Segmentation Mask')
            axarr[2].title.set_text('GT Segmentation Mask') if gt_mask.sum()==0 else None
            axarr[0].imshow(cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB))
            axarr[1].imshow(pred_mask)
            axarr[2].imshow(gt_mask)  if gt_mask.sum()==0 else None

        return pred_mask

"""
@description: Removes segmented object from image by setting it to zero (Works for binary classification)
@param: array of images, array of corresponding segmentation masks 
"""
def removeSegmentedPixels(images, segmentedImages):
    arr = np.ones((images[0].shape[0], images[0].shape[1]), np.uint8)
    orig_img = np.ones((images[0].shape[0], images[0].shape[1]), np.uint8)
    processedImages = []

    for i, img in enumerate(images):
        arr[segmentedImages[i] > 0.5] = 0
        arr[segmentedImages[i] < 0.5] = 1
        print(np.unique(arr))
        orig_img[:, :, 0] *= arr
        orig_img[:, :, 1] *= arr
        orig_img[:, :, 2] *= arr

        processedImages.append(orig_img)

    return processedImages


def main():
    #Get images
    imageNames = glob.glob("images/*.jpg")
    imageNames.sort()

    images = readImages(imageNames)

    #Align images
    images = alignImages(images)

    #Segment people in images
    segmentedMasks = []

    with torch.no_grad():
        for img in images:
            pred_mask = predict_image(model, img, showImages=False)
            segmentedMasks.append(pred_mask)

    plt.show()

    #Remove detected pixels from images
    processedImages = removeSegmentedPixels(images, segmentedMasks)


if __name__ == '__main__':
    #Choose device to load model
    DEVICE = 'cpu'
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    #Load model architecture / weights
    model = torch.load("models/model1.pth")
    model.to(DEVICE)
    model.eval()
    print("Model Loaded")

    main()
