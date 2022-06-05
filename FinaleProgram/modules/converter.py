from genericpath import exists
from tokenize import String
from cv2 import WINDOW_AUTOSIZE, VideoCapture, imwrite,namedWindow,waitKey,resize;
from time import sleep;
import os

def getImageFromVideo(fullPath : str, dest="./pictures"):
    """
    getImageFromVideo(fullPath, [dest])
    @brief Create several pictures from the given video
    @note
    dest must be given in the form of a full path in your PC 
    """
    urlSplit = fullPath.split(sep='\\');
    fullFileName = urlSplit.pop()
    fileNamePrefix = fullFileName.split(sep='.')[0];
    cap = VideoCapture(fullPath)
    # Resolution 640*480
    sleep(1)
    if cap is None or not cap.isOpened():
        print('Unable to open video')
        return
    namedWindow('Image', WINDOW_AUTOSIZE);
    n = 1
    count = 200
    if (not os.path.exists('%s/%s' % (dest,fileNamePrefix))):
        os.mkdir('%s/%s' % (dest,fileNamePrefix))
    while True:
        [success, img] = cap.read()
        waitKey(30)
        if success:
            #img = rotate(img, ROTATE_90_CLOCKWISE)
            imgROI = img[40:(40+480),:] # Create a 480x480 resolution picture
            imgROI = resize(imgROI,(250,250))
        else:
            break
        if n%4 == 0:
            filename = '%s/%s/%s%04d.bmp'%(dest,fileNamePrefix,fileNamePrefix,count)
            imwrite(filename,imgROI)
            count = count + 1
        n = n + 1
    print("Some pictures have been created inside %s/%s"%(dest,fileNamePrefix))
    return