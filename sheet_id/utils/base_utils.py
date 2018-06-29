import os
import pkg_resources
import yaml
import glob
import cv2
import numpy as np
import pandas as pd
import math

def loadSettings():
    settingsFilepath = pkg_resources.resource_filename('sheet_id', 'settings.yaml')
    with open(settingsFilepath, 'r') as stream:
        try:
            return yaml.load(stream)
        except yaml.YAMLError as exc:
            raise

def loadScoresDataset(path):
    """
    Returns a list of paths to scores (file extension: png)
    """ 
    return sorted(glob.glob(os.path.join(path, '*.png')))

def loadCellphoneScoresDataset(path):
    """
    Return a list of paths to cellphone scores (file extension: jpg)
    """
    return sorted(glob.glob(os.path.join(path, '*.jpg')))

def generateScoresDB(scorePaths):
    """
    Return a dictionary object 
        {
            <filename without extension>: <representation> (e.g. np.array)
        }
    """
    db = {}
    for i in range(len(scorePaths)):
        img = cv2.imread(scorePaths[i], 0)
        fileNameNoExt = os.path.splitext(os.path.split(scorePaths[i])[1])[0]
        db[fileNameNoExt] = img
    return db

def calculateMRR(ranks):
    """
    Return an MRR score based on the list of rank predictions
    """
    MRR = 0
    for rank in ranks:
        MRR += 1.0 / rank
    return MRR / len(ranks)

def generateSheetMaskAnnotation(img_path=loadSettings()['SCANNED_ANNOTATION_IMAGE_PATH'], 
                                csv_path=loadSettings()['SCANNED_ANNOTATION_PATH'], 
                                trimmed=True, plot=False):
    """
    Generate mask annotations from the bounding box annotations.

    Output:
        output - dictionary of images
    {
        filename: (image, mask, list of bounding boxes)
    } 
    """


    # Read image paths
    if trimmed: img_paths = sorted(glob.glob(os.path.join(img_path, '*trimmed.png')))
    else: img_paths = sorted(glob.glob(os.path.join(img_path, '*[!trimmed].png')))
    
    # Read annotations
    df = pd.read_csv(csv_path, index_col=False)
    
    # Drop annotations not in the image annotation
    if trimmed:
        for i, row in df.iterrows():
            df.at[i, 'filename'] = df.at[i, 'filename'] + '_trimmed'
            filePath = os.path.join(img_path, df.at[i, 'filename'] + '.png')
            img_shape = cv2.imread(filePath, 0).shape
            if df.at[i, 'vpix'] >= img_shape[0]:
                df.drop(i, inplace=True)
        df.reset_index(drop=True, inplace=True)
        
    # Generate output pairs
    output = {}
    for path in img_paths:
        filename = os.path.splitext(os.path.split(path)[1])[0]
        img = cv2.imread(path, 0)

        df_score = df[df['filename'] == filename]
        mask = np.zeros(img.shape)

        boxes = []
        for i, row in df_score.iterrows():
            note_height = math.ceil(row['staff_height'] / 8)
            (start_row, end_row) = (row['vpix'] - note_height, row['vpix'] + note_height)
            (start_col, end_col) = (row['hpix'] - 1, row['hpix'] + note_height)
            threshold = 60
            
            boxes.append([start_col, start_row, end_col, end_row])
            mask[start_row:end_row, start_col:end_col] = np.where(img[start_row:end_row, start_col:end_col] is not None,
                                                                  29, 0)
            if plot:
                plt.figure(figsize=(20,20))
                plt.subplot(1,2,1)
                plt.imshow(img, cmap='gray')
                plt.subplot(1,2,2)
                plt.imshow(mask, cmap='gray')
                plt.show()
            
        output[filename] = (img, mask, boxes)
        
    return output
