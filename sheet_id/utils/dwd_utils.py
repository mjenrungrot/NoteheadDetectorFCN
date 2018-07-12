import numpy as np
from itertools import product
from PIL import Image
import math, random

def generateGroundTruthMaps(bboxes, annotations, image_shape=(250,250), e_max=10, r=15):
    """
    Generate ground truth quantized energy map given the bounding boxes.
    
    This optimized version takes O(r^2 * |n_objects|) instead of O(row*col*|n_objects|).
    """
    energy_map = np.zeros((len(bboxes), image_shape[0], image_shape[1]))
    bbox_map = np.zeros((energy_map.shape[0], energy_map.shape[1], energy_map.shape[2], 2))

    # Create discrete energy bins
    bins = np.linspace(0, e_max + np.finfo(np.float32).eps, e_max+1)

    for idx in range(len(bboxes)):
        for obj in bboxes[idx]:
            minrow, maxrow = max(0, obj[1] - r), min(image_shape[0], obj[3] + r)
            mincol, maxcol = max(0, obj[0] - r), min(image_shape[1], obj[2] + r)
            for row in range(minrow, maxrow):
                for col in range(mincol, maxcol):
                    center = (obj[0] + obj[2])/2, (obj[1] + obj[3])/2
                    energy_map[idx,row,col] = max(energy_map[idx,row,col], 
                                                  e_max * (1 - np.sqrt((row - center[1])**2 + (col - center[0])**2) / r))
                    if energy_map[idx,row,col] >= bins[1]:
                        bbox_map[idx,row,col,0] = obj[3] - obj[1] # height
                        bbox_map[idx,row,col,1] = obj[2] - obj[0] # width 

    # Discretize energy map      
    energy_map_quantized = np.digitize(energy_map, bins) - 1
   
    # Generate class map
    class_map = (energy_map_quantized > 0) * (annotations[:,:,:,0])
    return energy_map_quantized, class_map, bbox_map

# Array based union find data structure

# P: The array, which encodes the set membership of all the elements

class UFarray:
    def __init__(self):
        # Array which holds label -> set equivalences
        self.P = []

        # Name of the next label, when one is created
        self.label = 0

    def makeLabel(self):
        r = self.label
        self.label += 1
        self.P.append(r)
        return r
    
    # Makes all nodes "in the path of node i" point to root
    def setRoot(self, i, root):
        while self.P[i] < i:
            j = self.P[i]
            self.P[i] = root
            i = j
        self.P[i] = root

    # Finds the root node of the tree containing node i
    def findRoot(self, i):
        while self.P[i] < i:
            i = self.P[i]
        return i
    
    # Finds the root of the tree containing node i
    # Simultaneously compresses the tree
    def find(self, i):
        root = self.findRoot(i)
        self.setRoot(i, root)
        return root
    
    # Joins the two trees containing nodes i and j
    # Modified to be less agressive about compressing paths
    # because performance was suffering some from over-compression
    def union(self, i, j):
        if i != j:
            root = self.findRoot(i)
            rootj = self.findRoot(j)
            if root > rootj: root = rootj
            self.setRoot(j, root)
            self.setRoot(i, root)
    
    def flatten(self):
        for i in range(1, len(self.P)):
            self.P[i] = self.P[self.P[i]]
    
    def flattenL(self):
        k = 1
        for i in range(1, len(self.P)):
            if self.P[i] < i:
                self.P[i] = self.P[self.P[i]]
            else:
                self.P[i] = k
                k += 1

def find_connected_comp(input):
    data = input
    width, height = input.shape

    # Union find data structure
    uf = UFarray()

    #
    # First pass
    #

    # Dictionary of point:label pairs
    labels = {}

    for y, x in product(range(height), range(width)):

        #
        # Pixel names were chosen as shown:
        #
        #   -------------
        #   | a | b | c |
        #   -------------
        #   | d | e |   |
        #   -------------
        #   |   |   |   |
        #   -------------
        #
        # The current pixel is e
        # a, b, c, and d are its neighbors of interest
        #
        # 255 is white, 0 is black
        # White pixels part of the background, so they are ignored
        # If a pixel lies outside the bounds of the image, it default to white
        #

        # If the current pixel is white, it's obviously not a component...
        if data[x, y] == 255:
            pass

        # If pixel b is in the image and black:
        #    a, d, and c are its neighbors, so they are all part of the same component
        #    Therefore, there is no reason to check their labels
        #    so simply assign b's label to e
        elif y > 0 and data[x, y - 1] == 0:
            labels[x, y] = labels[(x, y - 1)]

        # If pixel c is in the image and black:
        #    b is its neighbor, but a and d are not
        #    Therefore, we must check a and d's labels
        elif x + 1 < width and y > 0 and data[x + 1, y - 1] == 0:

            c = labels[(x + 1, y - 1)]
            labels[x, y] = c

            # If pixel a is in the image and black:
            #    Then a and c are connected through e
            #    Therefore, we must union their sets
            if x > 0 and data[x - 1, y - 1] == 0:
                a = labels[(x - 1, y - 1)]
                uf.union(c, a)

            # If pixel d is in the image and black:
            #    Then d and c are connected through e
            #    Therefore we must union their sets
            elif x > 0 and data[x - 1, y] == 0:
                d = labels[(x - 1, y)]
                uf.union(c, d)

        # If pixel a is in the image and black:
        #    We already know b and c are white
        #    d is a's neighbor, so they already have the same label
        #    So simply assign a's label to e
        elif x > 0 and y > 0 and data[x - 1, y - 1] == 0:
            labels[x, y] = labels[(x - 1, y - 1)]

        # If pixel d is in the image and black
        #    We already know a, b, and c are white
        #    so simpy assign d's label to e
        elif x > 0 and data[x - 1, y] == 0:
            labels[x, y] = labels[(x - 1, y)]

        # All the neighboring pixels are white,
        # Therefore the current pixel is a new component
        else:
            labels[x, y] = uf.makeLabel()

    #
    # Second pass
    #

    uf.flatten()

    colors = {}

    # Image to display the components in a nice, colorful way
    output_img = Image.new("RGB", (width, height))
    outdata = output_img.load()

    for (x, y) in labels:

        # Name of the component the current point belongs to
        component = uf.find(labels[(x, y)])

        # Update the labels with correct information
        labels[(x, y)] = component

        # Associate a random color with this component
        if component not in colors:
            colors[component] = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

        # Colorize the image
        outdata[y, x] = colors[component]

    return (labels, output_img)

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
 
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
 
    # initialize the list of picked indexes    
    pick = []
 
    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
 
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
 
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
 
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
 
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
 
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
 
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))
 
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")
