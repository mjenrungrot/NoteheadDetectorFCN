import os
import pkg_resources
import yaml
import glob

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
