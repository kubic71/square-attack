from PIL import Image
import random
import numpy as np
from google.cloud import vision
import io
import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "keys/Trema-14000fdb4eac.json"

def convert_to_pillow(img, channels_first=True):
    """Convert numpy img to pillow Image object"""

    # convert from channels-first to channels-last
    if channels_first:
        img = img.transpose(1, 2, 0)

    img = (img * 255).astype(np.uint8)
    return Image.fromarray(img)



class GVisionResults:
    def __init__(self, results):
        self.results = results
    
    def match(self, labelset, inverse=False):
        labels = []
        scores = []

        for label, score in self.results:

            if inverse:
                # add to result if none of the patterns match given label
                if not any([l.lower() in label.lower() for l in labelset]):
                    labels.append(label)
                    scores.append(score)
            else:
                if any([l.lower() in label.lower() for l in labelset]):
                    labels.append(label)
                    scores.append(score)
        
        return GVisionResults(list(zip(labels, scores)))

    @property
    def labels(self):
        return [l for l, s in self.results]

    @property
    def scores(self):
        return [s for l, s in self.results]

    @property
    def top_label(self):
        top_score = max([s for l, s in self.results])
        assert(top_score == self.results[0][1])
        return self.results[0][0]

    @property
    def top_score(self):
        top_score = max([s for l, s in self.results])
        assert(top_score == self.results[0][1])
        return top_score

    def __str__(self):
        return "\n".join([l + ": " + str(s)  for l, s in self.results])

def gvision_classify_numpy(img):
    """Return the labels and scores by calling the cloud API

    Args:
        img -- numpy [W, H, C] image with values [0, 1]
    
    """

    img = convert_to_pillow(img)
    fname = "/tmp/.temp_img_" + str(random.randint(0, 1000000)) + ".png"
    img.save(fname)

    client = vision.ImageAnnotatorClient()
    # Loads the image into memory
    with io.open(fname, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    # Performs label detection on the image file
    response = client.label_detection(image=image, max_results=100)
    labels = response.label_annotations

    descriptions = [label.description for label in labels]
    scores = [label.score for label in labels]

    return GVisionResults(list(zip(descriptions, scores)))