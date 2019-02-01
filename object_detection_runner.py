import glob
import os
import pathlib
import random
import string
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from PIL import Image

from object_detection.utils import label_map_util

# Change this value according to your pc specs (higher = more strain)
MAX_CONCURRENT_WORKERS = 15

MAX_NUMBER_OF_BOXES = 10
MINIMUM_CONFIDENCE = 0.5

PATH_TO_OUTPUT = 'Datasets/'
PATH_TO_LABELS = 'model-car/label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'Datasets/Raw'

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=sys.maxsize,
                                                            use_display_name=True)
CATEGORY_INDEX = label_map_util.create_category_index(categories)

# Path to frozen detection graph. This is the actual model that is used for the object detection.
MODEL_NAME = 'model-car'
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def detect_objects(path):
    image = Image.open(path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                             feed_dict={image_tensor: image_np_expanded})

    c = np.squeeze(classes).astype(np.int32)
    s = np.squeeze(scores)

    file_name = ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(10))
    for i in range(0, len(c)):
        if c[i] in CATEGORY_INDEX.keys():
            if s is not None or s[i] >= MINIMUM_CONFIDENCE:
                class_name = CATEGORY_INDEX[c[i]]['name']
                if pathlib.Path(PATH_TO_OUTPUT + class_name + "/").exists() is False:
                    pathlib.Path(PATH_TO_OUTPUT + class_name + "/").mkdir(parents=True)
                if pathlib.Path(
                        PATH_TO_OUTPUT + class_name + "/" + class_name + "-" + file_name + ".jpg").exists() is False:
                    image.save(PATH_TO_OUTPUT + class_name + "/" + class_name + "-" + file_name + ".jpg")


# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image-{}.jpg'.format(i)) for i in range(1, 4) ]
TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))

# Load model into memory
print('Loading model...')
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

print('detecting...')
with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
        detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
        detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
        detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = detection_graph.get_tensor_by_name('num_detections:0')

        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
            futures = {executor.submit(detect_objects, image_path): image_path for image_path in TEST_IMAGE_PATHS}
