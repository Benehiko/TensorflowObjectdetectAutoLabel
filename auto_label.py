import glob
import os
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import tensorflow as tf
from PIL import Image
from pascal_voc_writer import Writer

from object_detection.utils import label_map_util

# Set this based on your specs
MAX_CONCURRENT_WORKERS = 15

# Minimum confidence to use in labeling process
MINIMUM_CONFIDENCE = 0.5

# Class names to auto annotate
CLASS_NAME = ['car']
PATH_TO_OUTPUT = 'Datasets/'
PATH_TO_LABELS = 'model-car/label_map.pbtxt'
PATH_TO_TEST_IMAGES_DIR = 'Datasets/'

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


def detect_objects(img_path):
    image = Image.open(img_path)
    image_np = load_image_into_numpy_array(image)
    image_np_expanded = np.expand_dims(image_np, axis=0)

    (boxes, scores, classes, num) = sess.run([detection_boxes, detection_scores, detection_classes, num_detections],
                                             feed_dict={image_tensor: image_np_expanded})

    c = np.squeeze(classes).astype(np.int32)
    s = np.squeeze(scores)
    b = np.squeeze(boxes)
    file_name = img_path.split('/')[-1]
    c_name = file_name.split('-')[0]
    for i in range(0, len(c)):
        if c[i] in CATEGORY_INDEX.keys():
            class_name = CATEGORY_INDEX[c[i]]['name']
            if class_name == c_name:
                if s is not None or s[i] >= MINIMUM_CONFIDENCE:
                    out_dir = PATH_TO_OUTPUT + "/" + c_name + "/"
                    box = tuple(b[i].tolist())
                    width, height = image.size
                    ymin, xmin, ymax, xmax = box
                    ymin *= height
                    ymax *= height
                    xmin *= width
                    xmax *= width
                    if max(ymin, xmin, ymax, xmax) > 0:
                        image = Image.fromarray(image_np.astype(np.uint8))
                        fn = file_name.replace('.jpg', '')
                        writer = Writer(out_dir + file_name, width, height)
                        writer.addObject(class_name, int(xmin), int(ymin), int(xmax),
                                         int(ymax))
                        writer.save(out_dir + fn + '.xml')


# TEST_IMAGE_PATHS = [ os.path.join(PATH_TO_TEST_IMAGES_DIR, 'image-{}.jpg'.format(i)) for i in range(1, 4) ]
TEST_IMAGE_PATHS = []
for x in CLASS_NAME:
    TEST_IMAGE_PATHS += glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR + x + "/", '*.jpg'))

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

