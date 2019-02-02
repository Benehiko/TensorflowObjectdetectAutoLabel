# TensorflowObjectdetectAutoLabel
Auto Label training data using a small trained model trained by using a small hand labelled dataset


### Requirements

- Python (3 or 2)
- [Tensorflow](https://github.com/tensorflow/tensorflow)
- [pascal_voc_writer](https://github.com/AndrewCarterUK/pascal-voc-writer)
- numpy
- Pillow

### How to use it

This repository gives you all of the required dataset and trained model to test the supplied scripts with. The dataset supplied is a small sample set of [Stanford Ai car dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)

To use your own dataset and custom model follow the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection) to train the model and hand label the images with [labelImg](https://github.com/tzutalin/labelImg) or any PascalVoc labelling tool. Train for a couple thousand steps to get an adequate detection confidence (80% +). I used 200 hand labelled images to train with.

After you have acquired the frozen graph, add it to a folder and specify it inside the python scripts under MODEL_NAME and PATH_TO_CKPT. Also ensure to have the label map added here and specified under PATH_TO_LABELS. 

The Dataset/Raw folder is a dump folder for images scrapped or gathered which you would like to be sorted using your trained model. 

Run object_detection_runner.py to sort your classes into folders under the Dataset/ path. 

    python3 object_detection_runner.py
    
Then run auto_label.py to label the sorted classes (specify the classes in CLASS_NAME list). Change the other constants as required. 

    python3 auto_label.py
    
After the run is complete, check the labelled images by going to [labelImg](https://github.com/tzutalin/labelImg) and selecting "Open Dir" and selecting "Change Save Dir" to the class folder under "Dataset/". This will load the generated annotations for you to verify on each image. 

Please note that auto_label and object_detection_runner is not owned by me and a lot of the coding done was made by Google and [other open source contributors](https://github.com/bourdakos1/Custom-Object-Detection)

I only added some extra features to generate PascalVoc xml files for each class detected.

### Sources

[Stanford Ai Car Dataset](http://ai.stanford.edu/~jkrause/cars/car_dataset.html)
3D Object Representations for Fine-Grained Categorization
Jonathan Krause, Michael Stark, Jia Deng, Li Fei-Fei
4th IEEE Workshop on 3D Representation and Recognition, at ICCV 2013 (3dRR-13). Sydney, Australia. Dec. 8, 2013.

[Tensorflow](https://github.com/tensorflow/tensorflow)

[Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection)


### License

[Apache License 2.0](https://github.com/Benehiko/TensorflowObjectdetectAutoLabel/blob/master/LICENSE)
