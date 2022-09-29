# YOLO v1 object detection
The YOLO v1 algorithm for object detection is elegant in its simplicity and speed. I decided that it would be educational to implement it as a first try into object detection. Implementing the loss function from scratch has been the hardest part.

The original paper:

Redmon, J., Divvala, S., Girshick, R., and Farhadi, A., “You Only Look Once: Unified, Real-Time Object Detection” - https://arxiv.org/abs/1506.02640v3

Network is easily configurable but more options are needed to make things more customizable.

An experimentation dataset is included to make sure everything is running correctly. The network just has to look at an image of simple rectangles and learn to detect those. This dataset is created with "Build_Square_Dataset.py".


