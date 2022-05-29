from Common_Imports import *
from PIL import Image as im

# This should create a simple set of images with white boxes randomly drawn inside
# a black box. This will validate that the object detection training works.

if __name__ == '__main__':

    img_path = 'Square_Images/'
    if not path.exists(img_path):
        mkdir(img_path)

    with open('config.json' ,'r') as fp:
        config = json.load(fp)

    num_images = 1000
    box_height = config['input_shape'][2]
    box_width = config['input_shape'][3]
    max_size = 16
    max_boxes_per_image = 10

    for i in range(num_images):

        arr = np.zeros((box_height, box_width), dtype='uint8')
        temp_box_count = np.random.randint(1, max_boxes_per_image+1)

        for box in range(temp_box_count):
            rand_center = np.random.randint(max_size, box_width-max_size+1, 2)
            rand_size = np.random.randint(2, max_size+1, 2)
            arr[rand_center[0] - rand_size[0] : rand_center[0] + rand_size[0] + 1, \
                rand_center[1] - rand_size[1] : rand_center[1] + rand_size[1] + 1 ] = 255

        imdata = im.fromarray(arr)
        imdata.save(img_path + str(i) + '.png')
