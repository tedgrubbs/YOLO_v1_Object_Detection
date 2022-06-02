from Common_Imports import *
from PIL import Image as im

# This should create a simple set of images with white boxes randomly drawn inside
# a black box. This will validate that the object detection training works.

if __name__ == '__main__':

    img_path = 'Square_Images_Val/'
    if not path.exists(img_path):
        mkdir(img_path)

    annotation_path = 'Square_Dataset_Val/'
    if not path.exists(annotation_path):
        mkdir(annotation_path)

    with open('config.json' ,'r') as fp:
        config = json.load(fp)

    num_images = 2000
    box_height = config['input_shape'][2]
    box_width = config['input_shape'][3]
    max_size = 16
    max_boxes_per_image = 10

    for i in range(num_images):

        arr = np.zeros((box_height, box_width), dtype='uint8')
        temp_box_count = np.random.randint(1, max_boxes_per_image+1)
        output_string = []
        box_locations = np.zeros((temp_box_count,2))

        for box in range(temp_box_count):

            # prevents having any boxes within 7 units of each other
            while True:
                rand_center = np.random.randint(max_size, box_width-max_size, 2)
                if np.any(np.linalg.norm(rand_center - box_locations, axis=1) < 32):
                    continue
                else:
                    break

            rand_size = np.random.randint(2, max_size+1, 2)

            # arr[rand_center[0] - rand_size[0] : rand_center[0] + rand_size[0] + 1, \
            #     rand_center[1] - rand_size[1] : rand_center[1] + rand_size[1] + 1 ] = 255

            arr[rand_center[0] - rand_size[0], \
                rand_center[1] - rand_size[1] : rand_center[1] + rand_size[1] + 1 ] = 255

            arr[rand_center[0] + rand_size[0], \
                rand_center[1] - rand_size[1] : rand_center[1] + rand_size[1] + 1 ] = 255

            arr[rand_center[0] - rand_size[0] : rand_center[0] + rand_size[0] + 1, \
                rand_center[1] - rand_size[1]  ] = 255

            arr[rand_center[0] - rand_size[0] : rand_center[0] + rand_size[0] + 1, \
                rand_center[1] + rand_size[1]  ] = 255

            output_string.append(np.append(rand_center, rand_size, axis=0))
            box_locations[box] = rand_center


        imdata = im.fromarray(arr)
        imdata.save(img_path + str(i), format='png')
        output_string = np.array(output_string)
        np.save(annotation_path + str(i), output_string.astype('int'))
