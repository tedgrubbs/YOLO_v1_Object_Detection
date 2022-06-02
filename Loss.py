from Common_Imports import *

class YOLO_Loss(torch.nn.Module):
    def __init__(self, S, config, device):
        super(YOLO_Loss, self).__init__()

        self.config = config
        self.device = device

        self.lambda_coord = 5.0
        self.lambda_noobj = 0.5
        self.S = S
        self.cell_length = int(config['input_shape'][-1] / S)
        self.half_cell = self.cell_length / 2.

        self.iou_threshold = 0.25

    # takes in cell index, local position, and normalized dimensions, and outputs coordinates for global bounding box
    def get_global_box(self, cell, pos, dim):
        dim = torch.clamp(dim, 0., 1.)
        x_right = cell[1]*self.cell_length + pos[1]*self.half_cell + dim[1]*self.config['input_shape'][-1]
        y_top = cell[0]*self.cell_length + pos[0]*self.half_cell + dim[0]*self.config['input_shape'][-1]
        x_left = cell[1]*self.cell_length + pos[1]*self.half_cell - dim[1]*self.config['input_shape'][-1]
        y_bottom = cell[0]*self.cell_length + pos[0]*self.half_cell - dim[0]*self.config['input_shape'][-1]

        if x_right == x_left: x_right += 1
        if y_bottom == y_top: y_top += 1

        return torch.as_tensor([y_top, y_bottom, x_left, x_right])

    def plot_box(self, pos, true=True):

        if true:
            color = 'orange'
        else:
            color = 'cyan'

        plt.plot([pos[2],pos[2],pos[3],pos[3],pos[2]],[pos[1],pos[0],pos[0],pos[1],pos[1]], color = color)

    def IOU(self, boxA, boxB):
        # print('\n', boxA, '\n', boxB)
        x1 = torch.max(boxA[2], boxB[2])
        x2 = torch.min(boxA[3], boxB[3])
        y1 = torch.max(boxA[1], boxB[1])
        y2 = torch.min(boxA[0], boxB[0])
        # print(x2,x1)
        # print(y2,y1)
        width = x2-x1
        height = y2-y1
        # print(width,height)
        if width < 0 or height < 0: return torch.as_tensor(0.)
        area_overlap = width*height
        area_a = (boxA[0] - boxA[1]) * (boxA[3] - boxA[2])
        area_b = (boxB[0] - boxB[1]) * (boxB[3] - boxB[2])
        area_combined = area_a + area_b - area_overlap
        iou = area_overlap / area_combined
        return iou

    # passing the images themself to debug things
    def forward(self, y_pred, y, x):

        """
        Last tensor axis has 11 dimensions for simple square dataset. The following defines how these will be organized
        0,1 : box 1 location
        2,3 : box 2 location
        4,5 : box 1 width, height
        6,7 : box 2 width, height
        8   : box 1 confidence score
        9   : box 2 confidence score
        10  : class probability
        """
        y_pred = y_pred.view(y_pred.size(0), self.S, self.S, -1)

        global_positions = y[:, :, :2]
        box_dims = y[:, :, 2:] / self.config['input_shape'][-1] # Normalizing box size by image width to normalize it between 0 and 1

        # cell indices for each object
        grid_cells = torch.round(global_positions / self.cell_length).long()

        """
        I think the cleanest way to do this is to define the "correct" tensor (y_hat),
        perform the squared difference calculation,
        and then use different masks to pull out the various terms used in the sum.
        """

        obj_ij_pos = torch.zeros(y_pred.size()).to(self.device)
        obj_ij_dims = torch.zeros(y_pred.size()).to(self.device)
        obj_ij_conf = torch.zeros(y_pred.size()).to(self.device)
        noobj_ij_conf = torch.ones(y_pred.size()).to(self.device)
        obj_i_prob = torch.zeros(y_pred.size()).to(self.device)

        y_hat = torch.zeros(y_pred.size()).to(self.device)

        # loop over each image in batch
        for index in range(y.size(0)):
            # print('Real coordinates:', global_positions[index])
            # print('Cell index', grid_cells[index])

            # showing all the boxes that pass the confidence threshold
            if self.config['show_debug_plots']:

                for cellx in range(self.S):
                    for celly in range(self.S):
                        local_pos_pred_b1 = y_pred[index, celly, cellx, :2]
                        local_pos_pred_b2 = y_pred[index, celly, cellx, 2:4]
                        dim_pred_b1 = y_pred[index,  celly, cellx, 4:6]
                        dim_pred_b2 = y_pred[index,  celly, cellx, 6:8]

                        pred_box_b1 = self.get_global_box([cellx, celly], local_pos_pred_b1, dim_pred_b1)
                        pred_box_b2 = self.get_global_box([cellx, celly], local_pos_pred_b2, dim_pred_b2)

                        if y_pred[index,  celly, cellx] [8] > self.iou_threshold:
                            print('B1', y_pred[index,  celly, cellx] [8])
                            self.plot_box(pred_box_b1, False)

                        if y_pred[index,  celly, cellx] [9] > self.iou_threshold:
                            print('B2', y_pred[index,  celly, cellx] [9])
                            self.plot_box(pred_box_b2, False)



            # plt.imshow(x[index][0].detach().cpu())

            # Loop over each object in image
            # Each cell can only predict at most 2 boxes.
            # if 2 or more objects fall into the same cell, it will favor the one that appears last in the list
            for obj in range(grid_cells[index].size(0)):

                # no objects exist at (0,0). once you hit that, it's the end of the object list
                if (global_positions[index][obj][0] == 0 and global_positions[index][obj][1] == 0):
                    break

                # normalizes positions to between -1 and 1, making them relative to their particular grid cell center
                local_x = (global_positions[index][obj][1] - grid_cells[index][obj][1]*self.cell_length) / self.half_cell
                local_y = (global_positions[index][obj][0] - grid_cells[index][obj][0]*self.cell_length) / self.half_cell


                pos_true = torch.as_tensor([local_y, local_x])
                dim_true = box_dims[index][obj]

                true_box = self.get_global_box(grid_cells[index][obj], pos_true, dim_true)

                if self.config['show_debug_plots']:
                    self.plot_box(true_box)


                # getting predicted positions and box dimensions from both bounding boxes in that grid cell
                local_pos_pred_b1 = y_pred[index, grid_cells[index][obj][1], grid_cells[index][obj][0], :2]
                local_pos_pred_b2 = y_pred[index, grid_cells[index][obj][1], grid_cells[index][obj][0], 2:4]

                dim_pred_b1 = y_pred[index, grid_cells[index][obj][1], grid_cells[index][obj][0], 4:6]
                dim_pred_b2 = y_pred[index, grid_cells[index][obj][1], grid_cells[index][obj][0], 6:8]

                pred_box_b1 = self.get_global_box(grid_cells[index][obj], local_pos_pred_b1, dim_pred_b1)
                pred_box_b2 = self.get_global_box(grid_cells[index][obj], local_pos_pred_b2, dim_pred_b2)

                b1_IOU = self.IOU(true_box, pred_box_b1)
                b2_IOU = self.IOU(true_box, pred_box_b2)

                indexing = [index, grid_cells[index][obj][1], grid_cells[index][obj][0]]
                # if self.config['show_debug_plots']:
                #     self.plot_box(pred_box_b2, False)
                #     self.plot_box(pred_box_b1, False)

                # If both boxes are bad, this will choose one at random
                # this prevent one box from being favored at the start of training
                if b2_IOU == b1_IOU:
                    random_pick = np.random.rand() > 0.5
                else:
                    random_pick = False


                if b2_IOU > b1_IOU or random_pick:
                    # print('B2 better')
                    # if self.config['show_debug_plots']:
                    # #     if  y_pred[indexing] [9] > 0.5:
                    #     self.plot_box(pred_box_b2, False)

                    obj_ij_pos[indexing] [2:4] = 1
                    obj_ij_dims[indexing] [6:8] = 1
                    obj_ij_conf[indexing] [9] = 1
                    noobj_ij_conf[indexing] [9] = 0

                    y_hat[indexing] [2:4] = pos_true
                    y_hat[indexing] [6:8] = dim_true
                    y_hat[indexing] [9] = b2_IOU

                else:
                    # print('B1 better')
                    # if self.config['show_debug_plots']:
                    # #     if  y_pred[indexing] [8] > 0.5:
                    #     self.plot_box(pred_box_b1, False)
                    #     print('\nConfidence:', y_pred[indexing] [8], "Actual:", b1_IOU)
                    #     print('elsewhere:', y_pred[index, 3, 3] [10])
                    #     plt.scatter(self.cell_length*3,self.cell_length*3)

                    obj_ij_pos[indexing] [:2] = 1
                    obj_ij_dims[indexing] [4:6] = 1
                    obj_ij_conf[indexing] [8] = 1
                    noobj_ij_conf[indexing] [8] = 0

                    y_hat[indexing] [:2] = pos_true
                    y_hat[indexing] [4:6] = dim_true
                    y_hat[indexing] [8] = b1_IOU

                obj_i_prob[indexing][10] = 1
                y_hat[indexing] [10] = 1.



                # print(local_pos_pred_b1)
                # print(local_pos_pred_b2)
                # print(dim_pred_b1)
                # print(dim_pred_b2)

            if self.config['show_debug_plots']:
                for i in range(0,224,self.cell_length):
                    plt.vlines(i+self.cell_length/2, 0, 224-1,'r')
                    plt.hlines(i+self.cell_length/2, 0, 224-1,'r')

                plt.show()

        mega_diff = (y_pred - y_hat) ** 2.

        loss = self.lambda_coord * (obj_ij_pos * mega_diff).sum() + \
               self.lambda_coord * (obj_ij_dims * mega_diff).sum() + \
               (obj_ij_conf * mega_diff).sum() + \
               self.lambda_noobj * (noobj_ij_conf * mega_diff).sum() + \
               (obj_i_prob * mega_diff).sum()
        loss /= y.size(0)
        return loss
