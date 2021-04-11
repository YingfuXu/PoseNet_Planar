import torch
import torch.utils.data
# from torchsummary import summary
import time
import math
import numpy as np
import os

from util import charbonnier, absolutePose2homo8Pose, AverageMeter, Euler2Rot
from params import args, device

def test_pose(test_loader, PoseNet_model, save_path, test_writer=None):

    test_endPoint_loss = AverageMeter()
    warping_time = AverageMeter()
    net_time = AverageMeter()

    epoch_size = len(test_loader) if args.epoch_size == 0 else min(len(test_loader), args.epoch_size)

    # switch to evaluate mode
    PoseNet_model.eval() 

    result_list_1to2 = []
    result_list_2to1 = []
    img_name_list = []
    
    for batch_index, (inputImgs, targetPoses, imgName) in enumerate(test_loader):

        img_name_list.append(imgName)

        batch_size = targetPoses.size()[0]
        assert(batch_size == 1) # for now forced to be 1

        absolute_poseGT = targetPoses.to(device) # torch.Size([8, 2, 320, 448])

        # # NOTE this is the pose at the start time point of exposure of an image
        # pose_img1 = absolute_poseGT[:, 0, :] # torch.Size([batch_size, 6])
        # pose_img2 = absolute_poseGT[:, 2, :]
        # NOTE can be the average value of the start and end point of exposure.
        pose_img1 = (absolute_poseGT[:, 0, :] + absolute_poseGT[:, 1, :]) / 2; # torch.Size([batch_size, 6])
        pose_img2 = (absolute_poseGT[:, 2, :] + absolute_poseGT[:, 3, :]) / 2; 
        
        input_img1 = inputImgs[0].to(device)
        input_img2 = inputImgs[1].to(device)

        # 1. ground truth # torch.Size([batch_size, 8])
        homo8_1to2_GT,homo8_1to2_initial = absolutePose2homo8Pose(pose_img1, pose_img2, batch_size, rot_random_bias=0.0, slope_random_bias=0.0) # NOTE train without attitude noise
        homo8_2to1_GT,homo8_2to1_initial = absolutePose2homo8Pose(pose_img2, pose_img1, batch_size, rot_random_bias=0.0, slope_random_bias=0.0)

        # 2. nn forward
        homo8_1to2_nn, warpingTimer1, netTimer1 = PoseNet_model(input_img1,input_img2,homo8_1to2_initial)
        homo8_2to1_nn, warpingTimer2, netTimer2 = PoseNet_model(input_img2,input_img1,homo8_2to1_initial)
        
        if batch_index > 6: # not counting the first a few
            warping_time.update((warpingTimer1+warpingTimer2)*0.5, n=batch_size)
            net_time.update((netTimer1+netTimer2)*0.5, n=batch_size)
        
        # 3. error
        homo8_1to2_blockError = homo8_1to2_GT - homo8_1to2_nn
        homo8_2to1_blockError = homo8_2to1_GT - homo8_2to1_nn
        
        endPoint_loss = torch.mean(charbonnier(homo8_1to2_blockError[:, 5:])) + torch.mean(charbonnier(homo8_2to1_blockError[:, 5:]))
        test_endPoint_loss.update(endPoint_loss.item(), n=batch_size)

        # composed_RotMtrx_1to2 rotates a vector from world frame to cam2 frame
        composed_RotMtrx_1to2 = torch.mm(Euler2Rot(homo8_1to2_initial[0, 2], homo8_1to2_initial[0, 3], homo8_1to2_initial[0, 4]), \
                                         Euler2Rot(homo8_1to2_initial[0, 0], homo8_1to2_initial[0, 1], torch.zeros(1).squeeze().to(device)))
        transWorld_1to2_GT = torch.mm(torch.t(composed_RotMtrx_1to2), homo8_1to2_GT[:, 5:].view(3, 1))
        transWorld_1to2_nn = torch.mm(torch.t(composed_RotMtrx_1to2), homo8_1to2_nn[:, 5:].view(3, 1))
        result_list_1to2.append(torch.cat((transWorld_1to2_GT.view(1, 3), transWorld_1to2_nn.view(1, 3)),dim=1))

        composed_RotMtrx_2to1 = torch.mm(Euler2Rot(homo8_2to1_initial[0, 2], homo8_2to1_initial[0, 3], homo8_2to1_initial[0, 4]), \
                                         Euler2Rot(homo8_2to1_initial[0, 0], homo8_2to1_initial[0, 1], torch.zeros(1).squeeze().to(device)))
        transWorld_2to1_GT = torch.mm(torch.t(composed_RotMtrx_2to1), homo8_2to1_GT[:, 5:].view(3, 1))
        transWorld_2to1_nn = torch.mm(torch.t(composed_RotMtrx_2to1), homo8_2to1_nn[:, 5:].view(3, 1))
        result_list_2to1.append(torch.cat((transWorld_2to1_GT.view(1, 3), transWorld_2to1_nn.view(1, 3)),dim=1))

        if batch_index % args.print_freq == 0:
            print('Test: [{0}/{1}]\t Time {2}\t {3}\t Bidirection_Loss {4}'
                .format(batch_index, epoch_size, warping_time, net_time, test_endPoint_loss))

        if batch_index > epoch_size:
            break

    # save    
    result_tensor_1to2 = torch.cat(result_list_1to2, dim=0)
    result_array_1to2 = np.array(result_tensor_1to2.cpu())
    result_tensor_2to1 = torch.cat(result_list_2to1, dim=0)
    result_array_2to1 = np.array(result_tensor_2to1.cpu())

    if batch_index > 6:
        print('WarpingTime(ms):', warping_time.avg, 'NetworkTime(ms):', net_time.avg, 'Total fps:', 1000.0/(warping_time.avg+net_time.avg))
    print(' * EPE loss test: {:.3f}'.format(test_endPoint_loss.avg))

    # save to dataset training result path
    test_save_path = save_path+'/test'
    if not os.path.exists(test_save_path):
        os.makedirs(test_save_path)
    np.savetxt(test_save_path+'/result_1to2.txt', result_array_1to2)
    np.savetxt(test_save_path+'/result_2to1.txt', result_array_2to1)
    with open(test_save_path+'/filename_list.txt', 'w') as f2:
        for item in img_name_list:
            f2.write("%s\n" % item)

    return test_endPoint_loss.avg