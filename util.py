import os
import numpy as np
import shutil
import torch
import math
import time
import torch.nn as nn

D2R = math.pi / 180.0


def save_checkpoint(state, is_best, save_path, filename='checkpoint.pth.tar'):
    # torch.save(state, os.path.join(save_path,filename))
    if is_best:
        print("Saving Model Params ...")
        torch.save(state, os.path.join(save_path,filename))
        # shutil.copyfile(os.path.join(save_path,filename), os.path.join(save_path,'model_best.pth.tar'))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __repr__(self):
        return '{:.3f} ({:.3f})'.format(self.val, self.avg)


def charbonnier(x, alpha=0.5, epsilon=1.e-9):
    return torch.pow((torch.pow(x,2)+epsilon**2), alpha)


def Rot2Euler(rot):

    assert(rot.size() == torch.Size([3, 3]))

    sy = torch.sqrt(rot[1, 2]*rot[1, 2] + rot[2, 2]*rot[2, 2])

    if sy < 1e-6:
        print("Pitch is close to 90 degrees! ")
        yaw = torch.FloatTensor([0.0]).squeeze()
        roll = torch.atan2(-rot[2, 1], rot[1, 1])
    else:
        yaw = torch.atan2(rot[0, 1], rot[0, 0])
        roll = torch.atan2(rot[1, 2], rot[2, 2])

    pitch = torch.atan2(-rot[0, 2], sy)

    return torch.cat((roll.unsqueeze(0), pitch.unsqueeze(0), yaw.unsqueeze(0)), dim=0)


def Euler2Rot(roll, pitch, yaw): 

    RotMtrx = torch.stack(( \
        (torch.cos(pitch)*torch.cos(yaw)),
        (torch.cos(pitch)*torch.sin(yaw)),
        (-torch.sin(pitch)),
        (torch.cos(yaw)*torch.sin(pitch)*torch.sin(roll) - torch.cos(roll)*torch.sin(yaw)),
        (torch.cos(roll)*torch.cos(yaw) + torch.sin(pitch)*torch.sin(roll)*torch.sin(yaw)),
        (torch.cos(pitch)*torch.sin(roll)),
        (torch.sin(roll)*torch.sin(yaw) + torch.cos(roll)*torch.cos(yaw)*torch.sin(pitch)),
        (torch.cos(roll)*torch.sin(pitch)*torch.sin(yaw) - torch.cos(yaw)*torch.sin(roll)),
        (torch.cos(pitch)*torch.cos(roll))), dim=0).view(3, 3)

    return RotMtrx

def poseVec2RotMtrxAndPlaneNormVecSingle(homo8_tensor): # for cpp, trace, batch_size == 1

    RotMtrx_list = []
    NormalVector_list = []
    trans_img1_list = []

    roll_img1 = homo8_tensor[0, 0]
    pitch_img1 = homo8_tensor[0, 1]
    roll_re = homo8_tensor[0, 2]
    pitch_re = homo8_tensor[0, 3]
    yaw_re = homo8_tensor[0, 4]
    Rot_re = Euler2Rot(roll_re, pitch_re, yaw_re)
    img1_NormalVector_plane = torch.stack((-torch.sin(pitch_img1), (torch.cos(pitch_img1)*torch.sin(roll_img1)), (torch.cos(pitch_img1)*torch.cos(roll_img1))), dim=0).view(1, 3)
    RotMtrx_list.append(Rot_re.unsqueeze(0)) # [1 3 3]
    NormalVector_list.append(img1_NormalVector_plane) # [1 3]
    trans_img1_list.append( torch.mm( torch.t(Rot_re), homo8_tensor[0, 5:].view(3, 1) ).view(1, 3) )

    batch_Rot_re = torch.cat(RotMtrx_list, dim=0)
    batch_img1_NormalVector_plane = torch.cat(NormalVector_list, dim=0)
    batch_trans_img1 = torch.cat(trans_img1_list, dim=0)

    # homoMtrx = Rot_re + torch.mm(tVec, img1_NormalVector_plane) 
    return batch_Rot_re, batch_img1_NormalVector_plane, batch_trans_img1

def poseVec2RotMtrxAndPlaneNormVec(homo8_tensor):

    batch_size = homo8_tensor.size()[0]

    RotMtrx_list = []
    NormalVector_list = []
    trans_img1_list = []

    for i in range(batch_size):
        roll_img1 = homo8_tensor[i, 0] # img1 camera frame relative to the world frame (z-axis is orthogonal to the plane)
        pitch_img1 = homo8_tensor[i, 1]

        roll_re = homo8_tensor[i, 2]
        pitch_re = homo8_tensor[i, 3]
        yaw_re = homo8_tensor[i, 4]

        Rot_re = Euler2Rot(roll_re, pitch_re, yaw_re)
        img1_NormalVector_plane = torch.stack((-torch.sin(pitch_img1), (torch.cos(pitch_img1)*torch.sin(roll_img1)), (torch.cos(pitch_img1)*torch.cos(roll_img1))), dim=0).view(1, 3)

        RotMtrx_list.append(Rot_re.unsqueeze(0)) # [1 3 3]
        NormalVector_list.append(img1_NormalVector_plane) # [1 3]
        trans_img1_list.append( torch.mm( torch.t(Rot_re), homo8_tensor[i, 5:].view(3, 1) ).view(1, 3) )

    batch_Rot_re = torch.cat(RotMtrx_list, dim=0)
    batch_img1_NormalVector_plane = torch.cat(NormalVector_list, dim=0)
    batch_trans_img1 = torch.cat(trans_img1_list, dim=0)

    # homoMtrx = Rot_re + torch.mm(tVec, img1_NormalVector_plane) 
    return batch_Rot_re, batch_img1_NormalVector_plane, batch_trans_img1


def poseVec2homoMtrx(homo8_tensor): # size [8] , planeNormalVec=None, yaw_ground=None

    roll_img1 = homo8_tensor[0] # img1 camera frame relative to the world frame
    pitch_img1 = homo8_tensor[1]
    roll_re = homo8_tensor[2]
    pitch_re = homo8_tensor[3]
    yaw_re = homo8_tensor[4]
    # tVecX = homo8_tensor[5] # point from img2Cam to img1Cam expressed in img2Cam frame, scaled by the distance from img1Cam to the ground
    # tVecY = homo8_tensor[6]
    # tVecZ = homo8_tensor[7]
                
    # end = time.time()
    Rot_re = Euler2Rot(roll_re, pitch_re, yaw_re)  # rotate a vector from img1Cam frame to img2Cam frame
    # print("Euler2Rot", (time.time() - end)*1000.0)

    tVec = homo8_tensor[5:].view(3, 1)

    # the unit normal vector of the plane, expressed in the img1Cam frame 
    # img1_NormalVector_plane = torch.mm(Rot_img1, planeNormalVec)  # img1_NormalVector_plane = Rot_img1[:, 2]
    img1_NormalVector_plane = torch.stack((-torch.sin(pitch_img1), (torch.cos(pitch_img1)*torch.sin(roll_img1)), (torch.cos(pitch_img1)*torch.cos(roll_img1))), dim=0)

    homoMtrx = Rot_re + torch.mm(tVec, img1_NormalVector_plane.view(1, 3)) 

    # https://blog.csdn.net/heyijia0327/article/details/53782094
    return homoMtrx # [3, 3] 


def absolutePose2homo8Pose(batch_pose1, batch_pose2, batch_size, rot_random_bias, slope_random_bias, only_disturb_yaw=True):

    assert(batch_size == batch_pose1.size()[0])
    assert(batch_size == batch_pose2.size()[0])

    homoPose_list = []
    disturbed_homoPose_list = []

    for sample in range(batch_size):

        pose1 = batch_pose1[sample, :]
        pose2 = batch_pose2[sample, :]

        c1Euler = pose1[0:3]
        Pc1 = pose1[3:]
        c2Euler = pose2[0:3]
        Pc2 = pose2[3:]

        Rot_c1 = Euler2Rot(c1Euler[0], c1Euler[1], c1Euler[2])
        Rot_c2 = Euler2Rot(c2Euler[0], c2Euler[1], c2Euler[2])
        Rot_re = torch.mm(Rot_c2, torch.t(Rot_c1)) # rotate a vector from img1Cam frame to img2Cam frame
        re_Euler = Rot2Euler(Rot_re)

        tVecWorld = Pc1 - Pc2
        tVecCam2 = torch.mm(Rot_c2, tVecWorld.view(3,1))
        tVecCam2_d = tVecCam2 / torch.abs(Pc1[2]) # # point from img2Cam to img1Cam expressed in img2Cam frame, scaled by the distance from img1Cam to the ground

        # img1_NormalVector_plane = R_bw(c1Euler(1), c1Euler(2), c1Euler(3)) * [0 0 1]' # expressed in the img1Cam frame
        # Homo_Matrix = Rot_re + torch.mm(tVecCam2_d, img1_NormalVector_plane')

        homo_pose = torch.cat((c1Euler[0].unsqueeze(0), c1Euler[1].unsqueeze(0), re_Euler, tVecCam2_d.view(3)), dim=0)
        
        if only_disturb_yaw:
            disturbed_homo_pose = torch.cat(( \
                c1Euler[0].unsqueeze(0)+slope_random_bias*D2R*2.0*(torch.rand(1).to(homo_pose.device)-0.5), \
                c1Euler[1].unsqueeze(0)+slope_random_bias*D2R*2.0*(torch.rand(1).to(homo_pose.device)-0.5), \
                re_Euler[0:2], \
                re_Euler[2:]+rot_random_bias*D2R*2.0*(torch.rand(1).to(homo_pose.device)-0.5), \
                torch.zeros(3).to(homo_pose.device)), dim=0)
        else:
            disturbed_homo_pose = torch.cat(( \
                c1Euler[0].unsqueeze(0)+slope_random_bias*D2R*2.0*(torch.rand(1).to(homo_pose.device)-0.5), \
                c1Euler[1].unsqueeze(0)+slope_random_bias*D2R*2.0*(torch.rand(1).to(homo_pose.device)-0.5), \
                re_Euler+rot_random_bias*D2R*2.0*(torch.rand(3).to(homo_pose.device)-0.5), \
                torch.zeros(3).to(homo_pose.device)), dim=0)            
        # print(disturbed_homo_pose.shape) # torch.Size([8])

        homoPose_list.append(homo_pose.unsqueeze(0))
        disturbed_homoPose_list.append(disturbed_homo_pose.unsqueeze(0))
    
    batch_homoPose = torch.cat(homoPose_list, dim=0)
    batch_disturbed_homoPose = torch.cat(disturbed_homoPose_list, dim=0)

    return batch_homoPose, batch_disturbed_homoPose


def compose_trans(batch_size, batch_delta_trans, batch_homo_pose, rotMtrx):
    # homo_pose_1->2 = [slope_roll_img1, slope_pitch_img1, re_roll_1->2, re_pitch_1->2, re_yaw_1->2, t_vec_x/d1, t_vec_y/d1, t_vec_z/d1] (t_vec: 2->1 in 2 frame)
    delta_trans_list = []
    for i in range(batch_size): # torch.t(A) Transpose matrix A
        # the nn predict the trans_vec in warped_img2 frame (same as img1Cam frame). rotate it to img2 frame.
        delta_trans_img2_frame = torch.mm(rotMtrx[i, :, :], batch_delta_trans[i, :].view(3, 1))
        delta_trans_list.append(delta_trans_img2_frame.view(1, 3))
    batch_updated_homo_pose = batch_homo_pose.clone() # NOTE inplace operation not safe in training, so clone
    batch_updated_homo_pose[:, 5:] = batch_homo_pose[:, 5:] + torch.cat(delta_trans_list, dim=0)
    return batch_updated_homo_pose


def compose_trans_single(batch_size, batch_delta_trans, batch_homo_pose, rotMtrx):
    # assert(batch_size == 1) # it causes TracerWarning ...
    delta_trans_img2_frame = torch.mm(rotMtrx[0, :, :], batch_delta_trans[0, :].view(3, 1)) 
    batch_homo_pose[0, 5:] = batch_homo_pose[0, 5:] + delta_trans_img2_frame.view(3)
    return batch_homo_pose

    
def downsampling(pyramid_level, batch_stacked_input_imgs):
    if torch.__version__ < '0.4.0': 
        return nn.functional.interpolate(batch_stacked_input_imgs, size=None, scale_factor=0.5**pyramid_level, mode='bilinear')
    else:
        return nn.functional.interpolate(batch_stacked_input_imgs, size=None, scale_factor=0.5**pyramid_level, mode='bilinear', align_corners=True)