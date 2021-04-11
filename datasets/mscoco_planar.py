import os
import os.path
import glob 
import torch.utils.data as data
import torch
import cv2
import numpy as np
import torchvision.transforms as transforms

class ArrayToTensor(object):
    """Converts a numpy.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)."""

    def __call__(self, array):
        assert(isinstance(array, np.ndarray))

        # handle numpy array
        tensor = torch.from_numpy(array)

        # put it from HWC to CHW format
        # array = np.transpose(array, (2, 0, 1)) # if color image
        if tensor.dim() == 2: # gray-scale
            tensor = tensor.unsqueeze(0)
        return tensor.float()

def imgs_pose_loader(root,path_imgs,path_pose=None): # load img pairs and their poses
    imgs = [os.path.join(root,path) for path in path_imgs]

    if not path_pose==None:
        pose = np.loadtxt(os.path.join(root,path_pose)).astype(np.float32)
        return [cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(np.float32) for img in imgs],pose
    else:
        return [cv2.imread(img, cv2.IMREAD_GRAYSCALE).astype(np.float32) for img in imgs]

class ListDataset(data.Dataset):
    def __init__(self, root, path_list, input_transform, poseGT=True, loader=imgs_pose_loader, name_reqiured=False):

        self.root = root
        self.path_list = path_list
        self.input_transform = input_transform
        self.loader = loader
        self.poseGT = poseGT
        self.name_reqiured = name_reqiured

    def __getitem__(self, index):

        if not self.poseGT:
            inputs = self.path_list[index]
            inputs = self.loader(self.root, inputs)
        else:
            inputs, pose_target = self.path_list[index]
            inputs, pose_target = self.loader(self.root, inputs, pose_target)
            pose_target_tensor = torch.from_numpy(pose_target)

        if self.input_transform is not None:
            inputs[0] = self.input_transform(inputs[0])
            inputs[1] = self.input_transform(inputs[1])

        img_name = self.path_list[index][1].split('.')[0]

        if not self.poseGT:
            return inputs
        elif self.name_reqiured:
            return inputs, pose_target_tensor, img_name
        else:
            return inputs, pose_target_tensor

    def __len__(self):
        return len(self.path_list)


def make_dataset(dir, poseGT=True, absolutePose=True):
    '''Will search for triplets that go by the pattern '[name]-img_1.png [name]-img_2.png [name]-pose.txt' '''
    images = []

    # sort data according to number
    poseFilePaths = sorted(glob.glob(os.path.join(dir,'*/*-img_1.png')))

    poseFilePaths.sort(key = lambda fullPathandName: int(os.path.basename(fullPathandName).split('-')[0].split('.')[0].split('_')[2]))


    for img1_file in poseFilePaths:
        img1_file_basename = os.path.basename(img1_file)
        folder = img1_file.split('/')[-2] 
        img1_file = os.path.join(folder, img1_file_basename) 
        root_filename = img1_file[:-10] #
        
        img1 = root_filename+'-img_1.png'
        img2 = root_filename+'-img_2.png'

        if poseGT:
            pose_GT = root_filename+'-absolutePose.txt'
        
        if not (os.path.isfile(os.path.join(dir,img1)) and os.path.isfile(os.path.join(dir,img2))):
            print(img1, 'not good!')
            continue

        if poseGT:
            images.append([[img1,img2],pose_GT])
        else:
            images.append([[img1,img2]])

    return images


def mscoco_planar(root_train, root_validate, root_test, poseGT=True, absolutePose=True):

    grayscale_input_transform = transforms.Compose([
        ArrayToTensor(), # transform numpy array to PyTorch tensor
        transforms.Normalize(mean=[0], std=[255])
    ])

    train_list = make_dataset(root_train, poseGT, absolutePose)
    validate_list = make_dataset(root_validate, poseGT, absolutePose)
    test_list = make_dataset(root_test, poseGT, absolutePose)

    train_dataset = ListDataset(root_train, train_list, grayscale_input_transform, poseGT, loader=imgs_pose_loader, name_reqiured=False)
    validate_dataset = ListDataset(root_validate, validate_list, grayscale_input_transform, poseGT, loader=imgs_pose_loader, name_reqiured=False)
    test_dataset = ListDataset(root_test, test_list, grayscale_input_transform, poseGT, loader=imgs_pose_loader, name_reqiured=True) 
    
    return train_dataset, validate_dataset, test_dataset # return data.Dataset class object 