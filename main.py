import os
import torch
import torch.nn.functional as F
import torch.optim
import torch.utils.data
# from torchsummary import summary
import sys
import datetime
from tensorboardX import SummaryWriter # tensorboard --logdir=/path/to/checkoints

import models
import datasets
from params import args, device # NOTE where the parameters are defined (args)
from train import train_pose, validate_pose
from test import test_pose
from util import save_checkpoint
from run_UZHFPV import flight_dataset_test

# NOTE translation ONLY! for now

def main():
    global args
    best_pose_EPE = -1 # initialize

    # save important parameters in the save path
    save_path = '{},{}{}epochs{},b{},lr_{},block_{},drop_{}'.format(
        args.arch,
        'self-sup,' if args.self_supervised else 'sup,',
        args.epochs, # total training epochs
        ',epochSize'+str(args.epoch_size) if args.epoch_size > 0 else '',
        args.batch_size,
        args.learning_rate,
        # args.normalization, # norm_{},
        args.number_block,
        # args.block_net_type, # blockType_{},
        # args.share_fc, # # 'shareFC,' if args.share_fc else '',
        # args.share_conv, # 'shareConv,' if args.share_conv else '',
        args.fc_dropout_rate
        )

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = os.path.join(timestamp,save_path)
    save_path = os.path.join('training_output',save_path)
    save_path = os.path.join(args.dataset_dir,save_path) # save data generated in training to the dir of dataset
    if args.save_train_log:
        print('=> will save everything to {}'.format(save_path))

    if args.save_train_log:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        train_writer = SummaryWriter(os.path.join(save_path,'train'))
        validate_writer = SummaryWriter(os.path.join(save_path,'validate'))
    else:
        train_writer = []
        validate_writer = []

    # Data loading code
    print("=> fetching img pairs in '{}'".format(args.dataset_dir))
    print("=> call function '{}' to make the dataset".format(args.dataset))

    train_set, validate_set, test_set = datasets.__dict__[args.dataset]( 
        os.path.join(args.dataset_dir,args.train_folder),
        os.path.join(args.dataset_dir,args.validate_folder),
        os.path.join(args.dataset_dir,args.test_folder),
        poseGT=True, absolutePose=True
    )
    print('{} samples found, {} train samples, {} validate samples, and {} test samples '
            .format(len(validate_set)+len(train_set)+len(test_set),len(train_set),len(validate_set),len(test_set)))

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=True)
    
    validate_loader = torch.utils.data.DataLoader(
        validate_set, batch_size=args.batch_size,
        num_workers=args.workers, pin_memory=True, shuffle=False)

    test_loader = torch.utils.data.DataLoader(  
        test_set, batch_size=1, # NOTE to get the inference time, for now batch_size is forced to equal to 1
        num_workers=args.workers, pin_memory=True, shuffle=False)

    if args.use_pretrained or args.test or args.flight:
        pretrained_model_data = torch.load(args.model_path, map_location=torch.device('cpu'))
        args.arch = pretrained_model_data['arch'] # the network arch is depending on the loaded model
        print("=> using pre-trained model '{}', '{}'".format(args.arch, args.model_path))
    else:
        pretrained_model_data = None
        print("=> creating model '{}'".format(args.arch)) # 

    PoseNet_model = models.__dict__[args.arch](args.list_blocks_types, args.img_height, args.img_width, device, 
                                               args.self_supervised, pretrained_model=pretrained_model_data, 
                                               normalization=args.normalization, share_fc_weights=args.share_fc, share_conv_weights=args.share_conv, 
                                               fc_dropout_rate=args.fc_dropout_rate, show_img=args.show_img, trace_model=args.trace_model
                                               ).to(device)

    print("=> creating pose model '{}'".format(args.arch)) # 
 
    if args.flight: # UZH FPV dataset
        with torch.no_grad():
            PoseNet_model.eval()
            flight_dataset_test(PoseNet_model, device)
        return
    
    if args.test: # test set
        with torch.no_grad():
            PoseNet_model.eval()
            test_endPoint_loss_avg = test_pose(test_loader, PoseNet_model, save_path)
            print("Testing Average Loss:", test_endPoint_loss_avg)
        print("Test finished! Exit...")
        return

    if args.trace_model: # TROCH SCRIPT for cpp # NOTE only tested using cpu
        # An example input you would normally provide to your model's forward() method.
        img1 = torch.ones(1, 1, 224, 320)
        img2 = torch.ones(1, 1, 224, 320)
        homo8 = torch.ones(1, 8)
        PoseNet_model.eval()
        # print(PoseNet_model(img1,img2,homo8))
        traced_script_module = torch.jit.trace(PoseNet_model, (img1, img2, homo8)) # Use torch.jit.trace to generate a torch.jit.ScriptModule via tracing.
        traced_script_module.save(args.arch+"_traced_model.pt")
        print("Traced model file:", args.arch+"_traced_model.pt")
        return

    assert(args.solver in ['adam', 'sgd'])
    print('=> setting {} solver'.format(args.solver))
    param_groups = [{'params': PoseNet_model.bias_parameters(), 'weight_decay': args.bias_decay}, # PoseNet_model.module
                    {'params': PoseNet_model.weight_parameters(), 'weight_decay': args.weight_decay}] # PoseNet_model.module

    if args.solver == 'adam':
        pose_train_optimizer = torch.optim.Adam(param_groups, args.learning_rate,
                                                betas=(args.momentum, args.beta))
    elif args.solver == 'sgd':
        pose_train_optimizer = torch.optim.SGD(param_groups, args.learning_rate,
                                               momentum=args.momentum)

    pose_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(pose_train_optimizer, milestones=args.milestones, gamma=0.5)
    
    n_iter = 0

    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        print("\nLearning rate in epoch", epoch, ":", pose_train_optimizer.state_dict()['param_groups'][0]['lr'])
        loss_train, n_iter = train_pose(train_loader, PoseNet_model, pose_train_optimizer, epoch, train_writer, n_iter)

        pose_lr_scheduler.step() 
        
        if args.save_train_log:
            train_writer.add_scalar('trans_loss_epoch', loss_train, epoch)

        # evaluate on validation set 
        with torch.no_grad():
            loss_validate = validate_pose(validate_loader, PoseNet_model, epoch, validate_writer)
        
        if args.save_train_log:
            validate_writer.add_scalar('trans_loss_epoch', loss_validate, epoch)

        current_EPE = loss_validate
        train_EPE = loss_train

        if best_pose_EPE < 0:
            best_pose_EPE = current_EPE # 

        is_best = current_EPE < best_pose_EPE

        if is_best:
            print(" * best validate loss:", current_EPE)
            best_pose_EPE = current_EPE
            best_model = PoseNet_model

            if args.save_model:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch, 
                    'state_dict': PoseNet_model.state_dict(),
                    'best_EPE': best_pose_EPE
                }, is_best, save_path,
                filename='checkpoint,epoch{},val_EPE{},train_EPE{}.pth.tar'.format(epoch, best_pose_EPE, train_EPE))

        elif epoch == args.epochs-1: # save the model after the last epoch
            print(" * final validate loss:", current_EPE)
            final_pose_EPE = current_EPE
            if args.save_model:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch, 
                    'state_dict': PoseNet_model.state_dict(),
                    'best_EPE': final_pose_EPE
                }, True, save_path,
                filename='checkpoint,epoch{},val_EPE{},train_EPE{}.pth.tar'.format(epoch, final_pose_EPE, train_EPE))
    
    print("Training Finished!")

    print("Testing the best validated model!")
    with torch.no_grad():
        test_endPoint_loss_avg = test_pose(test_loader, best_model, save_path)
        print("Testing Average Loss:", test_endPoint_loss_avg)
    print("Test finished! Exit...")

if __name__ == '__main__':
    main()