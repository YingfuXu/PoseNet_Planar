clear all;
% clc;

%% mother image dataset
filePath = '/home/yingfu/datasets/ms-coco/';
datasetPath=[filePath, 'test2014/']; % train2014  test2014
dataset = fullfile(datasetPath);
dirOutput=dir(fullfile(dataset,'*.jpg'));
fileNames={dirOutput.name}';
num_motherImage = size(fileNames, 1);

%% virtual camera
global widthPixelSizeCamN heightPixelSizeCamN;
widthPixelSizeCamN = 320;
heightPixelSizeCamN = 224;

IntrinsicMatrixCn = ...
    [(widthPixelSizeCamN-1)/2 0 (widthPixelSizeCamN-1)/2; 
    0 (widthPixelSizeCamN-1)/2 (heightPixelSizeCamN-1)/2; 
    0 0 1]; % width FoV = 90 deg

global xy1; % [3*numPixel]
[u_grid,v_grid] = meshgrid(0:widthPixelSizeCamN-1, 0:heightPixelSizeCamN-1); 

uv1 = [reshape(u_grid', [1, numel(u_grid)]); reshape(v_grid', [1, numel(v_grid)]); ones(1, numel(v_grid))];
xy1 = IntrinsicMatrixCn^(-1) * uv1;

dt = 0.0001; % Kinetic integration
fps = 30;
exp = 0.01; % exposure duration
% exp = dt; %  no blur
exp_steps = round(exp/dt); % exposure steps
interval_steps = round(1/fps/dt); % interval steps

single_axis_vel_over_d_max = 7.5; % maximum distance-scaled velocity along each axis
single_axis_angular_rate_max = pi; % maxmum angular rate along each axis

% initialize mats to save data
Img1_batch = zeros(1, heightPixelSizeCamN, widthPixelSizeCamN);
poseImg1_batch = zeros(1, 6);
poseImg1_end_batch = zeros(1, 6);
Img2_batch = zeros(1, heightPixelSizeCamN, widthPixelSizeCamN);
poseImg2_batch = zeros(1, 6);
poseImg2_end_batch = zeros(1, 6);

img1_poses = zeros(exp_steps, 6);
img2_poses = zeros(exp_steps, 6);

%% loop the dataset
img_num = 1;
while(img_num <= num_motherImage) % generate one pair on one mother image
    
    %% ramdom initialize cam pose and start dynamics
    
    % uniform distribution for each image
    init_Euler = [(rand-0.5)*2*25, (rand-0.5)*2*25, (rand-0.5)*2*180]' / 180 * pi; % roll [-25 25] pitch [-25 25] yaw [-180 180] 
    
    % generate velocity and angular rate based on the boundaries
    % distance0scaled velocity expressed in the heading frame (rotate with yaw angle from world frame)
    velHeading_over_distance_vec = [single_axis_vel_over_d_max * 2*(rand(2, 1)-0.5); 0.5*single_axis_vel_over_d_max * 2*(rand(1, 1)-0.5)];  
    rate_body_vec = [single_axis_angular_rate_max * 2*(rand(2, 1)-0.5); 0.5*single_axis_angular_rate_max * 2*(rand(1, 1)-0.5)];
   
    init_position = [0 0 -1]';  % (z-axis down)
    
    poseImg1_start = [init_Euler', init_position'];
    pose(1, :) = poseImg1_start;
    img1_poses(1, :) = poseImg1_start;
    
    % calculate the point on the ground that the camera is pointing at at the beginning
    bRw = R_bw(init_Euler(1), init_Euler(2), init_Euler(3));
    wRb = bRw';
    sonar_dist = 1 / wRb(3,3); % when vertical_distance == 1
    img1Start_camPointing_posi = init_position + wRb * [0; 0; sonar_dist]; % a point lying on the plane
    
    goodPair = 0;
    
	%% kinetic loop
    img1Count = 1;
    img2Count = 0;
    for i = 2 : exp_steps+interval_steps
        
        dx = fn_dx(pose(i-1, :), rate_body_vec, velHeading_over_distance_vec); % 12 d Column vector
        pose(i, :) = pose(i-1, :) + dx' * dt;
        
        Euler = pose(i, 1:3)';
        position = pose(i, 4:6)';
        
        if i <= exp_steps
            img1_poses(i, :) = [Euler', position']; % save the camera pose within exposure duration into img1_poses
            img1Count = img1Count + 1;
            if i == exp_steps
                poseImg1_end = [Euler', position'];
            end
        end
        
        if i == interval_steps+1
            poseImg2_start = [Euler', position'];
        end
        
        if i > interval_steps
            img2_poses(i-interval_steps, :) = [Euler', position'];
            img2Count = img2Count + 1;
            if i == exp_steps+interval_steps
                poseImg2_end = [Euler', position'];
            end
        end
        
    end
    
    % calculate the point on the ground that the camera is pointing at in the end
    bRw = R_bw(Euler(1), Euler(2), Euler(3));
    wRb = bRw';    
    sonar_dist = - position(3) / wRb(3,3); 
    img2End_camPointing_posi = position + wRb * [0; 0; sonar_dist]; % a point lying on the plane
    
    % move the camera to make the camera point at the center of the mother image at the middle time point of exposure 
    position_offset = - (img1Start_camPointing_posi + img2End_camPointing_posi) / 2;
    
    poseImg1_start(:, 4:6) = poseImg1_start(:, 4:6) + position_offset';
    poseImg1_end(:, 4:6) = poseImg1_end(:, 4:6) + position_offset';
        
    poseImg2_start(:, 4:6) = poseImg2_start(:, 4:6) + position_offset';
    poseImg2_end(:, 4:6) = poseImg2_end(:, 4:6) + position_offset';
    
    img1_poses(:, 4:6) = img1_poses(:, 4:6) + position_offset';
    img2_poses(:, 4:6) = img2_poses(:, 4:6) + position_offset';
    
    %% read a new image from ms-coco
    imgName = fileNames{img_num, 1};
    mother_img = imread([dataset imgName]); % read image
    if size(mother_img, 3) == 3
        mother_img = rgb2gray(mother_img);
    end
    
    widthBasePixel = size(mother_img, 2);
    heightBasePixel = size(mother_img, 1);
    % a simulated camera is looking at the mother image orthogonally at the height == 1,
    % what the simulated camera captures is then exactly the mother image
    IntrinsicMatrixC0 = ... 
        [(widthBasePixel-1)/2 0 (widthBasePixel-1)/2; 
        0 (widthBasePixel-1)/2 (heightBasePixel-1)/2; 
        0 0 1]; % width FoV = 90 deg
    
    sumImg1 = 0;
    sumImg2 = 0;
    %% reprojection according to the camera poses within exposure duration
    enlarge_flag = 0;
    goodImgFlag = 0;
    distance_img1 = 0.25;
    
    while(~goodImgFlag)
        
        if distance_img1 < 0.1  && enlarge_flag == 0 
            % if the simuated camera is too close to the plane, then enlarge the mother image
            mother_img_enlarged = [mother_img, mother_img, mother_img; mother_img, mother_img, mother_img; mother_img, mother_img, mother_img];
            widthBasePixel_enlarged = size(mother_img_enlarged, 2);
            heightBasePixel_enlarged = size(mother_img_enlarged, 1);
            IntrinsicMatrixC0 = ... % 
                [(widthBasePixel_enlarged-1)/2 0 (widthBasePixel_enlarged-1)/2; 
                0 (widthBasePixel_enlarged-1)/2 (heightBasePixel_enlarged-1)/2; 
                0 0 1]; % width FoV = 90 deg
            enlarge_flag = 1;
            mother_img = mother_img_enlarged;
        end
    
        for exp_img_num = 1 : size(img1_poses, 1)
            [imgCn, cn_H_c0, goodImgFlag] = fn_imgCn(IntrinsicMatrixC0, mother_img, img1_poses(exp_img_num, 1:3)', distance_img1*img1_poses(exp_img_num, 4:6)'); 
            % sample an image of this pose
            if ~goodImgFlag
                break;
            end
            
            sumImg1 = sumImg1 + imgCn / exp_steps;
            [imgCn, cn_H_c0, goodImgFlag] = fn_imgCn(IntrinsicMatrixC0, mother_img, img2_poses(exp_img_num, 1:3)', distance_img1*img2_poses(exp_img_num, 4:6)');
            if ~goodImgFlag
                break;
            end
            
            sumImg2 = sumImg2 + imgCn / exp_steps;
        end

        if ~goodImgFlag % if cannot collect good images at this height, then reduce the height and restart
            sumImg1 = 0;
            sumImg2 = 0;
            distance_img1 = distance_img1 * 0.9;
        end
        
    end
    
    Img1 = sumImg1;
    Img2 = sumImg2;

    % textue check
    sum_gradient_1 = sum(abs(imgradient(Img1)),'all');
    sum_gradient_2 = sum(abs(imgradient(Img2)),'all');
    if sum_gradient_1 < 5.0e+05 && sum_gradient_2 < 5.0e+05
        goodPair = goodPair + 1;
    end
    
    if goodPair == 1
        fn_saveData(filePath, imgName, Img1, poseImg1_start, poseImg1_end, Img2, poseImg2_start, poseImg2_end);
        
        msg_char = imgName + " done! number: " + (img_num)
        img_num = img_num + 1;
    else
        msg_char = "goodPair: " + (goodPair) + " Redo " + img_num;
    end

end



