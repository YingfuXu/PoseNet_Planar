function [goodPair] = fn_saveData(filePath, imgName, img1, poseImg1, poseImg1_end, img2, poseImg2, poseImg2_end)

folder = 'random/'; % slow medium fast 

savePath = [filePath, 'posenet/',folder]; % posenet_validate posenet_train

if ~exist(savePath, 'dir')
    mkdir(savePath)
end

fid = fopen(sprintf([savePath, '%s-absolutePose.txt'], imgName),'wt');
fprintf(fid,'%f %f %f %f %f %f\n',poseImg1(1), poseImg1(2), poseImg1(3), poseImg1(4), poseImg1(5), poseImg1(6));
fprintf(fid,'%f %f %f %f %f %f\n',poseImg1_end(1), poseImg1_end(2), poseImg1_end(3), poseImg1_end(4), poseImg1_end(5), poseImg1_end(6));
fprintf(fid,'%f %f %f %f %f %f\n',poseImg2(1), poseImg2(2), poseImg2(3), poseImg2(4), poseImg2(5), poseImg2(6));
fprintf(fid,'%f %f %f %f %f %f\n',poseImg2_end(1), poseImg2_end(2), poseImg2_end(3), poseImg2_end(4), poseImg2_end(5), poseImg2_end(6));
fclose(fid);

imwrite(uint8(img1*255.0), sprintf([savePath, '%s-img_1.png'], imgName));
imwrite(uint8(img2*255.0), sprintf([savePath, '%s-img_2.png'], imgName));

goodPair = 1;

end

