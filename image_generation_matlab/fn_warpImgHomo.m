function [img, flag] = fn_warpImgHomo(imgC0, IntrinsicMatrixC0, c0_H_cn)

global xy1; % [3*numPixel]
global widthPixelSizeCamN heightPixelSizeCamN;

if isa(imgC0, 'uint8')
    V = single(imgC0)/255.0; % mother image from which to sample
else
    V = imgC0;
end

[X,Y] = meshgrid(0:size(imgC0,2)-1, 0:size(imgC0,1)-1);

sample_xyz = c0_H_cn * xy1;

sample_xy1 = sample_xyz ./ sample_xyz(3, :);

sample_uv1 = IntrinsicMatrixC0 * sample_xy1;

sample_u = sample_uv1(1, :);
sample_v = sample_uv1(2, :);

u_sq = reshape(sample_u, [widthPixelSizeCamN, heightPixelSizeCamN])';
v_sq = reshape(sample_v, [widthPixelSizeCamN, heightPixelSizeCamN])';

Xq = u_sq;
Yq = v_sq;

Vq = interp2(X,Y,V,Xq,Yq);
img = Vq;

% imshow(img)

flag = 1;
% check four corner pixels of the image, if any one is black, it means that
% the image is too small to sample from, need to reduce the height if the
% simulated camera
if isnan(img(1, widthPixelSizeCamN, :)) || ... 
    isnan(img(heightPixelSizeCamN, 1, :)) || ...
	isnan(img(heightPixelSizeCamN, widthPixelSizeCamN, :)) || ...
	isnan(img(1, 1, :))
	% msg_char = "black corner ... Redo this Image!"
	flag = 0;
end

end