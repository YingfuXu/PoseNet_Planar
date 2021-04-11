function [img_cn, cn_H_c0, flag] = fn_imgCn(IntrinsicMatrixC0, imgC0, cnEuler, P_cn)

distanceC0 = 1;
P_c0 = [0, 0, -distanceC0]';

cn_R_w = R_bw(cnEuler(1), cnEuler(2), cnEuler(3));

%% homography matrix 
normVec_N = [0, 0, 1]';
t_vec = cn_R_w * (P_c0 - P_cn);
cn_R_c0 = cn_R_w;
cn_H_c0 = cn_R_c0 + t_vec * normVec_N' / distanceC0; 
% transfer the points on the plane from cam0 frame to current cam frame

%% image reprojection

[img_cn, flag] = fn_warpImgHomo(imgC0, IntrinsicMatrixC0, cn_H_c0^(-1));

end

