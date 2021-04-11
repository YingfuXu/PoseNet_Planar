function [dx] = fn_dx(pose, rate_body_vec, vel_heading_vec) 

roll = pose(1);
pitch = pose(2);
yaw = pose(3);

headingRw = R_bw(0.0, 0.0, yaw);

Rwb2dEuler = [1 tan(pitch)*sin(roll) tan(pitch)*cos(roll);
                        0 cos(roll)                 -sin(roll);
                        0 sin(roll)/cos(pitch) cos(roll)/cos(pitch)];
d_Euler = Rwb2dEuler * rate_body_vec;
         
d_P = headingRw' * vel_heading_vec;

dx = [d_Euler; d_P]; % 6*1 dimensions

end

