clc, clear, close all

%Solve for H on the form H*p_camera = p_robot   

% H is 4 x 4

% Each column is a 3D point in homogeneous coordinates
p_robot = [-0.68, 0.15, 0.47,  1;
           -0.4, 0.025, 0.49,  1;
           -0.57, 0.41, 0.37,  1;
           -0.69, -0.29, 0.03, 1]';  % 4x4

p_camera = [0.411, 0.359, 1.38, 1;
            0.23, 0.34, 1.235, 1;
            0.41, 0.363, 1.643, 1;
            0.294, 0.051, 0.93, 1]';    % 4x4

H = p_robot * inv(p_camera);
 
% test
% p_robot_actual = [-0.689, - 0.015, 0.488, 1]'
% p_camera_test = [0.359, 0.364, 1.29, 1]';
% 
% p_robot_test = H * p_camera_test


% Inference
p_camera_infer = [0.35, 0.295, 1.07, 1]';
p_robot = H * p_camera_infer