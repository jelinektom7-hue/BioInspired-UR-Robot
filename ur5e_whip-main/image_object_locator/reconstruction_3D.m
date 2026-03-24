clc; clear; close all;
format compact;

syms s u v fx cx fy cy ...
    r11 r12 r13 r21 r22 r23 ...
    r31 r32 r33 t1 t2 t3 ...
    X Y Z

% Define the 3D pixel vector m
m = [u; v; 1]  % Transpose for column vector

% Camera intrinsic matrix
A = [fx, 0, cx; 
     0, fy, cy; 
     0, 0, 1]

% Rotation matrix
R = [r11, r12, r13;
     r21, r22, r23;
     r31, r32, r33]

% Translation vector
t = [t1; t2; t3]

% Combined rotation and translation matrix [R | t]
T = [R, t]

% Homogeneous coordinates for the 3D point
M = [X; Y; Z; 1]

% Equation: s * m' = A * [R | t] * M'
eq = s * m == A * T * M

% Now subs in known values
% Use the identity matrix for R and the zero vector for t
% eq = subs(eq, ...
%     [r11, r12, r13, r21, r22, r23, r31, r32, r33, t1, t2, t3], ...
%     [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0]);

seq = simplify(solve(eq(3), s))

eq = subs(eq, s, seq)

Xeq = expand(solve(eq(1), X));
disp('X = ')
pretty(Xeq)

Yeq = expand(solve(eq(2), Y));
disp('Y = ')
pretty(Yeq)