clear; clc;

% 1. 传感器位置 (单位: cm)
P_cam_G = [0; 18; 4];
P_rad_G = [12; -22; -14];

% 2. 摄像头坐标系定义
cam_z_G = [1; 0; 0];  % 视线 -> 全局 X+
cam_x_G = [0; 1; 0];  % 右手 -> 全局 Y+
cam_y_G = [0; 0; -1]; % 根据右手定则 X*Y=Z，则相机Y为全局 Z-
R_CG = [cam_x_G, cam_y_G, cam_z_G];

% 3. 雷达坐标系定义
P_rad_target = [12.8794; -22; -14.2599];
v_rad = P_rad_target - P_rad_G;
rad_z_G = v_rad / norm(v_rad); % 雷达视线
rad_y_G = [0; 0; 1];           % 假设雷达自身上方向为全局 Z+
rad_x_G = cross(rad_y_G, rad_z_G);
R_RG = [rad_x_G, rad_y_G, rad_z_G];

% 4. 计算转换矩阵
R_R2C = R_CG' * R_RG;
T_R2C = R_CG' * (P_rad_G - P_cam_G);
H_R2C = [R_R2C, T_R2C; 0 0 0 1];

% 输出结果
fprintf('========= 更新后的转换关系 (单位: cm) =========\n');
fprintf('相机坐标系设定：X向右(Y+), Y向下(Z-), Z向前(X+)\n\n');
disp('旋转矩阵 R_R2C:'); disp(R_R2C);
disp('平移向量 T_R2C:'); disp(T_R2C);

% 5. 物理意义
test_pt_R = [0; 0; 100; 1]; % 雷达正前方1米的点
test_pt_C = H_R2C * test_pt_R;
fprintf('验证：雷达正前方100cm的点，在相机坐标系下的位置：\n');
fprintf('X(左右): %.2f cm, Y(上下): %.2f cm, Z(深度): %.2f cm\n', ...
    test_pt_C(1), test_pt_C(2), test_pt_C(3));