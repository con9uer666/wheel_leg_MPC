% LQR_14state.m
% 14 状态 MPC 模型的 A/B 矩阵推导
%
% 在 10 状态版本 (LQR.m) 基础上扩展：
%   - 新增 2 个加速度变量：ddL_l, ddL_r
%   - 新增 2 个控制输入：F_leg_l, F_leg_r (沿腿杆方向的力)
%   - 新增 2 个状态：L_l, L_r 以及它们的导数 dL_l, dL_r
%
% 总状态: [s, ds, phi, dphi, theta_ll, dtheta_ll, theta_lr, dtheta_lr,
%          theta_b, dtheta_b, L_l, dL_l, L_r, dL_r]  → 14 维
% 总输入: [T_wl, T_wr, T_bl, T_br, F_leg_l, F_leg_r]  → 6 维
%
% 用法：MATLAB 里运行本脚本，会输出 14×14 A 和 14×6 B 的数值矩阵。

tic
clear all
clc

%%%%%%%%%%%%%%%%%%%%%%%%% Step 0：定义符号变量 %%%%%%%%%%%%%%%%%%%%%%%%%
syms R_w R_l l_l l_r l_wl l_wr l_bl l_br l_c
syms m_w m_l m_b I_w I_ll I_lr I_b I_z
syms theta_wl theta_wr dtheta_wl dtheta_wr
syms ddtheta_wl ddtheta_wr ddtheta_ll ddtheta_lr ddtheta_b
syms s ds phi dphi theta_ll dtheta_ll theta_lr dtheta_lr theta_b dtheta_b
syms T_wl T_wr T_bl T_br g

% ── 新增：腿长动力学相关变量 ──────────────────────────────────────────
syms L_l L_r dL_l dL_r ddL_l ddL_r
syms F_leg_l F_leg_r             % 沿腿杆方向力，伸长方向为正
syms damping_L                   % 腿伸缩阻尼系数（对应 MuJoCo XML 中 damping=5）

%%%%%%%%%%%%%%%%%%%%% Step 1：方程组（5 原方程 + 2 新方程） %%%%%%%%%%%%%

% 原 5 个方程（与 LQR.m 完全一致）
eqn1 = (I_w*l_l/R_w + m_w*R_w*l_l + m_l*R_w*l_bl)*ddtheta_wl ...
     + (m_l*l_wl*l_bl - I_ll)*ddtheta_ll ...
     + (m_l*l_wl + m_b*l_l/2)*g*theta_ll ...
     + T_bl - T_wl*(1 + l_l/R_w) == 0;

eqn2 = (I_w*l_r/R_w + m_w*R_w*l_r + m_l*R_w*l_br)*ddtheta_wr ...
     + (m_l*l_wr*l_br - I_lr)*ddtheta_lr ...
     + (m_l*l_wr + m_b*l_r/2)*g*theta_lr ...
     + T_br - T_wr*(1 + l_r/R_w) == 0;

eqn3 = -(m_w*R_w^2 + I_w + m_l*R_w^2 + m_b*R_w^2/2)*ddtheta_wl ...
     - (m_w*R_w^2 + I_w + m_l*R_w^2 + m_b*R_w^2/2)*ddtheta_wr ...
     - (m_l*R_w*l_wl + m_b*R_w*l_l/2)*ddtheta_ll ...
     - (m_l*R_w*l_wr + m_b*R_w*l_r/2)*ddtheta_lr ...
     + T_wl + T_wr == 0;

eqn4 = (m_w*R_w*l_c + I_w*l_c/R_w + m_l*R_w*l_c)*ddtheta_wl ...
     + (m_w*R_w*l_c + I_w*l_c/R_w + m_l*R_w*l_c)*ddtheta_wr ...
     + m_l*l_wl*l_c*ddtheta_ll + m_l*l_wr*l_c*ddtheta_lr ...
     - I_b*ddtheta_b + m_b*g*l_c*theta_b ...
     - (T_wl + T_wr)*l_c/R_w - (T_bl + T_br) == 0;

eqn5 = ((I_z*R_w)/(2*R_l) + I_w*R_l/R_w)*ddtheta_wl ...
     - ((I_z*R_w)/(2*R_l) + I_w*R_l/R_w)*ddtheta_wr ...
     + (I_z*l_l)/(2*R_l)*ddtheta_ll ...
     - (I_z*l_r)/(2*R_l)*ddtheta_lr ...
     - T_wl*R_l/R_w + T_wr*R_l/R_w == 0;

% ── 新增 2 个方程：沿腿杆方向的 F = m·a + damping·v + gravity·cos(theta) ──
% 在小角度线性化下，cos(theta_ll) ≈ 1 - theta_ll^2/2，对线性化模型来说在工作
% 点 theta_ll=0 处取一阶展开，cos 项贡献 0，gravity 项就是 m_l*g（常数偏置）。
% 偏置项不影响 Jacobian，会被 LQR/MPC 的 reference 修正吸收。
% 这里直接写 m_l*g 不写 cos(theta_ll)，保持线性。
eqn6 = m_l*ddL_l + damping_L*dL_l + m_l*g - F_leg_l == 0;
eqn7 = m_l*ddL_r + damping_L*dL_r + m_l*g - F_leg_r == 0;

% 同时解 7 个加速度
[ddtheta_wl, ddtheta_wr, ddtheta_ll, ddtheta_lr, ddtheta_b, ddL_l, ddL_r] = ...
    solve(eqn1, eqn2, eqn3, eqn4, eqn5, eqn6, eqn7, ...
          ddtheta_wl, ddtheta_wr, ddtheta_ll, ddtheta_lr, ddtheta_b, ddL_l, ddL_r);

%%%%%%%%%%%%%%%%%%%%% Step 2：Jacobian 计算 %%%%%%%%%%%%%%%%%%%%%%%%%%%

% 对状态向量取偏导：
%   角度类（5 个）：theta_ll, theta_lr, theta_b, L_l, L_r
%   速度类（2 个，只有 dL 进入加速度方程的 damping）：dL_l, dL_r
% 其他状态（s, ds, phi, dphi, dtheta_ll, dtheta_lr, dtheta_b）不出现在加速度
% 表达式中，所以 Jacobian 为 0，不需要算。

acc_vec = [ddtheta_wl, ddtheta_wr, ddtheta_ll, ddtheta_lr, ddtheta_b, ddL_l, ddL_r];

% 5 个"位置类"状态对应的列：theta_ll, theta_lr, theta_b, L_l, L_r
J_A_pos = jacobian(acc_vec, [theta_ll, theta_lr, theta_b, L_l, L_r]);

% 2 个"速度类"状态对应的列：dL_l, dL_r（damping 项贡献）
J_A_vel = jacobian(acc_vec, [dL_l, dL_r]);

% 6 个输入对应的列：T_wl, T_wr, T_bl, T_br, F_leg_l, F_leg_r
J_B = jacobian(acc_vec, [T_wl, T_wr, T_bl, T_br, F_leg_l, F_leg_r]);

%%%%%%%%%%%%%%%%%%%%% Step 3：填 A、B 矩阵（14×14 和 14×6） %%%%%%%%%%%%

% 14 状态向量顺序（1-indexed）：
%   1: s          2: ds
%   3: phi        4: dphi
%   5: theta_ll   6: dtheta_ll
%   7: theta_lr   8: dtheta_lr
%   9: theta_b    10: dtheta_b
%   11: L_l       12: dL_l
%   13: L_r       14: dL_r

A = sym('A', [14 14]);
B = sym('B', [14 6]);

% 全部初始化为 0
for r = 1:14
    A(r,:) = sym(zeros(1,14));
    B(r,:) = sym(zeros(1,6));
end

% 奇数行（位置类）的 d/dt = 对应的速度：A(odd, odd+1) = 1
%   A(1,2)=1   (ds=s_dot)
%   A(3,4)=1   (dphi=phi_dot)
%   A(5,6)=1   (dtheta_ll=theta_ll_dot)
%   A(7,8)=1   (dtheta_lr=theta_lr_dot)
%   A(9,10)=1  (dtheta_b=theta_b_dot)
%   A(11,12)=1 (dL_l=L_l_dot)
%   A(13,14)=1 (dL_r=L_r_dot)
for r = 1:2:13
    A(r, r+1) = 1;
end

% J_A_pos 的列对应位置类状态在 14 维向量中的索引：
%   col 1 (theta_ll) → state idx 5
%   col 2 (theta_lr) → state idx 7
%   col 3 (theta_b)  → state idx 9
%   col 4 (L_l)      → state idx 11
%   col 5 (L_r)      → state idx 13
pos_col_to_state = [5, 7, 9, 11, 13];

for col_j = 1:5
    state_col = pos_col_to_state(col_j);

    % A(2,state_col) = R_w/2 * (ddtheta_wl + ddtheta_wr) / d(state)
    A(2, state_col) = R_w/2 * (J_A_pos(1, col_j) + J_A_pos(2, col_j));

    % A(4,state_col): dphi 加速度（yaw 加速度 = R_w/(2R_l)(-J1+J2) - l_l/(2R_l) J3 + l_r/(2R_l) J4）
    A(4, state_col) = (R_w/(2*R_l)) * (-J_A_pos(1, col_j) + J_A_pos(2, col_j)) ...
                   - (l_l/(2*R_l)) * J_A_pos(3, col_j) ...
                   + (l_r/(2*R_l)) * J_A_pos(4, col_j);

    % 偶数行 6, 8, 10, 12, 14 → 对应加速度 ddtheta_ll, ddtheta_lr, ddtheta_b, ddL_l, ddL_r
    A(6,  state_col) = J_A_pos(3, col_j);   % ddtheta_ll
    A(8,  state_col) = J_A_pos(4, col_j);   % ddtheta_lr
    A(10, state_col) = J_A_pos(5, col_j);   % ddtheta_b
    A(12, state_col) = J_A_pos(6, col_j);   % ddL_l
    A(14, state_col) = J_A_pos(7, col_j);   % ddL_r
end

% J_A_vel 的列对应速度类状态在 14 维向量中的索引：
%   col 1 (dL_l) → state idx 12
%   col 2 (dL_r) → state idx 14
vel_col_to_state = [12, 14];

for col_j = 1:2
    state_col = vel_col_to_state(col_j);

    A(2, state_col) = R_w/2 * (J_A_vel(1, col_j) + J_A_vel(2, col_j));
    A(4, state_col) = (R_w/(2*R_l)) * (-J_A_vel(1, col_j) + J_A_vel(2, col_j)) ...
                   - (l_l/(2*R_l)) * J_A_vel(3, col_j) ...
                   + (l_r/(2*R_l)) * J_A_vel(4, col_j);
    A(6,  state_col) = J_A_vel(3, col_j);
    A(8,  state_col) = J_A_vel(4, col_j);
    A(10, state_col) = J_A_vel(5, col_j);
    A(12, state_col) = J_A_vel(6, col_j);
    A(14, state_col) = J_A_vel(7, col_j);
end

% B 矩阵：6 列输入
for h = 1:6
    B(2, h) = R_w/2 * (J_B(1, h) + J_B(2, h));
    B(4, h) = (R_w/(2*R_l)) * (-J_B(1, h) + J_B(2, h)) ...
            - (l_l/(2*R_l)) * J_B(3, h) ...
            + (l_r/(2*R_l)) * J_B(4, h);
    B(6,  h) = J_B(3, h);
    B(8,  h) = J_B(4, h);
    B(10, h) = J_B(5, h);
    B(12, h) = J_B(6, h);
    B(14, h) = J_B(7, h);
end

%%%%%%%%%%%%%%%%%%%%% Step 4：物理参数代入（与 LQR.m 一致） %%%%%%%%%%%%

g_ac = 9.81;
R_w_ac = 0.06;
R_l_ac = 0.19242;
l_c_ac = -0.01066729;
m_w_ac = 0.615; m_l_ac = 2.47507; m_b_ac = 12.634;
I_w_ac = 107156 * 10^(-8);
I_b_ac = 0.30949668;
I_z_ac = 0.62018662;
damping_L_ac = 5.0;          % 与 wheel_legged.xml 中 leg slide damping 一致

% 工作点：L = 0.15 m
l_l_ac = 0.15;
l_wl_ac = 0.07800145;
l_bl_ac = 0.07199846;
I_ll_ac = 0.05570304;
l_r_ac = 0.15;
l_wr_ac = 0.07800145;
l_br_ac = 0.07199846;
I_lr_ac = 0.05570304;

% 代入数值
subs_list_syms = [R_w R_l l_l l_r l_wl l_wr l_bl l_br l_c ...
                  m_w m_l m_b I_w I_ll I_lr I_b I_z g damping_L];
subs_list_vals = [R_w_ac R_l_ac l_l_ac l_r_ac l_wl_ac l_wr_ac l_bl_ac l_br_ac l_c_ac ...
                  m_w_ac m_l_ac m_b_ac I_w_ac I_ll_ac I_lr_ac I_b_ac I_z_ac g_ac damping_L_ac];

A_ac_num = double(subs(A, subs_list_syms, subs_list_vals));
B_ac_num = double(subs(B, subs_list_syms, subs_list_vals));

%%%%%%%%%%%%%%%%%%%%% Step 5：输出 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('=== A matrix (14x14) ==='); disp(A_ac_num);
disp('=== B matrix (14x6) ===');  disp(B_ac_num);

fprintf('\nA_ac_num =\n');
for i = 1:14
    fprintf('%.10g  ', A_ac_num(i,:));
    fprintf('\n');
end
fprintf('\nB_ac_num =\n');
for i = 1:14
    fprintf('%.10g  ', B_ac_num(i,:));
    fprintf('\n');
end

toc
