% main.m

% Clear workspace and command window
clear; clc; close all;

%% Load Dataset
% Path to the dataset
data_path = 'airplane/airplane-1/';

% Load ground truth data
groundtruth = dlmread([data_path 'groundtruth.txt']);

% Load full occlusion and out-of-view flags
full_occlusion = dlmread([data_path 'full_occlusion.txt']);
out_of_view = dlmread([data_path 'out_of_view.txt']);

% Number of frames
N = size(groundtruth, 1);

% Time step
dt = 0.5;

%% Define State Transition and Measurement Functions
% State Transition Function (Constant Acceleration Model)
F = @(x) [x(1) + x(3)*dt + 0.5*x(5)*dt^2;
          x(2) + x(4)*dt + 0.5*x(6)*dt^2;
          x(3) + x(5)*dt;
          x(4) + x(6)*dt;
          x(5);
          x(6)];

% Measurement Function
H = @(x) [x(1); x(2)];

% Jacobians (For EKF, but not used for UKF)
J_F = @(x) [1 0 dt 0 0.5*dt^2 0;
            0 1 0 dt 0 0.5*dt^2;
            0 0 1 0 dt 0;
            0 0 0 1 0 dt;
            0 0 0 0 1 0;
            0 0 0 0 0 1];

J_H = @(x) [1 0 0 0 0 0; 
            0 1 0 0 0 0];

% Initialize State and Covariance
initial_state = [groundtruth(1,1); groundtruth(1,2); 0; 0; 0; 0]; % [x, y, vx, vy, ax, ay]
initial_covariance = eye(6);

%% Generate Noisy Measurements
noise_std = 60; % Standard deviation of measurement noise
noisy_measurements = groundtruth(:,1:2) + noise_std * randn(N, 2);

%% Set Process and Measurement Noise Covariances
Q = 0.1 * eye(6); % Initial process noise covariance
R = 10 * eye(2);  % Initial measurement noise covariance


 % EKF Implementation Calls
  % Without PSO
  [rmse_ekf_no_pso, x_est_ekf_no_pso, x_smooth_ekf_no_pso, rmse_smooth_ekf_no_pso] = ekf_implementation(...
      F, H, J_F, J_H, initial_state, initial_covariance, Q, R, ...
%      noisy_measurements, groundtruth, full_occlusion, out_of_view, false);
 % 
 
  % With PSO
  [rmse_ekf_with_pso, x_est_ekf_with_pso, x_smooth_ekf_with_pso, rmse_smooth_ekf_with_pso] = ekf_implementation(...
      F, H, J_F, J_H, initial_state, initial_covariance, Q, R, ...
      noisy_measurements, groundtruth, full_occlusion, out_of_view, true);
  %% Display RMSE EKF Results
 
  fprintf('EKF - RMSE without PSO: %f\n', rmse_ekf_no_pso);
  fprintf('EKF - RMSE with PSO: %f\n', rmse_ekf_with_pso);
  fprintf('EKF - RMSE with RTS Smoothing (without PSO): %f\n', rmse_smooth_ekf_no_pso);
  fprintf('EKF - RMSE with PSO + RTS Smoothing: %f\n', rmse_smooth_ekf_with_pso);
% 
% 
 %% UKF 
% 
 % Without PSO
 [rmse_ukf_no_pso, x_est_ukf_no_pso, x_smooth_ukf_no_pso, rmse_smooth_ukf_no_pso] = ukf_implementation(...
     F, H, initial_state, initial_covariance, Q, R, ...
     noisy_measurements, groundtruth, full_occlusion, out_of_view, false);
 
 % With PSO
 [rmse_ukf_with_pso, x_est_ukf_with_pso, x_smooth_ukf_with_pso, rmse_smooth_ukf_with_pso] = ukf_implementation(...
     F, H, initial_state, initial_covariance, Q, R, ...
     noisy_measurements, groundtruth, full_occlusion, out_of_view, true);
 
 %% Display RMSE UKF Results
 fprintf('UKF - RMSE without PSO: %f\n', rmse_ukf_no_pso);
 fprintf('UKF - RMSE with PSO: %f\n', rmse_ukf_with_pso);
 fprintf('UKF - RMSE with RTS Smoothing (without PSO): %f\n', rmse_smooth_ukf_no_pso);
 fprintf('UKF - RMSE with PSO + RTS Smoothing: %f\n', rmse_smooth_ukf_with_pso);


 % EKF Implementation Calls
  % Without PSO
  [rmse_ekf_no_pso, x_est_ekf_no_pso, x_smooth_ekf_no_pso, rmse_smooth_ekf_no_pso] = ekf_implementation(...
      F, H, J_F, J_H, initial_state, initial_covariance, Q, R, ...
      noisy_measurements, groundtruth, full_occlusion, out_of_view, false);
 % 
 
  % With PSO
  [rmse_ekf_with_pso, x_est_ekf_with_pso, x_smooth_ekf_with_pso, rmse_smooth_ekf_with_pso] = ekf_implementation(...
      F, H, J_F, J_H, initial_state, initial_covariance, Q, R, ...
      noisy_measurements, groundtruth, full_occlusion, out_of_view, true);
  %% Display RMSE EKF Results
 
  fprintf('EKF - RMSE without PSO: %f\n', rmse_ekf_no_pso);
  fprintf('EKF - RMSE with PSO: %f\n', rmse_ekf_with_pso);
  fprintf('EKF - RMSE with RTS Smoothing (without PSO): %f\n', rmse_smooth_ekf_no_pso);
  fprintf('EKF - RMSE with PSO + RTS Smoothing: %f\n', rmse_smooth_ekf_with_pso);
% 


% CKF Implementation Calls
% Without PSO
[rmse_ckf_no_pso, x_est_ckf_no_pso, x_smooth_ckf_no_pso, rmse_smooth_ckf_no_pso] = ckf_implementation1(...
    F, H, J_F, J_H, initial_state, initial_covariance, Q, R, ...
    noisy_measurements, groundtruth, full_occlusion, out_of_view, false);

% With PSO
[rmse_ckf_with_pso, x_est_ckf_with_pso, x_smooth_ckf_with_pso, rmse_smooth_ckf_with_pso] = ckf_implementation1(...
    F, H, J_F, J_H, initial_state, initial_covariance, Q, R, ...
    noisy_measurements, groundtruth, full_occlusion, out_of_view, true);

% Display RMSE CKF Results
fprintf('CKF - RMSE without PSO: %f\n', rmse_ckf_no_pso);
fprintf('CKF - RMSE with PSO: %f\n', rmse_ckf_with_pso);
fprintf('CKF - RMSE with RTS Smoothing (without PSO): %f\n', rmse_smooth_ckf_no_pso);
fprintf('CKF - RMSE with PSO + RTS Smoothing: %f\n', rmse_smooth_ckf_with_pso);






%%% print 
% Collect all RMSE values into a vector
rmse_values = [
    rmse_ekf_no_pso, rmse_ekf_with_pso, rmse_smooth_ekf_no_pso, rmse_smooth_ekf_with_pso, ...
    rmse_ukf_no_pso, rmse_ukf_with_pso, rmse_smooth_ukf_no_pso, rmse_smooth_ukf_with_pso, ...
    rmse_ckf_no_pso, rmse_ckf_with_pso, rmse_smooth_ckf_no_pso, rmse_smooth_ckf_with_pso
];

% Corresponding names for each RMSE value
rmse_names = {
    'EKF without PSO', 'EKF with PSO', 'Smoothed EKF without PSO', 'Smoothed EKF with PSO', ...
    'UKF without PSO', 'UKF with PSO', 'Smoothed UKF without PSO', 'Smoothed UKF with PSO', ...
    'CKF without PSO', 'CKF with PSO', 'Smoothed CKF without PSO', 'Smoothed CKF with PSO'
};

% Find the minimum RMSE value and its index
[min_rmse, min_index] = min(rmse_values);

% Print the lowest RMSE value and its corresponding implementation
fprintf('Lowest RMSE: %f from %s\n', min_rmse, rmse_names{min_index});


% %% Plot Results for EKF and UKF
% figure;
% plot(groundtruth(:,1), groundtruth(:,2), 'g', 'LineWidth', 2); hold on;
% plot(noisy_measurements(:,1), noisy_measurements(:,2), 'rx');
% 
% % EKF Plots
% plot(x_est_ekf_no_pso(1,:), x_est_ekf_no_pso(2,:), 'b--', 'LineWidth', 2);
% plot(x_est_ekf_with_pso(1,:), x_est_ekf_with_pso(2,:), 'm-.', 'LineWidth', 2);
% plot(x_smooth_ekf_no_pso(1,:), x_smooth_ekf_no_pso(2,:), 'c:', 'LineWidth', 2);
% plot(x_smooth_ekf_with_pso(1,:), x_smooth_ekf_with_pso(2,:), 'k-', 'LineWidth', 2);
% 
% % UKF Plots
% plot(x_est_ukf_no_pso(1,:), x_est_ukf_no_pso(2,:), 'r--', 'LineWidth', 2);
% plot(x_est_ukf_with_pso(1,:), x_est_ukf_with_pso(2,:), 'y-.', 'LineWidth', 2);
% plot(x_smooth_ukf_no_pso(1,:), x_smooth_ukf_no_pso(2,:), 'b:', 'LineWidth', 2);
% plot(x_smooth_ukf_with_pso(1,:), x_smooth_ukf_with_pso(2,:), 'm-', 'LineWidth', 2);
% 
% legend('Ground Truth', 'Noisy Measurements', ...
%        'EKF without PSO', 'EKF with PSO', 'Smoothed EKF without PSO', 'Smoothed EKF with PSO', ...
%        'UKF without PSO', 'UKF with PSO', 'Smoothed UKF without PSO', 'Smoothed UKF with PSO');
% title('Object Tracking using EKF and UKF');
% xlabel('X Position');
% ylabel('Y Position');
% grid on;
% hold off;
% 
% %% Save data for future experiments with other filters (CKF, EnKF, etc.)
% save('tracking_data.mat', 'F', 'H', 'initial_state', ...
%      'initial_covariance', 'Q', 'R', 'noisy_measurements', 'groundtruth', ...
%      'full_occlusion', 'out_of_view', 'dt', 'N');
