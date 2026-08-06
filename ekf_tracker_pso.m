function [positions, time] = ekf_tracker_pso(seq, res_path, bSaveImage, parameters)

    % Load Dataset
    seq.path = '/Users/riyakhadgi/Documents/MATLAB';
    seq.name = '/airplane/airplane-1/';
    data_path = [seq.path seq.name 'img/'];
    
    img_files = dir([data_path '*.jpg']);
    num_frames = length(img_files);
    
    groundtruth = dlmread([seq.path seq.name 'groundtruth.txt']);
    full_occlusion = dlmread([seq.path seq.name 'full_occlusion.txt']);
    out_of_view = dlmread([seq.path seq.name 'out_of_view.txt']);
    
    % Initialize state transition and measurement functions
    dt = 1 / 30;  % Assuming 30 fps
    
    F = @(x) [x(1) + x(3)*dt + 0.5*x(5)*dt^2;
              x(2) + x(4)*dt + 0.5*x(6)*dt^2;
              x(3) + x(5)*dt;
              x(4) + x(6)*dt;
              x(5);
              x(6)];
          
    H = @(x) [x(1); x(2)];
    
    % Jacobians for EKF
    J_F = @(x) [1 0 dt 0 0.5*dt^2 0;
                0 1 0 dt 0 0.5*dt^2;
                0 0 1 0 dt 0;
                0 0 0 1 0 dt;
                0 0 0 0 1 0;
                0 0 0 0 0 1];
            
    J_H = @(x) [1 0 0 0 0 0;
                0 1 0 0 0 0];
    
    % Initial state and covariance matrices
    init_state = [groundtruth(1,1); groundtruth(1,2); zeros(4,1)];
    init_covariance = eye(6);
    
    % Use PSO to tune Q and R
    [Q_opt, R_opt] = tune_using_pso(F, H, J_F, J_H, init_state, init_covariance, groundtruth);
    
    % Initialize result storage
    positions = zeros(num_frames,4);
    time = zeros(num_frames,1);
    
    % EKF variables
    x_estimates = zeros(6, num_frames);   % Store all estimates for RTS smoothing
    P_estimates = zeros(6,6,num_frames);   % Store all covariances for RTS smoothing
    
    x = init_state;   % Initial state vector
    P = init_covariance;   % Initial covariance matrix
    
    for frame_idx = 1:num_frames
        
        tic_loop = tic;   % Start timing
        
        img_file_path = fullfile(data_path, img_files(frame_idx).name);
        img = imread(img_file_path);   % Load current image
        
        if isempty(img)
            break;   % If no image is found, break out of the loop
        end
        
        %% Prediction Step (Time Update)
        x_pred = F(x);          % Predicted state estimate
        F_k = J_F(x);           % Jacobian of F at current state estimate
        P_pred = F_k * P * F_k' + Q_opt;   % Predicted estimate covariance
        
        %% Update Step (Measurement Update)
        
        if ~full_occlusion(frame_idx) && ~out_of_view(frame_idx)
            z = groundtruth(frame_idx,1:2)';   % Measurement from ground truth
            
            H_k = J_H(x_pred);                 % Jacobian of H at predicted state estimate
            y = z - H(x_pred);                 % Innovation (measurement residual)
            S = H_k * P_pred * H_k' + R_opt;   % Innovation covariance
            
            K = P_pred * H_k' / S;             % Kalman gain
            
            x = x_pred + K * y;                % Updated state estimate
            P = (eye(size(K,1)) - K * H_k) * P_pred;   % Updated estimate covariance
            
        else
            x = x_pred;   % If occluded or out of view, use predicted state as updated state
            P = P_pred;   % Use predicted covariance as updated covariance
        end
        
        %% Store Results for Current Frame
        
        positions(frame_idx,:) = [x(1)-x(3)/2, x(2)-x(4)/2, x(3), x(4)];
        time(frame_idx) = toc(tic_loop);   % Store time taken for this frame
        
        %% Store estimates for RTS smoothing later
        x_estimates(:, frame_idx) = x;
        P_estimates(:,:,frame_idx) = P;
        
        %% Optional: Visualization of Tracking Result (if enabled)
        
        if bSaveImage
            visualize_tracking_result(img, frame_idx, positions(frame_idx,:), groundtruth(frame_idx,:), seq.name, res_path);
        end
        
    end
    
    %% Apply RTS Smoothing after EKF forward pass
    [x_smooth] = rts_smoother(x_estimates, P_estimates, F_k);
    
    %% Create the EKF_tracker_tracking_result structure to store all results
    
    EKF_tracker_tracking_result.positions_estimated = positions;     % EKF-only results (positions)
    EKF_tracker_tracking_result.time_per_frame      = time;          % Time taken per frame
    EKF_tracker_tracking_result.x_smooth            = x_smooth';     % Smoothed results from RTS smoother (transpose for consistency)
    
    %% Save Results into the structure
    
    result_dir_path = fullfile(res_path, seq.name);
    
    if ~exist(result_dir_path, 'dir')
        mkdir(result_dir_path);
    end
    
    save(fullfile(result_dir_path, 'EKF_tracker_tracking_result.mat'), 'EKF_tracker_tracking_result');
    
end

%% Particle Swarm Optimization (PSO)

function [Q_opt, R_opt] = tune_using_pso(F, H, J_F, J_H, init_state, init_covariance, groundtruth)

    % Define bounds for Q and R tuning (these are arbitrary ranges)
    Q_bounds_low = diag([1e-3*ones(2,1); ones(4,1)]);   % Lower bounds for Q
    Q_bounds_high = diag([10*ones(6,1)]);               % Upper bounds for Q

    R_bounds_low = diag([10*ones(2)]);                  % Lower bounds for R
    R_bounds_high = diag([100*ones(2)]);                % Upper bounds for R

    % Define fitness function for PSO optimization (minimizing RMSE)
    fitness_function = @(params) pso_fitness(params,F,H,J_F,J_H,...
                                             init_state,...
                                             init_covariance,...
                                             groundtruth);

    % Run PSO optimization 
    options = optimoptions('particleswarm','Display','iter',...
                           'SwarmSize',50,'MaxIterations',100);

    % Fix: Concatenate vertically instead of horizontally
    best_params = pso(fitness_function, [diag(Q_bounds_low); diag(R_bounds_low)], ...
                      [diag(Q_bounds_high); diag(R_bounds_high)], options);

    % Extract optimized Q and R from best_params 
    Q_opt = diag(best_params(1:6));   % First 6 elements are for Q
    R_opt = diag(best_params(7:8));   % Last 2 elements are for R

end

%% Fitness Function for PSO

function rmse = pso_fitness(params,F,H,J_F,J_H,...
                            init_state,...
                            init_covariance,...
                            groundtruth)

    % Extract Q and R from params
    Q = diag(params(1:6));
    R = diag(params(7:8));

    % Run EKF with given Q and R 
    [x_est] = run_ekf(F,H,J_F,J_H,Q,R,...
                      init_state,...
                      init_covariance,...
                      groundtruth);

    % Compute RMSE between estimated states and ground truth 
    rmse = sqrt(mean(sum((groundtruth(:,1:2) - x_est(:,[1:2])).^2)));

end


function visualize_tracking_result(img, frame_id, tracking_result, ground_truth, sequence_name, res_path)
    % Helper function to visualize tracking results
    
    % Construct the full path for saving the image
    save_dir = fullfile(res_path, 'res_fig', sequence_name);
    
    % Check if the directory exists, if not, create it
    if ~exist(save_dir, 'dir')
        mkdir(save_dir);
    end
    
    % Create the full path for the image filename
    frame_img_path = fullfile(save_dir, [num2str(frame_id) '.jpg']);
    
    % Plot and save the result
    figure(1); clf;
    imshow(img);
    hold on;
    
    % Ensure tracking_result has valid width and height
    if tracking_result(3) > 0 && tracking_result(4) > 0
        % Draw tracker result in red
        rectangle('Position', tracking_result, 'EdgeColor', 'r', 'LineWidth', 2);
    else
        warning('Invalid tracking result dimensions at frame %d: [%f, %f]', frame_id, tracking_result(3), tracking_result(4));
    end
    
    % Ensure ground truth has valid width and height
    if ground_truth(3) > 0 && ground_truth(4) > 0
        % Draw ground truth in green
        rectangle('Position', ground_truth, 'EdgeColor', 'g', 'LineWidth', 2);
    else
        warning('Invalid ground truth dimensions at frame %d: [%f, %f]', frame_id, ground_truth(3), ground_truth(4));
    end
    
    % Display frame number
    text(10, 20, ['#' num2str(frame_id)], 'Color', 'y', 'FontWeight', 'bold', 'FontSize', 24);
    
    hold off;
    drawnow;
    
    % Save the frame as an image
    frame_img_data = getframe(gcf);
    imwrite(frame_img_data.cdata, frame_img_path);
end

