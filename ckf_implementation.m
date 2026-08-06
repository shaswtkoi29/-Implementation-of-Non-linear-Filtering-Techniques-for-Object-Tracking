function [rmse, x_est, x_smooth, rmse_smooth] = ckf_implementation(...
    F, H, J_F, J_H, initial_state, initial_covariance, Q, R, ...
    noisy_measurements, groundtruth, full_occlusion, out_of_view, use_pso)

% Number of frames
N = size(groundtruth, 1);

% Initialize state and covariance
x = initial_state;
P = initial_covariance;

% Storage for estimates
x_est = zeros(length(initial_state), N);
P_store = zeros(length(initial_state), length(initial_state), N);

% Check if PSO optimization is to be used
if use_pso
    % Perform PSO to optimize Q and R
    [Q_opt, R_opt] = pso_optimize(...
        F, H, J_F, J_H, initial_state, initial_covariance, ...
        noisy_measurements, groundtruth, full_occlusion, out_of_view);
    Q = Q_opt;
    R = R_opt;
end

% CKF Loop
for k = 1:N
    % Prediction
    [x_pred, P_pred] = ckf_predict(x, P, Q, F);
    
    % Update
    if ~full_occlusion(k) && ~out_of_view(k)
        z = noisy_measurements(k, :)'; % Measurement
        [x, P] = ckf_update(x_pred, P_pred, z, R, H);
    else
        % If object is occluded or out of view
        x = x_pred;
        P = P_pred;
    end
    
    % Store estimates
    x_est(:, k) = x;
    P_store(:, :, k) = P;
end

% Calculate RMSE
rmse = sqrt(mean(sum((groundtruth(:,1:2)' - x_est(1:2,:)).^2, 1)));

% RTS Smoothing
[x_smooth, rmse_smooth] = rts_smoothing(x_est, P_store, F, J_F, Q, groundtruth);

end

function [x_pred, P_pred] = ckf_predict(x, P, Q, F)
    n = length(x);
    m = 2 * n;
    
    % Generate cubature points
    Xi = sqrt(n) * [eye(n), -eye(n)];
    X = repmat(x, 1, m) + chol(P)' * Xi;
    
    % Propagate cubature points
    Y = zeros(size(X));
    for i = 1:m
        Y(:,i) = F(X(:,i));
    end
    
    % Compute predicted state and covariance
    x_pred = mean(Y, 2);
    P_pred = (Y - x_pred) * (Y - x_pred)' / m + Q;
end

function [x_upd, P_upd] = ckf_update(x_pred, P_pred, z, R, H)
    n = length(x_pred);
    m = 2 * n;
    
    % Generate cubature points
    Xi = sqrt(n) * [eye(n), -eye(n)];
    X = repmat(x_pred, 1, m) + chol(P_pred)' * Xi;
    
    % Propagate cubature points through measurement function
    Z = zeros(length(z), m);
    for i = 1:m
        Z(:,i) = H(X(:,i));
    end
    
    % Compute predicted measurement and covariances
    z_pred = mean(Z, 2);
    Pzz = (Z - z_pred) * (Z - z_pred)' / m + R;
    Pxz = (X - x_pred) * (Z - z_pred)' / m;
    
    % Compute Kalman gain and update state
    K = Pxz / Pzz;
    x_upd = x_pred + K * (z - z_pred);
    P_upd = P_pred - K * Pzz * K';
end

function [Q_opt, R_opt] = pso_optimize(F, H, J_F, J_H, initial_state, initial_covariance, ...
    noisy_measurements, groundtruth, full_occlusion, out_of_view)

    % PSO Parameters
    num_particles = 50;
    num_iterations = 100;
    w = 0.729; % Inertia weight
    c1 = 1.49445; % Cognitive parameter
    c2 = 1.49445; % Social parameter
    
    % Initialize particles
    particles = rand(num_particles, 8); % 6 for Q diagonal, 2 for R diagonal
    velocities = zeros(num_particles, 8);
    pbest = particles;
    gbest = particles(1, :);
    pbest_fitness = inf(num_particles, 1);
    gbest_fitness = inf;
    
    N = size(groundtruth, 1);
    
    for iter = 1:num_iterations
        for i = 1:num_particles
            % Extract Q and R from particle
            Q_particle = diag(particles(i, 1:6));
            R_particle = diag(particles(i, 7:8));
            
            % Ensure positive definiteness
            Q_particle = Q_particle + 1e-6 * eye(size(Q_particle));
            R_particle = R_particle + 1e-6 * eye(size(R_particle));
            
            % Run CKF with current Q and R
            x = initial_state;
            P = initial_covariance;
            x_est_pso = zeros(length(initial_state), N);
            
            for k = 1:N
                % Prediction
                [x_pred, P_pred] = ckf_predict(x, P, Q_particle, F);
                
                % Update
                if ~full_occlusion(k) && ~out_of_view(k)
                    z = noisy_measurements(k, :)';
                    [x, P] = ckf_update(x_pred, P_pred, z, R_particle, H);
                else
                    x = x_pred;
                    P = P_pred;
                end
                x_est_pso(:, k) = x;
            end
            
            % Calculate fitness (RMSE)
            fitness = sqrt(mean(sum((groundtruth(:,1:2)' - x_est_pso(1:2,:)).^2, 1)));
            
            % Update personal best
            if fitness < pbest_fitness(i)
                pbest(i, :) = particles(i, :);
                pbest_fitness(i) = fitness;
            end
            
            % Update global best
            if fitness < gbest_fitness
                gbest = particles(i, :);
                gbest_fitness = fitness;
            end
        end
        
        % Update velocities and positions
        for i = 1:num_particles
            velocities(i, :) = w * velocities(i, :) + ...
                c1 * rand * (pbest(i, :) - particles(i, :)) + ...
                c2 * rand * (gbest - particles(i, :));
            particles(i, :) = particles(i, :) + velocities(i, :);
            
            % Ensure non-negative values for covariance matrices
            particles(i, :) = max(particles(i, :), 1e-6);
        end
    end
    
    % Extract optimal Q and R
    Q_opt = diag(gbest(1:6));
    R_opt = diag(gbest(7:8));
    
    % Ensure positive definiteness
    Q_opt = Q_opt + 1e-6 * eye(size(Q_opt));
    R_opt = R_opt + 1e-6 * eye(size(R_opt));

end

function [x_smooth, rmse_smooth] = rts_smoothing(x_est, P_store, F, J_F, Q, groundtruth)
    N = size(x_est, 2);
    x_smooth = zeros(size(x_est));
    P_smooth = zeros(size(P_store));
    
    % Initialize with last estimates
    x_smooth(:, N) = x_est(:, N);
    P_smooth(:, :, N) = P_store(:, :, N);
    
    for k = N-1:-1:1
        F_k = J_F(x_est(:, k));
        P_pred = F_k * P_store(:, :, k) * F_k' + Q;
        G = P_store(:, :, k) * F_k' / P_pred;
        x_smooth(:, k) = x_est(:, k) + G * (x_smooth(:, k+1) - F(x_est(:, k)));
        P_smooth(:, :, k) = P_store(:, :, k) + G * (P_smooth(:, :, k+1) - P_pred) * G';
    end
    
    % Calculate RMSE after smoothing
    rmse_smooth = sqrt(mean(sum((groundtruth(:,1:2)' - x_smooth(1:2,:)).^2, 1)));

end