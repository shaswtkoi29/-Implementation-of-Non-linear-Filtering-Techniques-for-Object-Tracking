function [rmse, x_est, x_smooth, rmse_smooth] = ekf_implementation(F, H, J_F, J_H, initial_state, initial_covariance, Q, R, noisy_measurements, groundtruth, full_occlusion, out_of_view, use_opt, use_smooth)

% Number of frames
N = size(groundtruth, 1);

% Initialize state and covariance
x = initial_state;
P = initial_covariance;

% Storage for estimates
x_est = zeros(length(initial_state), N);
P_store = zeros(length(initial_state), length(initial_state), N);

% Check if optimization is to be used
if use_opt == 1
    [Q_opt, R_opt] = pso_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);
    Q = Q_opt;
    R = R_opt;
elseif use_opt == 2
    [Q_opt, R_opt] = ga_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);
    Q = Q_opt;
    R = R_opt;
elseif use_opt == 3
    [Q_opt, R_opt] = sa_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);
    Q = Q_opt;
    R = R_opt;
elseif use_opt == 4
    [Q_opt, R_opt] = pso_ga_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);
    Q = Q_opt;
    R = R_opt;
elseif use_opt == 5
    [Q_opt, R_opt] = pso_sa_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);
    Q = Q_opt;
    R = R_opt;
end

n = length(initial_state);
[m, ~] = size(noisy_measurements);

assert(size(initial_covariance, 1) == n && size(initial_covariance, 2) == n, 'initial_covariance must be n x n');
assert(size(Q, 1) == n && size(Q, 2) == n, 'Q must be n x n');
assert(size(R, 1) == m && size(R, 2) == m, 'R must be m x m');

% EKF Loop
for k = 1:N
    % Prediction
    x_pred = F(x);
    F_k = J_F(x);
    
    % Add debugging output
    disp(['Iteration: ', num2str(k)]);
    disp(['Size of F_k: ', num2str(size(F_k))]);
    disp(['Size of P: ', num2str(size(P))]);
    disp(['Size of Q: ', num2str(size(Q))]);
    
    assert(size(F_k, 1) == n && size(F_k, 2) == n, 'F_k must be n x n');
    
    P_pred = F_k * P * F_k' + Q;

    
    % Update
    if ~full_occlusion(k) && ~out_of_view(k)
        z = noisy_measurements(k, :)'; % Measurement
        H_k = J_H(x_pred);
        y = z - H(x_pred); % Innovation
        S = H_k * P_pred * H_k' + R; % Innovation covariance
        % Regularization to avoid singularity
        S = S + 1e-6 * eye(size(S));
        K = P_pred * H_k' / S; % Kalman gain
        x = x_pred + K * y; % State update
        P = (eye(length(x)) - K * H_k) * P_pred; % Covariance update
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

if use_smooth
    [x_smooth, rmse_smooth] = rts_smoothing(x_est, P_store, F, J_F, Q, groundtruth);
else
    x_smooth = x_est;
    rmse_smooth = rmse;
end

end

% PSO Optimization Function
function [Q_opt, R_opt] = pso_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view)
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

            % Run EKF with current Q and R
            x = initial_state;
            P = initial_covariance;
            x_est_pso = zeros(length(initial_state), N);

            for k = 1:N
                % Prediction
                x_pred = F(x);
                F_k = J_F(x);
                P_pred = F_k * P * F_k' + Q_particle;

                % Update
                if ~full_occlusion(k) && ~out_of_view(k)
                    z = noisy_measurements(k, :)';
                    H_k = J_H(x_pred);
                    y = z - H(x_pred);
                    S = H_k * P_pred * H_k' + R_particle;
                    S = S + 1e-6 * eye(size(S)); % Regularization
                    K = P_pred * H_k' / S;
                    x = x_pred + K * y;
                    P = (eye(length(x)) - K * H_k) * P_pred;
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

% RTS Smoothing Function
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

% Genetic Algorithm Optimization Function
function [Q_opt,R_opt] = ga_optimize(F,H,J_F,J_H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view)
    options = optimoptions('ga','MaxGenerations',50,'Display','off');
    fitnessFunc = @(params)ekf_fitness(F,H,J_F,J_H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view,...
        diag(params(1:6)),diag(params(7:8)));
    lb = 1e-6*ones(1,8);
    ub = 10*ones(1,8);
    [bestParams,~] = ga(fitnessFunc,length(lb),[],[],[],[],lb,ub,[],options);
    Q_opt = diag(bestParams(1:6));
    R_opt = diag(bestParams(7:8));
end

% Simulated Annealing Optimization Function
function [Q_opt,R_opt] = sa_optimize(F,H,J_F,J_H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view)
    optionsSA = optimoptions('simulannealbnd','MaxIterations',100,'Display','off');
    fitnessFuncSA = @(params)ekf_fitness(F,H,J_F,J_H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view,...
        diag(params(1:6)),diag(params(7:8)));
    lbSA = 1e-6*ones(1,8);
    ubSA = 10*ones(1,8);
    bestParamsSA = simulannealbnd(fitnessFuncSA,[0.1*ones(1,6),0.01*ones(1,2)],lbSA,ubSA,optionsSA);
    Q_opt = diag(bestParamsSA(1:6));
    R_opt = diag(bestParamsSA(7:8));
end

% Fitness Function for Optimization Algorithms
function fitness = ekf_fitness(F,H,J_F,J_H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view,Q,R)
    N = size(true_vals,1);
    x = x_init;
    P = P_init;
    x_est = zeros(length(x_init),N);
    for k = 1:N
        x_pred = F(x);
        F_k = J_F(x);
        P_pred = F_k*P*F_k'+Q;
        if ~is_occluded(k) && ~is_out_of_view(k)
            z = noisy_meas(k,:)';
            H_k = J_H(x_pred);
            y = z-H(x_pred);
            S = H_k*P_pred*H_k'+R;
            S = S+1e-6*eye(size(S));
            K = P_pred*H_k'/S;
            x = x_pred+K*y;
            P = (eye(length(x))-K*H_k)*P_pred;
        else
            x = x_pred;
            P = P_pred;
        end
        x_est(:,k) = x;
    end
    fitness = sqrt(mean(sum((true_vals(:,1:2)'-x_est(1:2,:)).^2)));
end

% PSO-GA Hybrid Optimization Function
function [Q_opt, R_opt] = pso_ga_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view)
    % Step 1: Use PSO to optimize
    [Q_pso, R_pso] = pso_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);
    
    % Step 2: Use GA to refine the PSO solution
    [Q_ga, R_ga] = ga_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);
    
    % Combine results (weighted average)
    Q_opt = (Q_pso + Q_ga) / 2;
    R_opt = (R_pso + R_ga) / 2;
end

% PSO-SA Hybrid Optimization Function
function [Q_opt, R_opt] = pso_sa_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view)
    % Step 1: Use PSO to optimize
    [Q_pso, R_pso] = pso_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);
    
    % Step 2: Use SA to refine the PSO solution
    [Q_sa, R_sa] = sa_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);
    
    % Combine results (weighted average)
    Q_opt = (Q_pso + Q_sa) / 2;
    R_opt = (R_pso + R_sa) / 2;
end