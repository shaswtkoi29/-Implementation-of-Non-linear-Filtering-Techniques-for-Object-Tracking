function [rmse, x_est, x_smooth, rmse_smooth] = ukf_implementation(F, H, initial_state, initial_covariance, Q, R, noisy_measurements, groundtruth, full_occlusion, out_of_view, use_pso, use_smooth)
    % UKF parameters
    alpha = 1e-3;
    beta = 2;
    kappa = 0;
    L = length(initial_state);
    lambda = alpha^2 * (L + kappa) - L;
    
    % Calculate weights
    Wm = [lambda / (L + lambda), repmat(1 / (2 * (L + lambda)), 1, 2*L)];
    Wc = Wm;
    Wc(1) = Wc(1) + (1 - alpha^2 + beta);
    
    % Run UKF once to get initial estimates
    [x_est, P_store] = run_ukf(F, H, initial_state, initial_covariance, Q, R, noisy_measurements, full_occlusion, out_of_view, Wm, Wc, lambda);
    
    % Optimization
    if use_pso > 0
        switch use_pso
            case 1
                [Q_opt, R_opt] = pso_optimize(F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda, x_est);
            case 2
                [Q_opt, R_opt] = ga_optimize(F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda, x_est);
            case 3
                [Q_opt, R_opt] = sa_optimize(F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda, x_est);
            otherwise
                error('Invalid optimization method');
        end
        Q = Q_opt;
        R = R_opt;
        
        % Rerun UKF with optimized parameters
        [x_est, P_store] = run_ukf(F, H, initial_state, initial_covariance, Q, R, noisy_measurements, full_occlusion, out_of_view, Wm, Wc, lambda);
    end
    
    % Calculate RMSE
    rmse = sqrt(mean(sum((groundtruth(:,1:2)' - x_est(1:2,:)).^2)));
    
    % RTS Smoothing
    if use_smooth
        [x_smooth, rmse_smooth] = rts_smoothing(x_est, P_store, F, Q, Wm, Wc, lambda);
    else
        x_smooth = x_est;
        rmse_smooth = rmse;
    end
end

function [x_est, P_store] = run_ukf(F, H, initial_state, initial_covariance, Q, R, noisy_measurements, full_occlusion, out_of_view, Wm, Wc, lambda)
    N = size(noisy_measurements, 1);
    L = length(initial_state);
    x_est = zeros(L, N);
    P_store = zeros(L, L, N);
    
    x = initial_state;
    P = initial_covariance;
    
    for k = 1:N
        sigma_points = generate_sigma_points(x, P, L, lambda);
        [x_pred, P_pred] = ukf_predict(sigma_points, Wm, Wc, Q, F);
        
        if ~full_occlusion(k) && ~out_of_view(k)
            z = noisy_measurements(k,:)';
            [x, P] = ukf_update(x_pred, P_pred, z, R, H, Wm, Wc, L, lambda);
        else
            x = x_pred;
            P = P_pred;
        end
        
        x_est(:, k) = x;
        P_store(:, :, k) = P;
    end
end


function sigma_points = generate_sigma_points(x, P, L, lambda)
    gamma = sqrt(L + lambda);
    try
        chol_P = chol(P)';
    catch
        % If chol fails, use SVD (this is slower)
        [U, S, ~] = svd(P);
        chol_P = U * sqrt(S);
    end
    sigma_points = [x, repmat(x, 1, L) + gamma * chol_P, repmat(x, 1, L) - gamma * chol_P];
end


function [x_pred, P_pred] = ukf_predict(sigma_points, Wm, Wc, Q, F)
    L = size(sigma_points, 1);
    n_sigma = size(sigma_points, 2);
    
    X_pred = zeros(size(sigma_points));
    for i = 1:n_sigma
        X_pred(:,i) = F(sigma_points(:,i));
    end
    
    x_pred = X_pred * Wm';
    
    P_pred = Q;
    for i = 1:n_sigma
        diff = X_pred(:,i) - x_pred;
        P_pred = P_pred + Wc(i) * (diff * diff');
    end
    P_pred = (P_pred + P_pred') / 2; % Ensure symmetry
end

function [x_upd, P_upd] = ukf_update(x_pred, P_pred, z, R, H, Wm, Wc, L, lambda)
    sigma_points = generate_sigma_points(x_pred, P_pred, L, lambda);
    
    n_sigma = size(sigma_points, 2);
    Z_pred = zeros(length(z), n_sigma);
    for i = 1:n_sigma
        Z_pred(:,i) = H(sigma_points(:,i));
    end
    
    z_pred = Z_pred * Wm';
    
    Pzz = R;
    Pxz = zeros(size(x_pred, 1), size(z, 1));
    for i = 1:n_sigma
        diff_z = Z_pred(:,i) - z_pred;
        diff_x = sigma_points(:,i) - x_pred;
        Pzz = Pzz + Wc(i) * (diff_z * diff_z');
        Pxz = Pxz + Wc(i) * (diff_x * diff_z');
    end
    Pzz = (Pzz + Pzz') / 2; % Ensure symmetry
    
    K = Pxz / Pzz;
    x_upd = x_pred + K * (z - z_pred);
    P_upd = P_pred - K * Pzz * K';
    P_upd = (P_upd + P_upd') / 2; % Ensure symmetry
end


function fitness = evaluate_fitness(Q, R, F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda)
    N = size(groundtruth, 1);
    L = length(initial_state);
    x_est = zeros(L, N);
    x = initial_state;
    P = initial_covariance;

    for k = 1:N
        sigma_points = generate_sigma_points(x, P, L, lambda);
        [x_pred, P_pred] = ukf_predict(sigma_points, Wm, Wc, Q, F);
        if ~full_occlusion(k) && ~out_of_view(k)
            z = noisy_measurements(k,:)';
            [x, P] = ukf_update(x_pred, P_pred, z, R, H, Wm, Wc, L, lambda);
        else
            x = x_pred;
            P = P_pred;
        end
        x_est(:,k) = x;
    end

    fitness = sqrt(mean(sum((groundtruth(:, 1:2)' - x_est(1:2,:)).^2)));
end

%%  Optimization Function

function [Q_opt, R_opt] = pso_optimize(F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda, x_est_initial)
    % PSO parameters
    num_particles = 30; % Reduced from 50
    num_iterations = 50; % Reduced from 100
    w = 0.729; % Inertia weight
    c1 = 2.05; % Cognitive parameter
    c2 = 2.05; % Social parameter

    % Initialize particles
    particles = rand(num_particles, 3); % qx, qy, sd
    velocities = zeros(num_particles, 3);
    pbest = particles;
    gbest = particles(1,:);
    pbest_fitness = inf(num_particles, 1);
    gbest_fitness = inf;

    N = size(groundtruth, 1);
    L = length(initial_state);

    for iter = 1:num_iterations
        parfor i = 1:num_particles
            qx = particles(i,1);
            qy = particles(i,2);
            sd = particles(i,3);
            Q = diag([qx*0.1^3/3, qy*0.1^3/3, qx*0.1, qy*0.1, 0.1, 0.1]);
            R = sd^2 * eye(2);
            
            fitness = evaluate_fitness_efficient(Q, R, F, H, x_est_initial, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda);
            
            if fitness < pbest_fitness(i)
                pbest(i,:) = particles(i,:);
                pbest_fitness(i) = fitness;
            end
            
            if fitness < gbest_fitness
                gbest = particles(i,:);
                gbest_fitness = fitness;
            end
        end

        % Update particles
        velocities = w * velocities + ...
            c1 * rand(num_particles, 3) .* (pbest - particles) + ...
            c2 * rand(num_particles, 3) .* (repmat(gbest, num_particles, 1) - particles);
        particles = particles + velocities;
        particles = max(particles, 1e-6); % Ensure positive values
    end

    qx_opt = gbest(1);
    qy_opt = gbest(2);
    sd_opt = gbest(3);
    Q_opt = diag([qx_opt*0.1^3/3, qy_opt*0.1^3/3, qx_opt*0.1, qy_opt*0.1, 0.1, 0.1]);
    R_opt = sd_opt^2 * eye(2);
end

function fitness = evaluate_fitness_efficient(Q, R, F, H, x_est_initial, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda)
    N = size(groundtruth, 1);
    L = size(x_est_initial, 1);
    x_est = x_est_initial;
    
    % Only update the last step
    x = x_est(:, end-1);
    P = eye(L); % Approximate covariance
    
    sigma_points = generate_sigma_points(x, P, L, lambda);
    [x_pred, P_pred] = ukf_predict(sigma_points, Wm, Wc, Q, F);
    
    if ~full_occlusion(end) && ~out_of_view(end)
        z = noisy_measurements(end, :)';
        [x, ~] = ukf_update(x_pred, P_pred, z, R, H, Wm, Wc, L, lambda);
    else
        x = x_pred;
    end
    
    x_est(:, end) = x;
    
    fitness = sqrt(mean(sum((groundtruth(:, 1:2)' - x_est(1:2,:)).^2)));
end

function [Q_opt, R_opt] = ga_optimize(F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda, x_est)
    % GA parameters
    population_size = 50;
    num_generations = 100;
    mutation_rate = 0.01;

    % Initialize population for qx, qy, and sd
    population = rand(population_size, 3);

    for gen = 1:num_generations
        % Evaluate fitness
        fitness_values = zeros(population_size, 1);
        for i = 1:population_size
            qx = population(i,1);
            qy = population(i,2);
            sd = population(i,3);
            Q_candidate = diag([qx*0.1^3/3, qy*0.1^3/3, qx*0.1, qy*0.1, 0.1, 0.1]);
            R_candidate = sd^2 * eye(2);
            fitness_values(i) = evaluate_fitness(Q_candidate, R_candidate, F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda);
        end

        % Selection
        [~, sorted_indices] = sort(fitness_values);
        selected = population(sorted_indices(1:floor(population_size/2)), :);

        % Crossover
        children = zeros(population_size - size(selected, 1), 3);
        for i = 1:size(children, 1)
            parents = selected(randi([1, size(selected, 1)], 1, 2), :);
            crossover_point = randi([1, 3]);
            children(i, :) = [parents(1, 1:crossover_point), parents(2, crossover_point+1:end)];
        end

        % Mutation
        mutation_mask = rand(size(children)) < mutation_rate;
        children(mutation_mask) = rand(sum(mutation_mask(:)), 1);

        % New population
        population = [selected; children];
    end

    best_solution = population(1,:);
    qx_opt = best_solution(1);
    qy_opt = best_solution(2);
    sd_opt = best_solution(3);
    Q_opt = diag([qx_opt*0.1^3/3, qy_opt*0.1^3/3, qx_opt*0.1, qy_opt*0.1, 0.1, 0.1]);
    R_opt = sd_opt^2 * eye(2);
end

function [Q_opt, R_opt] = sa_optimize(F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda, x_est)
    % SA parameters
    initial_temperature = 10.0;
    final_temperature = 0.001;
    alpha = 0.9;
    max_iterations_sa = 50;

    % Initialize solution for qx, qy, and sd
    current_solution = rand(1, 3);
    current_fitness = evaluate_fitness_sa(current_solution, F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda);
    
    best_solution = current_solution;
    best_fitness = current_fitness;
    
    temperature = initial_temperature;
    
    while temperature > final_temperature
        for iter = 1:max_iterations_sa
            new_solution = current_solution + randn(1, 3) * temperature;
            new_solution = max(new_solution, 1e-6);
            new_fitness = evaluate_fitness_sa(new_solution, F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda);
            
            if new_fitness < current_fitness || exp((current_fitness - new_fitness) / temperature) > rand()
                current_solution = new_solution;
                current_fitness = new_fitness;
                if new_fitness < best_fitness
                    best_solution = new_solution;
                    best_fitness = new_fitness;
                end
            end
        end
        temperature = temperature * alpha;
    end

    qx_best = best_solution(1);
    qy_best = best_solution(2);
    sd_best = best_solution(3);
    Q_opt = diag([qx_best*0.1^3/3, qy_best*0.1^3/3, qx_best*0.1, qy_best*0.1, 0.1, 0.1]);
    R_opt = sd_best^2 * eye(2);
end

function fitness = evaluate_fitness_sa(solution, F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda)
    qx = solution(1);
    qy = solution(2);
    sd = solution(3);
    Q = diag([qx*0.1^3/3, qy*0.1^3/3, qx*0.1, qy*0.1, 0.1, 0.1]);
    R = sd^2 * eye(2);
    fitness = evaluate_fitness(Q, R, F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, Wm, Wc, lambda);
end

%%%%% EXTRA CODE MIX OPTIMIZATION %%

function [Q_opt, R_opt] = pso_ga_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view)

    % Step 1: Use PSO to optimize
    [Q_pso, R_pso] = pso_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);

    % Step 2: Use GA to refine the PSO solution
    [Q_opt, R_opt] = ga_optimize(F, H, J_F, J_H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view);

    % Combine results (weighted average can be used if needed)
    Q_opt = (Q_pso + Q_opt) / 2;
    R_opt = (R_pso + R_opt) / 2;
end

function [Q_opt,R_opt] = pso_sa_optimize(F,H,J_F,J_H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view)

    % Step 1: Use PSO to optimize
    [Q_pso,R_pso] = pso_optimize(F,H,J_F,J_H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view);

    % Step 2: Use SA to refine the PSO solution
    [Q_opt,R_opt] = sa_optimize(F,H,J_F,J_H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view);

    % Combine results (weighted average can be used if needed)
    Q_opt = (Q_pso + Q_opt) / 2;
    R_opt = (R_pso + R_opt) / 2;
end


%% SMOOTHING
function [x_smooth, rmse_smooth] = rts_smoothing(x_est, P_store, F, Q, Wm, Wc, lambda)
    N = size(x_est, 2);
    L = size(x_est, 1);
    x_smooth = zeros(size(x_est));
    P_smooth = zeros(size(P_store));
    x_smooth(:, N) = x_est(:, N);
    P_smooth(:, :, N) = P_store(:, :, N);

    for k = N-1:-1:1
        sigma_points = generate_sigma_points(x_est(:, k), P_store(:, :, k), L, lambda);
        [x_pred, P_pred] = ukf_predict(sigma_points, Wm, Wc, Q, F);

        % Calculate cross-covariance
        Pxk_xk1 = zeros(L, L);
        for i = 1:2*L+1
            diff_k = sigma_points(:,i) - x_est(:,k);
            diff_k1 = F(sigma_points(:,i)) - x_pred;
            Pxk_xk1 = Pxk_xk1 + Wc(i) * (diff_k * diff_k1');
        end

        G = Pxk_xk1 / P_pred;
        x_smooth(:, k) = x_est(:, k) + G * (x_smooth(:, k+1) - x_pred);
        P_smooth(:, :, k) = P_store(:, :, k) + G * (P_smooth(:, :, k+1) - P_pred) * G';
    end

    rmse_smooth = sqrt(mean(sum((x_smooth(1:2, :) - x_est(1:2, :)).^2)));
end