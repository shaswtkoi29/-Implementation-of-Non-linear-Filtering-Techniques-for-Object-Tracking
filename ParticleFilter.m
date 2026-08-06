function [rmse, x_est, x_smooth, rmse_smooth] = ParticleFilter(...
    F, H, initial_state, initial_covariance, Q, R, ...
    noisy_measurements, groundtruth, full_occlusion, out_of_view, use_pso,smoothing)
    
    % Number of frames
    N = size(groundtruth, 1);
    L = length(initial_state);
    
    % Particle Filter parameters
    n_particles = 1000;
    
    % Initialize particles and weights
    particles = repmat(initial_state, 1, n_particles) + mvnrnd(zeros(L,1), initial_covariance, n_particles)';
    weights = ones(1, n_particles) / n_particles;
    
    % Storage for estimates
    x_est = zeros(L, N);
    P_store = zeros(L, L, N);
    
    % Check if PSO optimization is to be used
    if use_pso == 1
       
        [Q, R] = pso_optimize(F, H, initial_state, initial_covariance, ...
            noisy_measurements, groundtruth, full_occlusion, out_of_view, n_particles);
 
    end
    if use_pso == 3
         [Q, R] =sa_optimize(F, H, initial_state, initial_covariance, ...
            noisy_measurements, groundtruth, full_occlusion, out_of_view, n_particles);
    end

    
    % Particle Filter Loop
    
    for k = 1:N
        % Prediction Step
        particles = F(particles) + mvnrnd(zeros(L,1), Q, n_particles)';
        
        % Update Step
        if ~full_occlusion(k) && ~out_of_view(k)
            z = noisy_measurements(k, :)';
            
            % Calculate likelihood
            likelihood = zeros(1, n_particles);
            for i = 1:n_particles
                likelihood(i) = mvnpdf(z, H(particles(:,i)), R);
            end
            
            % Update weights
            weights = weights .* likelihood;
            weights = weights / sum(weights);
            
            % Resample if effective sample size is too low
            Neff = 1 / sum(weights.^2);
            if Neff < n_particles/2
                indices = randsample(n_particles, n_particles, true, weights);
                particles = particles(:, indices);
                weights = ones(1, n_particles) / n_particles;
            end
        end
        
        % Estimate state
        x_est(:, k) = particles * weights';
        P_store(:, :, k) = cov(particles');
        
        % Progress reporting

    end
    
    % Calculate RMSE
    rmse = sqrt(mean(sum((groundtruth(:,1:2)' - x_est(1:2,:)).^2)));
    
    if smoothing
        [x_smooth, rmse_smooth] = rts_smoothing(x_est, P_store, F, Q, groundtruth);
    else
        x_smooth = rmse;
        rmse_smooth = rmse;
    end

    
end

function [Q_opt, R_opt] = pso_optimize(F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view, n_particles)
    % PSO parameters
    num_particles = 50;
    num_iterations = 100;
    w = 0.729; % Inertia weight
    c1 = 2.05; % Cognitive parameter
    c2 = 2.05; % Social parameter

    % Initialize particles
    particles = rand(num_particles, 8); % 6 for Q diagonal, 2 for R diagonal
    velocities = zeros(num_particles, 8);
    pbest = particles;
    gbest = particles(1,:);
    pbest_fitness = inf(num_particles, 1);
    gbest_fitness = inf;
    
    N = size(true_vals, 1);
    L = length(x_init);

    for iter = 1:num_iterations
        for i = 1:num_particles
            Q = diag(particles(i, 1:6));
            R = diag(particles(i, 7:8));
            
            [x_est_pso, ~] = particle_filter(F, H, x_init, P_init, Q, R, noisy_meas, is_occluded, is_out_of_view, n_particles, N);
            
            % Calculate fitness (RMSE)
            fitness = sqrt(mean(sum((true_vals(:,1:2)' - x_est_pso(1:2,:)).^2)));

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
        for i = 1:num_particles
            velocities(i,:) = w * velocities(i,:) + ...
                c1 * rand(1,8) .* (pbest(i,:) - particles(i,:)) + ...
                c2 * rand(1,8) .* (gbest - particles(i,:));
            particles(i,:) = particles(i,:) + velocities(i,:);
            particles(i,:) = max(particles(i,:), 1e-6); % Ensure positive values
        end
        
        % Progress reporting
        if mod(iter, 10) == 0
            fprintf('PSO iteration %d of %d, best fitness: %f\n', iter, num_iterations, gbest_fitness);
        end
    end

    Q_opt = diag(gbest(1:6));
    R_opt = diag(gbest(7:8));
end

function [x_est, P_store] = particle_filter(F, H, x_init, P_init, Q, R, noisy_meas, is_occluded, is_out_of_view, n_particles, N)
    L = length(x_init);
    particles = repmat(x_init, 1, n_particles) + mvnrnd(zeros(L,1), P_init, n_particles)';
    weights = ones(1, n_particles) / n_particles;
    
    x_est = zeros(L, N);
    P_store = zeros(L, L, N);
    
    for k = 1:N
        % Prediction
        particles = F(particles) + mvnrnd(zeros(L,1), Q, n_particles)';
        
        % Update
        if ~is_occluded(k) && ~is_out_of_view(k)
            z = noisy_meas(k, :)';
            likelihood = zeros(1, n_particles);
            for i = 1:n_particles
                likelihood(i) = mvnpdf(z, H(particles(:,i)), R);
            end
            weights = weights .* likelihood;
            weights = weights / sum(weights);
            
            % Resample if needed
            Neff = 1 / sum(weights.^2);
            if Neff < n_particles/2
                indices = randsample(n_particles, n_particles, true, weights);
                particles = particles(:, indices);
                weights = ones(1, n_particles) / n_particles;
            end
        end
        
        x_est(:, k) = particles * weights';
        P_store(:, :, k) = cov(particles');
    end
end

function [x_smooth, rmse_smooth] = rts_smoothing(x_est, P_store, F, Q, groundtruth)
    N = size(x_est, 2);
    L = size(x_est, 1);
    
    x_smooth = zeros(size(x_est));
    P_smooth = zeros(size(P_store));
    
    x_smooth(:, N) = x_est(:, N);
    P_smooth(:, :, N) = P_store(:, :, N);
    
    for k = N-1:-1:1
        F_k = jacobian(F, x_est(:, k));
        P_pred = F_k * P_store(:, :, k) * F_k' + Q;
        
        G = P_store(:, :, k) * F_k' / (P_pred + 1e-8 * eye(size(P_pred))); % Added small regularization term
        x_smooth(:, k) = x_est(:, k) + G * (x_smooth(:, k+1) - F(x_est(:, k)));
        P_smooth(:, :, k) = P_store(:, :, k) + G * (P_smooth(:, :, k+1) - P_pred) * G';
        
        % Progress reporting
        if mod(k, 100) == 0
            fprintf('Smoothed frame %d of %d\n', N-k, N);
        end
    end
    
    rmse_smooth = sqrt(mean(sum((groundtruth(:,1:2)' - x_smooth(1:2,:)).^2)));
end

function J = jacobian(func, x)
    h = 1e-5;
    n = length(x);
    J = zeros(n, n);
    for i = 1:n
        x_plus = x;
        x_plus(i) = x_plus(i) + h;
        J(:, i) = (func(x_plus) - func(x)) / h;
    end
end


%%% sa

function [Q_opt, R_opt] = sa_optimize(F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view, n_particles)
    % SA parameters
    initial_temp = 100;
    final_temp = 1e-3;
    cooling_rate = 0.95;
    iterations_per_temp = 10; % Further reduced from 20
    max_iterations = 500; % Reduced from 1000

    % Initialize current solution
    current_solution = rand(1, 8);
    current_energy = evaluate_energy(current_solution, F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view, n_particles);
    
    best_solution = current_solution;
    best_energy = current_energy;
    
    temp = initial_temp;
    iteration = 0;
    
    % Precompute some values
    log_cooling_rate = log(cooling_rate);
    log_final_temp = log(final_temp);
    
    while log(temp) > log_final_temp && iteration < max_iterations
        for i = 1:iterations_per_temp
            % Generate new solution
            new_solution = current_solution + randn(1, 8) * 0.1;
            new_solution = max(new_solution, 1e-6);
            
            % Evaluate new solution
            new_energy = evaluate_energy(new_solution, F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view, n_particles);
            
            % Decide whether to accept the new solution
            if new_energy < current_energy || rand < exp((current_energy - new_energy) / temp)
                current_solution = new_solution;
                current_energy = new_energy;
                
                if new_energy < best_energy
                    best_solution = new_solution;
                    best_energy = new_energy;
                end
            end
        end
        
        % Cool down
        temp = exp(log(temp) + log_cooling_rate);
        iteration = iteration + 1;
        
        % Progress reporting (reduced frequency)
        if mod(iteration, 50) == 0
            fprintf('SA iteration %d, temperature: %f, best energy: %f\n', iteration, temp, best_energy);
        end
    end
    
    % Extract optimal Q and R
    Q_opt = diag(best_solution(1:6));
    R_opt = diag(best_solution(7:8));
end

function energy = evaluate_energy(solution, F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view, n_particles)
    Q = diag(solution(1:6));
    R = diag(solution(7:8));
    [x_est, ~] = particle_filter(F, H, x_init, P_init, Q, R, noisy_meas, is_occluded, is_out_of_view, n_particles, size(true_vals, 1));
    energy = sqrt(mean(sum((true_vals(:,1:2)' - x_est(1:2,:)).^2))); % RMSE as energy
end