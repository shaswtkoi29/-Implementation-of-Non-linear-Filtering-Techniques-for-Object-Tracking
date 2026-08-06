function [rmse, x_est, x_smooth, rmse_smooth] = pf_implementation(...
    F, H, initial_state, initial_covariance, Q, R, ...
    noisy_measurements, groundtruth, full_occlusion, out_of_view, use_optimization, use_smooth)

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
    
    % Check if optimization is to be used
    if use_optimization == 1
  
        [Q, R] = pso_optimize(F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, n_particles);

    elseif use_optimization == 2
        
        [Q, R] = ga_optimize(F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, n_particles);
        
    elseif use_optimization == 3
    
        [Q, R] = sa_optimize(F, H, initial_state, initial_covariance, noisy_measurements, groundtruth, full_occlusion, out_of_view, n_particles);
      
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
        
    end


    % Calculate RMSE
    rmse = sqrt(mean(sum((groundtruth(:,1:2)' - x_est(1:2,:)).^2)));

  if use_smooth
            [x_smooth, rmse_smooth] = rts_smoothing(x_est, P_store, F, Q, groundtruth); 
  else
            x_smooth = x_est;
            rmse_smooth = rmse; 
  end
    % Calculate RMSE after smoothing (ignoring NaNs)
 valid_indices = ~isnan(x_smooth(1,:)) & ~isnan(x_smooth(2,:));
 if any(valid_indices)
     rmse_smooth = sqrt(mean(sum((groundtruth(valid_indices,1:2)' - x_smooth(1:2,valid_indices)).^2)));
 else
     warning('No valid smoothed states available for RMSE calculation');
     rmse_smooth = NaN;
 end
end    
% Calculate RMSE after smoothing (ignoring NaNs)
% valid_indices = ~isnan(x_smooth(1,:)) & ~isnan(x_smooth(2,:));
% if any(valid_indices)
%     rmse_smooth = sqrt(mean(sum((groundtruth(valid_indices,1:2)' - x_smooth(1:2,valid_indices)).^2)));
% else
%     warning('No valid smoothed states available for RMSE calculation');
%     rmse_smooth = NaN;
% end
% end

function [Q_opt, R_opt] = pso_optimize(F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view, n_particles)
    % PSO parameters
    num_particles = 50;
    num_iterations = 100;
    w = 0.729; % Inertia weight
    c1 = 2.05; % Cognitive parameter
    c2 = 2.05; % Social parameter
    
    % Initialize particles
    lb = 1e-6 * ones(1, 8); % Lower bound
    ub = 100 * ones(1, 8); % Upper bound
    particles = lb + (ub - lb) .* rand(num_particles, 8);
    velocities = zeros(num_particles, 8);
    
    pbest = particles;
    gbest = particles(1,:);
    pbest_fitness = inf(num_particles, 1);
    gbest_fitness = inf;
    
    N = size(true_vals, 1);
    eval_indices = select_evaluation_frames(N);
    
    % Initialize parallel pool if not already running
    if isempty(gcp('nocreate'))
        parpool('local');
    end
    
    no_improvement_count = 0;
    improvement_threshold = 1e-6;
    
    % Main PSO loop
    for iter = 1:num_iterations
        prev_gbest_fitness = gbest_fitness;
        
        % Evaluate fitness for all particles in parallel
        parfor i = 1:num_particles
            Q_particle = diag(particles(i, 1:6));
            R_particle = diag(particles(i, 7:8));
            
            % Ensure positive definiteness
            Q_particle = Q_particle + 1e-6 * eye(size(Q_particle));
            R_particle = R_particle + 1e-6 * eye(size(R_particle));
            
            % Run Particle Filter with current Q and R values
            [x_est_pso] = particle_filter_parallel(F, H, x_init, P_init, Q_particle, R_particle,...
                noisy_meas(eval_indices,:), true_vals(eval_indices,:), is_occluded(eval_indices), is_out_of_view(eval_indices), n_particles);
            
            % Calculate fitness (RMSE)
            fitness = calculate_rmse(true_vals(eval_indices,:), x_est_pso(1:2,:));
            
            % Update personal best
            if fitness < pbest_fitness(i)
                pbest(i,:) = particles(i,:);
                pbest_fitness(i) = fitness;
            end
        end
        
        % Update global best
        [min_fitness, min_idx] = min(pbest_fitness);
        if min_fitness < gbest_fitness
            gbest = pbest(min_idx,:);
            gbest_fitness = min_fitness;
            no_improvement_count = 0;
        else
            no_improvement_count = no_improvement_count + 1;
        end
        
        % Early stopping condition
        if no_improvement_count >= 10 || abs(gbest_fitness - prev_gbest_fitness) < improvement_threshold
            break;
        end
        
        % Update velocities and positions of particles
        r1 = rand(num_particles, 8);
        r2 = rand(num_particles, 8);
        velocities = w * velocities + ...
            c1 * r1 .* (pbest - particles) + ...
            c2 * r2 .* (repmat(gbest, num_particles, 1) - particles);
        
        % Apply velocity clamping
        vmax = 0.1 * (ub - lb);
        velocities = max(min(velocities, vmax), -vmax);
        
        particles = particles + velocities;
        
        % Apply boundary condition
        particles = max(min(particles, ub), lb);
        
        fprintf('PSO iteration %d/%d complete. Best fitness: %.4f\n', iter, num_iterations, gbest_fitness);
    end
    
    Q_opt = diag(gbest(1:6));
    R_opt = diag(gbest(7:8));
end

function eval_indices = select_evaluation_frames(N)
    % Select a subset of frames for evaluation
    num_eval_frames = min(100, N); % Use at most 100 frames for evaluation
    eval_indices = sort(randperm(N, num_eval_frames));
end

function rmse = calculate_rmse(true_vals, est_vals)
    % Ensure the arrays have the same size
    min_length = min(size(true_vals, 1), size(est_vals, 2));
    true_vals = true_vals(1:min_length, :);
    est_vals = est_vals(:, 1:min_length);
    
    % Calculate RMSE
    rmse = sqrt(mean(sum((true_vals' - est_vals).^2)));
    
end

function [x_est] = particle_filter_parallel(F, H, x_init, P_init, Q_particle, R_particle, noisy_meas_subset, true_vals_subset, is_occluded, is_out_of_view, n_particles)
    L = length(x_init);
    N_subset_frames = size(noisy_meas_subset, 1);
    
    particles = repmat(x_init, 1, n_particles) + mvnrnd(zeros(L,1), P_init, n_particles)';
    weights = ones(1, n_particles) / n_particles;
    
    x_est = zeros(L, N_subset_frames);
    
    for k = 1:N_subset_frames
        % Prediction Step
        particles = F(particles) + mvnrnd(zeros(L,1), Q_particle, n_particles)';
        
        % Update Step
        if ~is_occluded(k) && ~is_out_of_view(k)
            z = noisy_meas_subset(k, :)';
            likelihood = zeros(1, n_particles);
            
            for i = 1:n_particles
                likelihood(i) = mvnpdf(z, H(particles(:,i)), R_particle);
            end
            
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
    end
end


function [Q_opt, R_opt] = ga_optimize(F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view, n_particles)
    % GA parameters
    population_size = 50;
    num_generations = 100;
    mutation_rate = 0.01;
    
    % Initialize population
    population = rand(population_size, 8);
    
    for gen = 1:num_generations
        fitness_values = zeros(population_size, 1);
        
        for i = 1:population_size
            Q = diag(population(i, 1:6));
            R = diag(population(i, 7:8));
            
            [x_est_ga, ~] = particle_filter(F, H, x_init, P_init, Q, R, noisy_meas, is_occluded, is_out_of_view, n_particles, size(true_vals, 1));
            
            fitness_values(i) = sqrt(mean(sum((true_vals(:,1:2)' - x_est_ga(1:2,:)).^2)));
        end
        
        [~, sorted_indices] = sort(fitness_values);
        population = population(sorted_indices,:);
        
        % Select top half for reproduction
        parents = population(1:floor(population_size/2), :);
        
        % Crossover and mutation
        children = zeros(size(parents));
        for i = 1:2:size(parents, 1)
            crossover_point = randi([1, 7]);
            children(i, :) = [parents(i, 1:crossover_point), parents(i+1, crossover_point+1:end)];
            children(i+1, :) = [parents(i+1, 1:crossover_point), parents(i, crossover_point+1:end)];
            
            % Mutation
            for j = 1:8
                if rand < mutation_rate
                    children(i, j) = rand;
                    children(i+1, j) = rand;
                end
            end
        end
        
        % Replace bottom half of population with children
        population(floor(population_size/2)+1:end, :) = children;
        
        % Progress reporting
    end
    
    best_solution = population(1, :);
    Q_opt = diag(best_solution(1:6));
    R_opt = diag(best_solution(7:8));
end

function [x_smooth, rmse_smooth] = rts_smoothing(x_est, P_store, F, Q, groundtruth)
    N = size(x_est, 2);
    L = size(x_est, 1);
    x_smooth = zeros(size(x_est));
    P_smooth = zeros(size(P_store));

    % Initialize with last estimates
    x_smooth(:, N) = x_est(:, N);
    P_smooth(:, :, N) = P_store(:, :, N);

    for k = N-1:-1:1
        % Prediction step
        x_pred = F(x_est(:, k));
        P_pred = P_store(:, :, k) + Q;

        % Smoothing gain
        % Add a small value to the diagonal of P_pred to ensure it's invertible
        P_pred_inv = inv(P_pred + 1e-10 * eye(size(P_pred)));
        K = P_store(:, :, k) * P_pred_inv;

        % Update smoothed state and covariance
        x_smooth(:, k) = x_est(:, k) + K * (x_smooth(:, k+1) - x_pred);
        P_smooth(:, :, k) = P_store(:, :, k) + K * (P_smooth(:, :, k+1) - P_pred) * K';

        % Ensure symmetry for numerical stability
        P_smooth(:, :, k) = (P_smooth(:, :, k) + P_smooth(:, :, k)') / 2;

        % Check for NaN values and handle them
        if any(isnan(x_smooth(:,k)))      
            x_smooth(:,k) = x_est(:,k);
        end

        if any(isnan(P_smooth(:,:,k)))         
            P_smooth(:,:,k) = P_store(:,:,k);
        end
    end

    % Calculate RMSE after smoothing
    rmse_smooth = sqrt(mean(sum((groundtruth(:,1:2)' - x_smooth(1:2,:)).^2)));
end