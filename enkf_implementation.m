function [rmse, x_est, x_smooth, rmse_smooth] = enkf_implementation(...
    F, H, initial_state, initial_covariance, Q, R, ...
    noisy_measurements, groundtruth, full_occlusion, out_of_view, use_pso)

    % Number of frames
    N = size(groundtruth, 1);
    L = length(initial_state);

    % EnKF parameters
    n_ensemble = 20; % Number of ensemble members

    % Initialize state and ensemble
    x = repmat(initial_state, 1, n_ensemble) + mvnrnd(zeros(L,1), initial_covariance, n_ensemble)';
    P = initial_covariance;

    % Storage for estimates
    x_est = zeros(L, N);
    P_store = zeros(L, L, N);

    % Check if optimization is to be used and set Q, R
    if use_pso == 1
        [Q, R] = pso_optimize(F, H, initial_state, initial_covariance, ...
            noisy_measurements, groundtruth, full_occlusion, out_of_view, n_ensemble);
    end
    if use_pso == 2
        fprintf('Starting GA optimization...\n');
        [Q, R] = ga_optimize(F,H,initial_state, initial_covariance, ...
            noisy_measurements, groundtruth, full_occlusion, out_of_view);
    end

    if use_pso == 3
        [Q, R] = sa_optimize(F,H,initial_state, initial_covariance, ...
            noisy_measurements, groundtruth, full_occlusion, out_of_view);
    end


    % EnKF Loop
    for k = 1:N
        % Prediction Step
        for i = 1:n_ensemble
            x(:,i) = F(x(:,i)) + mvnrnd(zeros(L,1), Q)';
        end

        % Calculate predicted mean and covariance
        x_pred = mean(x, 2);
        P_pred = (x - x_pred) * (x - x_pred)' / (n_ensemble - 1);

        % Update Step
        if ~full_occlusion(k) && ~out_of_view(k)
            z = noisy_measurements(k, :)';

            % Generate perturbed observations
            Z = repmat(z, 1, n_ensemble) + mvnrnd(zeros(size(z,1),1), R, n_ensemble)';

            % Compute measurement predictions
            H_x = zeros(size(z,1), n_ensemble);
            for i = 1:n_ensemble
                H_x(:,i) = H(x(:,i));
            end

            % Calculate innovation covariance and cross-covariance
            y_mean = mean(H_x, 2);
            Y = H_x - y_mean;
            P_xy = (x - x_pred) * Y' / (n_ensemble - 1);
            P_yy = Y * Y' / (n_ensemble - 1) + R;

            % Calculate Kalman gain
            K = P_xy / (P_yy + 1e-8 * eye(size(P_yy))); % Added small regularization term

            % Update ensemble members
            for i = 1:n_ensemble
                x(:,i) = x(:,i) + K * (Z(:,i) - H(x(:,i)));
            end
        end

        % Store estimates
        x_est(:, k) = mean(x, 2);
        P_store(:, :, k) = (x - x_est(:,k)) * (x - x_est(:,k))' / (n_ensemble - 1);

    end

    % Calculate RMSE
    rmse = sqrt(mean(sum((groundtruth(:,1:2)' - x_est(1:2,:)).^2)));

    % RTS Smoothing
    [x_smooth, rmse_smooth] = rts_smoothing(x_est, P_store, F, Q, groundtruth);
end

function [Q_opt, R_opt] = pso_optimize(F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view, n_ensemble)

    num_particles = 20;
    num_iterations = 100; % Set to 100 as required
    w = 0.729;
    c1 = 2.05;
    c2 = 2.05;

    particles = rand(num_particles, 8); % 6 for Q diagonal, 2 for R diagonal
    velocities = zeros(num_particles, 8);
    pbest = particles;
    gbest = particles(1,:);
    pbest_fitness = inf(num_particles, 1);
    gbest_fitness = inf;

    N = size(true_vals, 1);
    L = length(x_init);

    % Precompute chol(P_init) for use in mvnrnd
    chol_P_init = chol(P_init)';

    for iter = 1:num_iterations
        parfor i = 1:num_particles
            Q = diag(particles(i, 1:6));
            R = diag(particles(i, 7:8));

            x = repmat(x_init, 1, n_ensemble) + chol_P_init * randn(L, n_ensemble);
            x_est_pso = zeros(length(x_init), N);

            for k = 1:N
                for j = 1:n_ensemble
                    x(:,j) = F(x(:,j)) + chol(Q)' * randn(L, 1);
                end

                x_pred = mean(x, 2);

                if ~is_occluded(k) && ~is_out_of_view(k)
                    z = noisy_meas(k,:)';
                    Z = repmat(z, 1, n_ensemble) + chol(R)' * randn(size(z,1), n_ensemble);

                    H_x = arrayfun(@(j) H(x(:,j)), 1:n_ensemble,'UniformOutput',false);
                    H_x = cell2mat(H_x);

                    y_mean = mean(H_x, 2);
                    Y = H_x - y_mean;

                    P_xy = (x - x_pred) * Y' / (n_ensemble - 1);
                    P_yy = Y * Y' / (n_ensemble - 1) + R;

                    K = P_xy / (P_yy + eye(size(P_yy)) * 1e-8);

                    x = x + K * (Z - H_x);
                end

                x_est_pso(:,k) = mean(x, 2);
            end

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

        % Update velocities and positions
        for i = 1:num_particles
            velocities(i,:) = w * velocities(i,:) + ...
                c1 * rand(1,8) .* (pbest(i,:) - particles(i,:)) + ...
                c2 * rand(1,8) .* (gbest - particles(i,:));
            particles(i,:) = particles(i,:) + velocities(i,:);
            particles(i,:) = max(particles(i,:), 1e-6); % Ensure positive values
        end
    end

    Q_opt = diag(gbest(1:6));
    R_opt = diag(gbest(7:8));
end



% RTS smoothing function
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

% Numerical approximation of the Jacobian function
function J = jacobian(func, x)
    % Numerical approximation of the Jacobian
    h = 1e-5;
    n = length(x);
    J = zeros(n, n);
    for i = 1:n
        x_plus = x;
        x_plus(i) = x_plus(i) + h;
        J(:, i) = (func(x_plus) - func(x)) / h;
    end
end



%%%5 NEW CODE %%%

function [Q_opt, R_opt] = ga_optimize(F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view)

    % GA parameters
    population_size = 50;
    num_generations = 100;
    mutation_rate = 0.01;

    % Initialize population
    population = rand(population_size, 8); % 6 for Q diagonal, 2 for R diagonal

    for gen = 1:num_generations
        % Evaluate fitness
        fitness_values = zeros(population_size, 1);
        for i = 1:population_size
            fitness_values(i) = evaluate_fitness(population(i,:), F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view);
        end

        % Sort population by fitness
        [~, sorted_indices] = sort(fitness_values);
        population = population(sorted_indices,:);

        % Selection and Crossover
        next_generation = population(1:floor(population_size/2), :); % Select top half
        while size(next_generation, 1) < population_size
            parent_indices = randperm(floor(population_size/2), 2);
            parent1 = population(parent_indices(1), :);
            parent2 = population(parent_indices(2), :);
            crossover_point = randi([1, 8]);
            child = [parent1(1:crossover_point), parent2(crossover_point+1:end)];
            if rand() < mutation_rate
                mutation_index = randi([1, 8]);
                child(mutation_index) = rand();
            end
            next_generation(end+1,:) = child; %#ok<AGROW>
        end

        % Update population
        population = next_generation;
    end

    % Extract best solution
    best_solution = population(1,:);
    Q_opt = diag(best_solution(1:6)) + eye(6) * 1e-6;
    R_opt = diag(best_solution(7:8)) + eye(2) * 1e-6;

end

function [Q_opt, R_opt] = sa_optimize(F, H, x_init, P_init, noisy_meas, true_vals, is_occluded, is_out_of_view)

    % SA parameters
    initial_temperature = 100;
    final_temperature = 0.001;
    alpha = 0.9;
    max_iterations = 100;

    % Initialize current solution
    current_solution = rand(1, 8); % Initial random solution
    current_fitness = evaluate_fitness(current_solution,F,H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view);

    best_solution = current_solution;
    best_fitness = current_fitness;

    temperature_anneal= initial_temperature;

    while temperature_anneal > final_temperature
        for iter= 1:max_iterations
            new_solution= current_solution+ randn(1 ,8)* temperature_anneal ;
            new_solution(new_solution<0)=0; % Ensure non-negative values
            
            new_fitness= evaluate_fitness(new_solution,F,H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view);

            if new_fitness<current_fitness||exp((current_fitness-new_fitness)/temperature_anneal)>rand()
                current_solution=new_solution ;
                current_fitness=new_fitness ;

                if new_fitness<best_fitness 
                    best_solution=new_solution ;
                    best_fitness=new_fitness ;
                end
                
             end
            
         end
        
         temperature_anneal=alpha*temperature_anneal ; % Cool down the temperature
        
     end

     Q_opt=diag(best_solution(1:6))+eye(6)*10^-6 ;
     R_opt=diag(best_solution(7:8))+eye(2)*10^-6 ;

end

function fitness = evaluate_fitness(solution,F,H,x_init,P_init,noisy_meas,true_vals,is_occluded,is_out_of_view)

N=size(true_vals ,1);
x_est=zeros(length(x_init),N);

Q_sa=diag(solution(1:6))+eye(6)*10^-6 ;
R_sa=diag(solution(7:8))+eye(2)*10^-6 ;

x=x_init;
P=P_init;

for k=1:N
    
     x_pred=F(x);
     P_pred=P+Q_sa; 
    
     if ~is_occluded(k)&&~is_out_of_view(k)
         z=noisy_meas(k,:)';
         y=z-H*x_pred; 
         S=H*P_pred*H'+R_sa; 
         S=S+eye(size(S))*10^-6; 
         K=P_pred*H'/S; 
         x=x_pred+K*y; 
         P=(eye(length(x))-K*H)*P_pred; 
     else
         x=x_pred;
         P=P_pred;
     end

   x_est(:,k)=x;
    
end

fitness=sqrt(mean(sum((true_vals(:,1:2)'-x_est(1:2,:)).^2)));

end