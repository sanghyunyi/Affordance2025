load 'model_recov_results/log_evidence.mat';

n_sim = 100;
n_model = 7;
count = zeros(n_model, n_model);
for i = 1:100
    for target_j = 1:n_model
        temp = nan(n_model, 1);
        for model_j = 1:n_model
            arr = bic(target_j, model_j);
            %temp(model_j) = sum(arr{1,1}(5*(i-1)+1:5*i));
            temp(model_j) = sum(arr{1,1}((i-1)+1:i));
        end
        [M, I] = min(temp);
        count(target_j, I) = count(target_j, I) + 1;
    end
end

count
