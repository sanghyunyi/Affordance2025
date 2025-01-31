load lap_out/lap_RL_performance_2_avg_aff_score_precise_rt.mat;
target_params = cbm.output.parameters;
load lap_out/lap_RL_performance_avg_aff_score_precise_rt.mat;
recovered_params = cbm.output.parameters;

tiledlayout(5,2);
axis square;

% Plot 1: Alpha
target_alpha = target_params(:,1);
recov_alpha = recovered_params(:,1);
nexttile
scatter(target_alpha, recov_alpha);
xlabel('Updated');
ylabel('Original');
addCorrelationTitle(target_alpha, recov_alpha, 'Alpha');
adjustLimits(target_alpha, recov_alpha);

% Plot 2: Beta
target_beta = target_params(:,2);
recov_beta = recovered_params(:,2);
nexttile
scatter(target_beta, recov_beta);
xlabel('Updated');
ylabel('Original');
addCorrelationTitle(target_beta, recov_beta, 'Beta');
adjustLimits(target_beta, recov_beta);

% Plot 3: Beta Aff
target_beta_aff = target_params(:,3);
recov_beta_aff = recovered_params(:,3);
nexttile
scatter(target_beta_aff, recov_beta_aff);
xlabel('Updated');
ylabel('Original');
addCorrelationTitle(target_beta_aff, recov_beta_aff, 'Beta Aff');
adjustLimits(target_beta_aff, recov_beta_aff);

% Plot 4: Beta Perf
target_beta_perf = target_params(:,4);
recov_beta_perf = recovered_params(:,4);
nexttile
scatter(target_beta_perf, recov_beta_perf);
xlabel('Updated');
ylabel('Original');
addCorrelationTitle(target_beta_perf, recov_beta_perf, 'Beta Perf');
adjustLimits(target_beta_perf, recov_beta_perf);

% Plot 5: Beta0 Perf
target_beta0_perf = target_params(:,5);
recov_beta0_perf = recovered_params(:,5);
nexttile
scatter(target_beta0_perf, recov_beta0_perf);
xlabel('Updated');
ylabel('Original');
addCorrelationTitle(target_beta0_perf, recov_beta0_perf, 'Beta0 Perf');
adjustLimits(target_beta0_perf, recov_beta0_perf);

% Plot 6: B Clench - B Pinch
target_b_pinch = target_params(:,6);
recov_b_pinch = recovered_params(:,7)-recovered_params(:,6);
nexttile
scatter(target_b_pinch, recov_b_pinch);
xlabel('Updated');
ylabel('Original');
addCorrelationTitle(target_b_pinch, recov_b_pinch, 'B Clench - B Pinch');
adjustLimits(target_b_pinch, recov_b_pinch);

% Plot 7: B Poke - B Clench
target_b_clench = target_params(:,7);
recov_b_clench = recovered_params(:,8)-recovered_params(:,6);
nexttile
scatter(target_b_clench, recov_b_clench);
xlabel('Updated');
ylabel('Original');
addCorrelationTitle(target_b_clench, recov_b_clench, 'B Poke - B Pinch');
adjustLimits(target_b_clench, recov_b_clench);



% Helper function to adjust axis limits
function adjustLimits(x, y)
    margin = 0.1 * max(abs([x; y])); % calculate 10% margin of the max range
    lims = [min([x; y]) - margin, max([x; y]) + margin];
    xlim(lims);
    ylim(lims);
end

% Helper function to add title with correlation coefficient and p-value
function addCorrelationTitle(x, y, variableName)
    [R, P] = corrcoef(x, y);
    if numel(R) > 1 && numel(P) > 1 % Ensure R and P are matrices
        corrValue = R(1,2);
        pValue = P(1,2);
        if pValue < 0.001
            title(sprintf('%s\nr = %.2f, p < 0.001', variableName, corrValue));
        else
            title(sprintf('%s\nr = %.2f, p = %.3f', variableName, corrValue, pValue));
        end
    end
end
