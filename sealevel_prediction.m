clc;
clear all;
close all;

% Load the dataset
data = readtable('C:\Users\varsh\OneDrive\Documents\globalsealevelnew.csv'); % Update the file path
% Consider only data from 1983-Apr to 2013-Sep (last 30 years)
start_idx = find(strcmp(data.Time, '1983-Apr'));
end_idx = find(strcmp(data.Time, '2013-Sep'));
data = data(start_idx:end_idx, :);

% Convert time to numeric values (years from 1983)
time = (1:height(data))'; % Time as numeric values (relative to 1983)
GMSL = data.GMSL;
GMSLuncertainty = data.GMSLuncertainty; % Including uncertainty

% Remove missing data
time = time(~isnan(GMSL) & ~isnan(GMSLuncertainty)); % Removing rows with missing GMSL or uncertainty
GMSL = GMSL(~isnan(GMSL) & ~isnan(GMSLuncertainty));
GMSLuncertainty = GMSLuncertainty(~isnan(GMSL) & ~isnan(GMSLuncertainty));

% Split data into training and testing (last 30 years)
split_index = round(0.8 * length(GMSL)); % 80% training, 20% testing
train_time = time(1:split_index);
train_GMSL = GMSL(1:split_index);
train_GMSLuncertainty = GMSLuncertainty(1:split_index); % Training uncertainty
test_time = time(split_index+1:end);
test_GMSL = GMSL(split_index+1:end);
test_GMSLuncertainty = GMSLuncertainty(split_index+1:end); % Testing uncertainty

% Convert time to years
start_year = 1983; % Starting year
train_time_years = start_year + (train_time - 1) / 12; % Adjusting for months
test_time_years = start_year + (test_time - 1) / 12; % Adjusting for months

% Plot original data with uncertainty
figure;
bar(train_time_years, train_GMSL, 'FaceColor', [0.2, 0.5, 0.9]);
xlabel('Time (Years)');
ylabel('Sea Level Rise (mm)');
title('Sea Level Rise: Original Data (1983-2013)');
grid on;
hold on;

% Create transparent uncertainty shading
upper_bound = train_GMSL + train_GMSLuncertainty;  % Upper bound with uncertainty
lower_bound = train_GMSL - train_GMSLuncertainty;  % Lower bound with uncertainty
fill([train_time_years; flipud(train_time_years)], ...
     [upper_bound; flipud(lower_bound)], ...
     [0.2, 0.5, 0.9], 'FaceAlpha', 0.1, 'EdgeColor', 'none'); % Shaded region for uncertainty

errorbar(train_time_years, train_GMSL, train_GMSLuncertainty, 'k.', 'MarkerSize', 10); % Add uncertainty as error bars
legend('Original Data', 'Uncertainty');
hold off;


% --- Linear Regression Training ---
fprintf('Training Linear Regression...\n');
tic; % Start timer
mdl = fitlm(train_time_years, train_GMSL); % Train Linear Regression model
lr_train_time = toc; % Elapsed time for training
fprintf('Linear Regression Training Time: %.4f seconds\n', lr_train_time);

% Linear Regression Predictions
predicted_lr = predict(mdl, test_time_years);

% Calculate MAE for Linear Regression
mae_lr = mean(abs(predicted_lr - test_GMSL));
fprintf('Linear Regression MAE: %.2f mm\n', mae_lr);

% Future Year Predictions for Linear Regression (after 2014)
future_years_lr = (2014:2100)'; % Years for future prediction
future_lr_features = future_years_lr; % Features are simply future years in this case
predicted_future_lr = predict(mdl, future_lr_features); % Predict for future years
[max_future_lr, idx_future_lr] = max(predicted_future_lr + GMSLuncertainty(end)); % Add uncertainty for future predictions
max_future_lr_year = future_years_lr(idx_future_lr); % Corresponding year for highest prediction

fprintf('Highest Predicted Future Sea Level (LR): %.2f mm in Year: %.0f\n', max_future_lr, max_future_lr_year);

% Plot Linear Regression Results
figure;
subplot(2, 1, 1);
bar(train_time_years, train_GMSL, 'FaceColor', [0.2, 0.5, 0.9]);
hold on;
% Add uncertainty shading for original data (training set)
fill([train_time_years; flipud(train_time_years)], ...
    [train_GMSL + train_GMSLuncertainty; flipud(train_GMSL - train_GMSLuncertainty)], ...
    [0.2, 0.5, 0.9], 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Light transparent shading
title('Sea Level Rise: Original Data (1983-2013)');
xlabel('Time (Years)');
ylabel('Sea Level Rise (mm)');

subplot(2, 1, 2);
plot(train_time_years, train_GMSL, 'b', 'LineWidth', 1.5); hold on;
plot(test_time_years, test_GMSL, 'g', 'LineWidth', 1.5);
plot(test_time_years, predicted_lr, 'r--', 'LineWidth', 1.5);
% Add uncertainty shading for forecasted data
fill([test_time_years; flipud(test_time_years)], ...
    [predicted_lr + GMSLuncertainty(end); flipud(predicted_lr - GMSLuncertainty(end))], ...
    [1, 0, 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Light transparent shading for forecast
legend('Training Data', 'Test Data', 'Predicted');
title('Sea Level Forecast Using Linear Regression');
xlabel('Time (Years)');
ylabel('Sea Level Rise (mm)');
grid on;


% --- SVM Training --
fprintf('Training SVM...\n');
tic; % Start timer
svm_model = fitrsvm(train_time_years, train_GMSL, 'KernelFunction', 'linear'); % Train SVM
svm_train_time = toc; % Elapsed time for training
fprintf('SVM Training Time: %.4f seconds\n', svm_train_time);

% SVM Predictions
predicted_svm = predict(svm_model, test_time_years);

% Calculate MAE for SVM
mae_svm = mean(abs(predicted_svm - test_GMSL));
fprintf('SVM MAE: %.2f mm\n', mae_svm);

% Future Year Predictions for SVM (after 2014)
predicted_future_svm = predict(svm_model, future_years_lr);
[max_future_svm, idx_future_svm] = max(predicted_future_svm + GMSLuncertainty(end)); % Add uncertainty for future predictions
max_future_svm_year = future_years_lr(idx_future_svm); % Corresponding year for highest prediction

fprintf('Highest Predicted Future Sea Level (SVM): %.2f mm in Year: %.0f\n', max_future_svm, max_future_svm_year);

% Plot SVM Results with Uncertainty
figure;
subplot(2, 1, 1);
bar(train_time_years, train_GMSL, 'FaceColor', [0.2, 0.5, 0.9]);
hold on;
% Add uncertainty shading for original data
fill([train_time_years; flipud(train_time_years)], ...
     [train_GMSL - GMSLuncertainty(1:length(train_GMSL)); flipud(train_GMSL + GMSLuncertainty(1:length(train_GMSL)))], ...
     [0.2, 0.5, 0.9], 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Light transparent shading
title('Sea Level Rise: Original Data (1983-2013)');
xlabel('Time (Years)');
ylabel('Sea Level Rise (mm)');

subplot(2, 1, 2);
plot(train_time_years, train_GMSL, 'b', 'LineWidth', 1.5); hold on;
plot(test_time_years, test_GMSL, 'g', 'LineWidth', 1.5);
plot(test_time_years, predicted_svm, 'r--', 'LineWidth', 1.5);
% Add uncertainty shading for forecasted data
fill([test_time_years; flipud(test_time_years)], ...
     [predicted_svm - GMSLuncertainty(end-length(test_time_years)+1:end); flipud(predicted_svm + GMSLuncertainty(end-length(test_time_years)+1:end))], ...
     [1, 0, 0], 'FaceAlpha', 0.2, 'EdgeColor', 'none'); % Light transparent shading
legend('Training Data', 'Test Data', 'Predicted');
title('Sea Level Forecast Using SVM');
xlabel('Time (Years)');
ylabel('Sea Level Rise (mm)');
grid on;


% --- Neural Network Training ---
fprintf('Training Neural Network...\n');
input_size = 5;
train_features = zeros(length(train_GMSL)-input_size, input_size);
train_targets = train_GMSL(input_size+1:end);

for i = 1:(length(train_GMSL)-input_size)
    train_features(i, :) = train_GMSL(i:i+input_size-1)';
end

test_features = zeros(length(test_GMSL)-input_size, input_size);
test_targets = test_GMSL(input_size+1:end);

for i = 1:(length(test_GMSL)-input_size)
    test_features(i, :) = test_GMSL(i:i+input_size-1)';
end

tic;
net = fitnet([10, 5]); % Define neural network architecture
[net, tr] = train(net, train_features', train_targets'); % Train the network
nn_train_time = toc;
predicted_nn = net(test_features')'; % Predictions on test data
mae_nn = mean(abs(predicted_nn - test_targets));

fprintf('Neural Network Training Time: %.4f seconds\n', nn_train_time);
fprintf('Neural Network MAE: %.2f mm\n', mae_nn);

% Adjust prediction length for plotting
% Ensure that predicted_nn matches the length of test_time
predicted_nn_full = nan(length(test_time_years), 1);
predicted_nn_full(input_size+1:end) = predicted_nn;

% Future Year Predictions for Neural Network (after 2014)
future_years_nn = (2014:2100)'; % Years for future prediction
future_nn_features = zeros(length(future_years_nn), input_size); % Features for future predictions

% Generate features for future years using sliding window
for i = 1:length(future_years_nn)
    if future_years_nn(i) <= 2014
        future_nn_features(i, :) = GMSL(i:i+input_size-1)';
    end
end

% Predictions for future years
predicted_future_nn = net(future_nn_features')';
[max_future_nn, idx_future_nn] = max(predicted_future_nn + GMSLuncertainty(end)); % Add uncertainty
max_future_nn_year = future_years_nn(idx_future_nn); % Year for highest prediction

fprintf('Highest Predicted Future Sea Level (NN): %.2f mm in Year: %.0f\n', max_future_nn, max_future_nn_year);

% Plot Neural Network Results with Uncertainty
figure;

% --- Original Data Subplot ---
subplot(2, 1, 1);
bar(train_time_years, train_GMSL, 'FaceColor', [0.2, 0.5, 0.9]);
hold on;
% Add uncertainty for original data (using fill for transparency)
fill([train_time_years; flipud(train_time_years)], ...
    [train_GMSL + train_GMSLuncertainty; flipud(train_GMSL - train_GMSLuncertainty)], ...
    [0.2, 0.5, 0.9], 'FaceAlpha', 0.3, 'EdgeColor', 'none'); % Light and transparent shading
title('Sea Level Rise: Original Data (1983-2013)');
xlabel('Time (Years)');
ylabel('Sea Level Rise (mm)');

% --- Forecasted Data Subplot ---
subplot(2, 1, 2);
plot(train_time_years, train_GMSL, 'b', 'LineWidth', 1.5); hold on;
plot(test_time_years, test_GMSL, 'g', 'LineWidth', 1.5);
plot(test_time_years, predicted_nn_full, 'r--', 'LineWidth', 1.5);

% Add uncertainty for forecasted data (using fill for transparency)
fill([test_time_years; flipud(test_time_years)], ...
    [predicted_nn_full + GMSLuncertainty(end); flipud(predicted_nn_full - GMSLuncertainty(end))], ...
    'r', 'FaceAlpha', 0.3, 'EdgeColor', 'none'); % Light and transparent shading
legend('Training Data', 'Test Data', 'Predicted');
title('Sea Level Forecast Using Neural Network');
xlabel('Time (Years)');
ylabel('Sea Level Rise (mm)');
grid on;

% --- Final Comparison ---
figure;
subplot(2, 1, 1);
plot(test_time_years, test_GMSL, 'g', 'LineWidth', 1.5); hold on;
plot(test_time_years, predicted_lr, 'r--', 'LineWidth', 1.5);
plot(test_time_years, predicted_svm, 'b--', 'LineWidth', 1.5);
plot(test_time_years, predicted_nn_full, 'm--', 'LineWidth', 1.5);
legend('Test Data', 'Linear Regression', 'SVM', 'Neural Network');
xlabel('Time (Years)');
ylabel('Sea Level Rise (mm)');
title('Sea Level Forecast Comparison');
grid on;

% --- User Input for Year Prediction ---
year_input = input('Enter a year between 2014 and 2100 for prediction: ');

% Ensure the input is within the valid range
if year_input < 2014 || year_input > 2100
    error('Year must be between 2014 and 2100.');
end

% Predict using Linear Regression
predicted_lr = predict(mdl, year_input);
fprintf('Predicted Sea Level for %d (Linear Regression): %.2f mm\n', year_input, predicted_lr);

% Predict using SVM
predicted_svm = predict(svm_model, year_input);
fprintf('Predicted Sea Level for %d (SVM): %.2f mm\n', year_input, predicted_svm);

% Prepare features for future year prediction using Neural Network
future_nn_features = zeros(1, input_size); % Features for future prediction
% Use the last input_size values from the training data for predictions
future_nn_features(1, :) = GMSL(end-input_size+1:end);

% Predict using Neural Network
predicted_nn = net(future_nn_features')';
fprintf('Predicted Sea Level for %d (Neural Network): %.2f mm\n', year_input, predicted_nn);

% Plot the predictions for the entered year with different bar colors
figure;
bar_colors = [0.2, 0.5, 0.9; 0.9, 0.3, 0.3; 0.3, 0.7, 0.2]; % Different colors for the models
b = bar([1, 2, 3], [predicted_lr, predicted_svm, predicted_nn], 'FaceColor', 'flat');
b.FaceColor = 'flat';
b.CData = bar_colors; % Set the different colors for each model

% Set axis labels and title
set(gca, 'XTick', [1, 2, 3], 'XTickLabel', {'Linear Regression', 'SVM', 'Neural Network'});
ylabel('Predicted Sea Level Rise (mm)');
title(['Sea Level Rise Prediction for the Year ', num2str(year_input)]);
grid on;


% --- Predictions on Test Data ---
% Linear Regression Prediction on Test Data
predicted_lr_test = predict(mdl, test_time_years); 
mse_lr = mean((predicted_lr_test - test_GMSL).^2); % Mean Squared Error for LR
rmse_lr = sqrt(mse_lr); % Root Mean Squared Error for LR

% SVM Prediction on Test Data
predicted_svm_test = predict(svm_model, test_time_years);
mse_svm = mean((predicted_svm_test - test_GMSL).^2); % Mean Squared Error for SVM
rmse_svm = sqrt(mse_svm); % Root Mean Squared Error for SVM

% Neural Network Prediction on Test Data
test_nn_features = zeros(length(test_GMSL)-input_size, input_size);
test_nn_targets = test_GMSL(input_size+1:end);

for i = 1:(length(test_GMSL)-input_size)
    test_nn_features(i, :) = test_GMSL(i:i+input_size-1)';
end

predicted_nn_test = net(test_nn_features')';
mse_nn = mean((predicted_nn_test - test_GMSL(input_size+1:end)).^2); % Mean Squared Error for NN
rmse_nn = sqrt(mse_nn); % Root Mean Squared Error for NN

% --- Display RMSE for Each Model ---
fprintf('Root Mean Squared Error for Linear Regression: %.2f mm\n', rmse_lr);
fprintf('Root Mean Squared Error for SVM: %.2f mm\n', rmse_svm);
fprintf('Root Mean Squared Error for Neural Network: %.2f mm\n', rmse_nn);

% --- Sort Models by RMSE ---
errors = [rmse_lr, rmse_svm, rmse_nn];
models = {'Linear Regression', 'SVM', 'Neural Network'};

% Sort models based on RMSE (ascending order, best model first)
[sorted_errors, sorted_idx] = sort(errors); % Sorting errors and corresponding indices
sorted_models = models(sorted_idx); % Sorted models based on RMSE

% --- Plot 1: Model Performance Comparison ---
figure;

% Bar chart for RMSE values (performance comparison)
bar(errors, 'FaceColor', [0.2, 0.5, 0.9]); % Original bar chart for RMSE values
set(gca, 'xticklabel', models); % Set the x-tick labels to the original models
title('Model Performance Comparison (RMSE)');
xlabel('Model');
ylabel('Root Mean Squared Error (mm)');
grid on;

% --- Plot 2: Best Model Ranking ---
figure;

% Bar chart for sorted RMSE values (best model ranking)
hold on;
bar(sorted_errors, 'FaceColor', [0.2, 0.7, 0.2]); % Bar chart for sorted RMSE values
set(gca, 'xticklabel', sorted_models); % Set the x-tick labels to the sorted models

% Highlight the bars with different colors
colors = [0.2, 0.7, 0.2; 0.8, 0.8, 0.2; 0.8, 0.2, 0.2]; % Green for best, yellow for second, red for third
for i = 1:length(sorted_errors)
    bar(i, sorted_errors(i), 'FaceColor', colors(i, :), 'LineWidth', 1.5); % Apply different colors
end

% Plot settings
xlabel('Model');
ylabel('Root Mean Squared Error (mm)');
title('Best Model Ranking (Sorted by RMSE)');
legend('Model Performance', 'Best Model', 'Second Best Model', 'Third Best Model', 'Location', 'NorthEast');
grid on;
hold off;