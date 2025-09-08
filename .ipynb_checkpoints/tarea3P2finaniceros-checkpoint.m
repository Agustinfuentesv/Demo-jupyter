clc; clear;

%% === PASO 1: Leer y limpiar datos ===
T = readtable('KOSPI Historical Data.csv', 'TextType', 'string');
T.Date = datetime(T.Date, 'InputFormat', 'MM/dd/yyyy');
T.Price = str2double(erase(T.Price, ','));
T = sortrows(T, 'Date');

fecha_ini = datetime(2019,7,1);
fecha_fin = datetime(2025,7,1);
T = T(T.Date >= fecha_ini & T.Date <= fecha_fin, :);
prices = T.Price;
dates = T.Date;

%% === PASO 2: Calcular retornos logarítmicos ===
returns = diff(log(prices));
ret_dates = dates(2:end);
n = length(returns);

%% === PASO 3: Ajuste manual GARCH(1,1) ===
loglik = @(theta) -garch_loglik(theta, returns);
theta0 = [0.01 * var(returns); 0.05; 0.9];
lb = [1e-6; 0; 0]; ub = [Inf; 1; 1];
A = [0, 1, 1]; b = 0.999;

opts = optimoptions('fmincon', 'Display', 'final', 'Algorithm', 'sqp');
[theta_est, ~, flag] = fmincon(loglik, theta0, A, b, [], [], lb, ub, [], opts);
if flag <= 0, error('❌ La optimización no convergió.'); end

omega = theta_est(1); alpha = theta_est(2); beta  = theta_est(3);

%% === PASO 4: Volatilidad condicional ===
sigma2 = zeros(n, 1); sigma2(1) = var(returns);
for t = 2:n
    sigma2(t) = omega + alpha * returns(t-1)^2 + beta * sigma2(t-1);
end
volatility = sqrt(sigma2);

%% === PASO 5: Gráfico de volatilidad ===
figure;
plot(ret_dates, volatility, 'r', 'LineWidth', 1.5);
title('Volatilidad estimada por GARCH(1,1)');
xlabel('Fecha'); ylabel('Volatilidad'); grid on;

%% === PASO 6: Evaluación del modelo ===
residuales = returns ./ volatility;
fprintf('\n--- Parámetros estimados ---\n');
fprintf('Omega = %.6e\nAlpha = %.4f\nBeta = %.4f\nAlpha+Beta = %.4f\n', omega, alpha, beta, alpha+beta);

% Cálculo de medidas estadísticas
media_res = mean(residuales);
std_res = std(residuales);
kurt_res = kurtosis(residuales);
asim_res = skewness(residuales);  % Asimetría de Fisher

fprintf('\n--- Residuos ---\n');
fprintf('Media = %.4f\nStd = %.4f\nKurtosis = %.4f\nAsimetría (Fisher) = %.4f\n', ...
        media_res, std_res, kurt_res, asim_res);


%% === PASO 7: SIMULACIÓN DE PRECIOS ===
n_sim = 5; T_pred = 140; mu = 0.0002;
figure;
for i = 1:n_sim
    r = zeros(T_pred,1); sigma2 = zeros(T_pred,1);
    sigma2(1) = omega / (1 - alpha - beta);
    for t = 2:T_pred
        epsilon = randn;
        r(t) = mu + sqrt(sigma2(t-1)) * epsilon;
        sigma2(t) = omega + alpha * r(t-1)^2 + beta * sigma2(t-1);
    end
    precios = prices(end) * exp(cumsum(r));
    subplot(n_sim, 1, i);
    plot(precios, 'b'); hold on;
    plot(101:T_pred, precios(101:end), 'm');
    ylim([min(precios)*0.95, max(precios)*1.05]);
end
sgtitle('Predicción de Precios');

%% === FUNCIÓN AUXILIAR ===
function L = garch_loglik(theta, r)
    omega = theta(1); alpha = theta(2); beta = theta(3);
    n = length(r); sigma2 = zeros(n,1);
    sigma2(1) = var(r);  % inicialización segura
    for t = 2:n
        sigma2(t) = omega + alpha * r(t-1)^2 + beta * sigma2(t-1);
    end
    if any(sigma2 <= 0)
        L = Inf;
    else
        L = -0.5 * sum(log(2*pi) + log(sigma2) + r.^2 ./ sigma2);
    end
end

