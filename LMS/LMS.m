clear clc
X = 1:10;
Y = 5*X + 3 + 3*sin(1:10);
Y = Y';
%scatter(X, Y)

one = ones(10,1);
X = [one, X'];
alpha = 0.5;
d = 10;
L = zeros(2*d+1, 2*d+1);
for i = -d:d
    for j = -d:d
        for k = 1:10
            L(i+d+1,j+d+1) = L(i+d+1, j+d+1) + (i + j*X(k,2) - Y(k)).^2;
        end
        L(i+d+1,j+d+1) = L(i+d+1,j+d+1)/2;
    end
end

hold on
contour(-d:d, -d:d, L, 'showtext', 'on')
T = [0;0];
a = 0.001;

while sum((Y-X*T).^2)/2 > 21
    Y_hat = X*T;
    T = T + a*((Y - Y_hat)'*X)';
    plot(T(1),T(2), '*r')
    disp(sum((Y-X*T).^2)/2)
end
plot(T(1), T(2), '*b')



