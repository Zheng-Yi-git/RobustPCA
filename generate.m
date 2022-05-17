function [lowrank] = generate(n, s_rate, rank_rate)
s = s_rate * n * n;
rank = rank_rate * n;
temp = normrnd(0, 1, n, n);
[Q, R] = qr(temp);
tmp = Q(:, 1:rank);
lowrank = tmp * tmp' * 100;
pos = randi(n^2, 1, 2*s);
pos = unique(pos);
for i = 1:s
    t = pos(i);
    lowrank(ceil(t/n), mod(t, n)+1) = lowrank(ceil(t/n), mod(t, n)+1) + (rand()-0.5) * 100;
end
end

