n_seq = [100, 200, 500, 1000];
rate_seq = [0.01, 0.05, 0.1];
for i = 1:4
    for j = 1:2
        for k = 2:3
            rank_hat = 0;
            zero_hat = 0;
            Iter = 0;
            tm = 0;
            for p = 1:20
                t0 = cputime;
                [A, E, Y, mu, iter] = IALM(generate(n_seq(i), rate_seq(j), rate_seq(k)));
                t1 = cputime - t0;
                rank_hat = rank_hat + rank(A);
                zero_hat = zero_hat + sum(sum(abs(E)>0.01));
                Iter = Iter + iter;
                tm = tm + t1;
            end
            sprintf("n: %f, zero_rate : %f, rank_rate: %f", n_seq(i), rate_seq(j), rate_seq(k))
            sprintf("rank_avg: %f, zero_avg: %f, iter_avg: %f, time_avg: %f", rank_hat/20, zero_hat/20, Iter/20, tm/20)
        end
    end
end