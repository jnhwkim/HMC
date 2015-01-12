function merge_samples_batch()

%% Parameters
samples = 49;
variation = 'zjm';

%% 4,8,12,16 => 20,28,40
do_merge_samples(samples, variation, 4, 16);
do_merge_samples(samples, variation, 8, 20);
do_merge_samples(samples, variation, 12, 28);

end

function S_te = do_merge_samples(samples, variation, n1, n2)

%% Merge two sample sets.
te_label = sprintf('%ds%dr%s_te', samples, n1, variation);
load(strcat('mat/',te_label,'.mat'));
S1 = S_te;
te_label = sprintf('%ds%dr%s_te', samples, n2, variation);
load(strcat('mat/',te_label,'.mat'));
S2 = S_te;
S_te = merge_samples(S1, S2, n1, n2);

%% Save the result.
te_label = sprintf('%ds%dr%s_te', samples, n1+n2, variation);
save(strcat('mat/',te_label,'.mat'), 'S_te');

end