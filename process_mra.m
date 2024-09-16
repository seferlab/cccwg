train_data = double(train_data);
test_data = double(test_data);

a = [train_data, test_data];
len = length(a);

lev = 5;

wname = 'haar';
wavs_a = (modwt(a(1, :), wname, lev));
wavs_b = (modwt(a(2, :), wname, lev));

train_data = single(train_data);
test_data = single(test_data);
wavs_a = single(wavs_a);
wavs_b = single(wavs_b);