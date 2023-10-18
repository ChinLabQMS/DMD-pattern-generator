%% Import a data file
[file, path] = uigetfile;

%% Pre-process the data
data = readmatrix(strcat(path, file), "FileType","text");
data = data(:, 2:end);

%% Plot the image
figure
imagesc(data)
colorbar
daspect([1 1 1])

%% Peak finder
signal = prctile(data, 100, 'all');
background = prctile(data, 50, 'all');
peak_range = ((background + (signal - background) * 0.2) <= data) & (data <= signal);
data_new = data;
data_new(~peak_range) = nan;

figure
imagesc(data_new)
colorbar
daspect([1 1 1])