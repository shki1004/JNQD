%% generate imdb.mat
function [imdb] = generate_imdb(data,labels)

images.id = 1:length(data(:,:,1,:));

cols = 1:length(data(:,:,1,:));

order = randperm(length(cols));
images.data = data(:, :, :, order);
images.labels = labels(:, :, :, order);

% data set
cols_Trainingset = intersect(cols,randsample(cols,round(0.75*length(cols)))); % 75% training set
cols_Testingset = cols(~ismember(cols,cols_Trainingset)); % 25% testing set

images.set(1,cols_Trainingset) = 1;
images.set(1,cols_Testingset) = 2;

imdb = [];
imdb.images = images;

end