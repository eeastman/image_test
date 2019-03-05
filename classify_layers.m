layer_dir = '~/Desktop/';
layers = {'pool5'};
NPY_matlab_dir = '~/repos/npy-matlab/npy-matlab/';
addpath(NPY_matlab_dir)

resample_runs = 20;
run_pca = 1;
num_PCs = 50;
acc = zeros(length(layers), resample_runs);
ho_scenes = 1; %% 1 to hold out two scnees, 0 for random 80-20 split

for l = 1:length(layers)
%% Load data
layer_file = [layer_dir layers{l} '.npy'];
layer = readNPY(layer_file);
layer_reshape = reshape(layer,52,[]);

%% Run PCA
if run_pca
%imageFeatures_orig = imageFeatures;
[coeff,imageFeatures_pca,latent,tsquared,explained,mu] = pca(layer_reshape); 
imageFeatures = imageFeatures_pca(:,1:num_PCs);
else
imageFeatures = layer_reshape;
end

%% Remove 013 images
imageFeatures = imageFeatures([1:24, 27:50],:);
labels = [ones(1,24), 2*ones(1,24)];
inds = [randperm(12) randperm(12)];

for i = 1:resample_runs
    if ho_scenes
    %% Hold two scenes out
    test = [(inds(i)-1)*2+[1:2, 25:26] (inds(i+1)-1)*2+[1:2, 25:26]];
    train = setdiff(1:48, test);
    else
    %% Random 80-20 crossvalidation split
    [train,test] = crossvalind('holdout',labels,0.2);
    end
    
    train_labels = labels(train);
    test_labels = labels(test)';
    train_data = imageFeatures(train,:);
    test_data = imageFeatures(test,:);

    SVMStruct = fitcsvm(train_data,train_labels);
    pred = predict(SVMStruct, test_data);
    acc(l,i) = sum(pred==test_labels)/length(test_labels);
     
end
end

%% plot results
figure
bar(mean(acc,2)); hold on
errorbar(mean(acc,2), std(acc'), 'k')
plot([0 l+1], [0.5 0.5], 'k--')
set(gca, 'XTickLabel', layers)
set(gca, 'XTickLabelRotation', 45)
set(gca, 'FontSize', 20)
ylabel('Classification accuracy')
xlim([0 l+1])

    