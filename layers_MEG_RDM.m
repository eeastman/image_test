clear all

layer_dir = '~/Desktop/';
layers = {'pool5'};
NPY_matlab_dir = '~/repos/npy-matlab/npy-matlab/';
MEG_file = 'rsa_decoding_exp2_reorder.mat'; %% Add
addpath(NPY_matlab_dir)

run_pca = 1;
num_PCs = 50;
ho_scenes = 1; %% 1 to hold out two scnees, 0 for random 80-20 split

%% Load MEG results
load(MEG_file)

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

%% Calculate corr over subj/time
CNN_rdm(:,l) = pdist(imageFeatures);

for s = 1:size(md,1)
    for t = 1:size(md,2)
        MEG_layer_corr(s,t,l) = corr(squeeze(CNN_rdm(:,l)), squeeze(md(s,t,:)));
    end
end
end

%% plot results
time = -226:10:964;
figure
plot(time,squeeze(mean(MEG_layer_corr)))
hold on;
plot([time(1) time(end)], [0 0], 'k')
xlabel('Time (ms)')
xlim([time(1) time(end)])
set(gca, 'FontSize', 20)
ylabel('MEG_model_correlation')
legend(layers)

    