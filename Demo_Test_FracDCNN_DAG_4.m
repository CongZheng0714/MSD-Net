
addpath('matconvnet-1.0-beta25\matlab');
vl_compilenn; 
addpath(fullfile('utilities'));
folder_test = 'data';%
showresult  = 1;
gpu         = 1; 
load('pure.mat')
load('noise.mat')
load('./model/MSDNet.mat');
net = dagnn.DagNN.loadobj(net) ;
net.removeLayer('loss') ;
out1 = net.getVarIndex('prediction') ;
net.vars(net.getVarIndex('prediction')).precious = 1 ;
net.mode = 'test';
if gpu
    net.move('gpu');
end
% read images
ext=  {'*.mat'};
filePaths_test   =  [];
for i = 1 : length(ext)
    filePaths_test = cat(1,filePaths_test, dir(fullfile(folder_test, ext{i})));
end

for i = 1 : length(filePaths_test)
    label = load(fullfile(folder_test,filePaths_test(i).name));                                                                  
    label = struct2cell(label);
    label = cell2mat(label);
    label = modcrop(label,8);
    input = single(label);
    if gpu
       gpu_input = gpuArray(input);
    end
    
    tic
    net.eval({'input', gpu_input}) ;
    toc
    
    output = gather(squeeze(gather(net.vars(out1).value)));   
   
end

SNR = 10*log(sum(sum(pure.^2))/sum(sum((output-pure).^2)))/log(10);
RMSE = sqrt(sum(sum((output-pure).^2))/2400/240);


