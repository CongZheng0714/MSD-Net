function [imdb] = generatepatches

addpath('utilities');
batchSize     = 128;        % batch size
global CurTask
if strcmpi(CurTask, 'Denoising')%= strcmpi(s1,s2) 将比较 s1 和 s2，并忽略字母大小写差异。如果二者相同，函数将返回 1 (true)，否则返回 0 (false)。如果文本的大小和内容相同，则它们将视为相等，不考虑大小写。返回结果 tf 的数据类型为 logical。
   dataName  ='TrainingPatches';
   folder    = 'Train_Set';  %
%     folder    = 'BSD300';
   crosswide_stride    = 5;%
   lengthways_stride   =10;
   stride = 10;
   scales  = [1]; % scale the image to augment the training data
else
   dataName  ='TrainingPatches';
   folder = 'Train_Set';
   crosswide_stride    = 5;%
   lengthways_stride   =10;
   stride = 10;

end
%nchannel      = 1;           % number of channels
patchsize     = 64;%贴片大小
step          = 0;
% step1         = randi(stride)-1;
% step2         = randi(stride)-1;
count         = 0;
ext           =  {'*.mat'};
filepaths     =  [];


for i = 1 : length(ext)
    filepaths = cat(1,filepaths, dir(fullfile(folder, ext{i})));
end
scales  = [1]; % scale the image to augment the training data% count the number of extracted patches
for i = 1 : length(filepaths)
    
    image = load(fullfile(folder,filepaths(i).name)); % uint8
%     if size(image,3)==3
%         if strcmpi(CurTask, 'Denoising')
%             image = rgb2gray(image);  %
%         else
     image = struct2cell(image);
     image = cell2mat(image);
           % image = image(:,:,1); 
end
   
    %[~, name, exte] = fileparts(filepaths(i).name);
    if mod(i,100)==0
        disp([i,length(filepaths)]);                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 
    end
    for s = 1:1
        %image = imresize(image,scales(s),'bicubic');
        [hei,wid,~] = size(image);
        for x = 1+step :crosswise_stride : (hei-patchsize+1)
            for y = 1+step :length_stride : (wid-patchsize+1)
                count = count+1;
            end
        end
    end
end

numPatches  = ceil(count/batchSize)*batchSize;
%diffPatches = numPatches - count;
disp([int2str(numPatches),' = ',int2str(numPatches/batchSize),' X ', int2str(batchSize)]);
disp([numPatches,batchSize,numPatches/batchSize]);
disp('输入规模，批规模，二者相除');


count = 0;
imdb.labels  = zeros(patchsize, patchsize, 1, numPatches,'single');
tic;%保存当前时间
for i = 1 : length(filepaths)
    image = load(fullfile(folder,filepaths(i).name)); % uint8
    image= struct2cell(image);
    image= cell2mat(image);
    %[~, name, exte] = fileparts(filepaths(i).name);
%     if size(image,3)==3
%         if strcmpi(CurTask, 'Denoising')
%             image = rgb2gray(image);  %
%         else
%             image = rgb2ycbcr(image);
%             image = image(:,:,1);
%         end
%     end
    if mod(i,100)==0
        disp([i,length(filepaths)]);
    end
    for s = 1:1
%         image = imresize(image,scales(s),'bicubic');
        for j = 1:1
%             image_aug   = data_augmentation(image, j);  % augment data
            im_label    = single(image);         % single
            [hei,wid,~] = size(im_label);
            
            for x = 1+step :crosswise_stride : (hei-patchsize+1)
                for y = 1+step :length_stride : (wid-patchsize+1)
                    count       = count+1;
%                     imdb.labels(:, :, :, count)   = im_label(x : x+patchsize-1, y : y+patchsize-1,:);
%                     if count<=diffPatches
                        inputs(:, :, :, count+1)   = im_label(x : x+patchsize-1, y : y+patchsize-1,:);
                    end
                end
            end
        end
    end
end
toc:%记录程序完成时间
set    = uint8(ones(1,size(inputs,4)));
disp('-------Datasize-------');
disp('输入规模，   批规模，    二者相除');
disp([size(inputs,4),batchSize,size(inputs,4)/batchSize]);
if ~exist(dataName,'file')
    mkdir(dataName);
end
save(fullfile(dataName,['imdb_',num2str(patchsize),'_',num2str(batchSize)]), 'inputs','set','-v7.3') ;