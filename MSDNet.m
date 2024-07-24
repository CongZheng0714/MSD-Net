 function net = MSDNet()
net = dagnn.DagNN();
blockNum = 1;
inVar0 = 'input';
[net,inVar11,blockNum]=addPooling(net,blockNum,inVar0,'max');
[net,inVar,blockNum]=addConv(net,blockNum,inVar11,[3,3,1,64],[1,1],[1,1],1,[1,1]);
[net,inVar1A,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar12,blockNum]=addPooling(net,blockNum,inVar11,'max');
[net,inVar,blockNum]=addConv(net,blockNum,inVar12,[3,3,1,128],[1,1],[1,1],1,[1,1]);
[net,inVar2A,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addPooling(net,blockNum,inVar12,'max');
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,1,256],[1,1],[1,1],1,[1,1]);
[net,inVar3A,blockNum]=addReLU(net,blockNum,inVar);

[net,inVar,blockNum]=addConv(net,blockNum,inVar0,[3,3,1,64],[1,1],[1,1],1,[1,1]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,64,64],[2,2],[1,1],2,[1,1]);
[net,inVar2D,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar2D,[3,3,64,64],[3,3],[1,1],3,[1,1]);
[net,inVar3D,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar2D,[3,3,64,64],[1,1],[1,1],1,[1,1]);
[net,inVar2E,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar3D,[3,3,64,64],[1,1],[1,1],1,[1,1]);
[net,inVar2F,blockNum]=addReLU(net,blockNum,inVar);
[net,inVarA,blockNum]=addSum(net,blockNum,{inVar2D,inVar2E,inVar2F});

[net,inVar,blockNum]=addConv(net,blockNum,inVarA,[3,3,64,64],[1,1],[1,1],1,[1,1]);
[net,inVarB,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVarB,[3,3,64,64],[1,1],[1,1],1,[1,1]);
[net,inVarC,blockNum]=addReLU(net,blockNum,inVar);

[net,inVar,blockNum]=addConv(net,blockNum,inVarC,[3,3,64,64],[1,1],[2,2],1,[1,0]);%1/2
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,64,64],[1,1],[1,1],1,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,64,64],[1,1],[1,1],1,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addSum(net,blockNum,{inVar,inVar1A});
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,64,128],[1,1],[1,1],1,[1,0]);
[net,inVar2,blockNum]=addReLU(net,blockNum,inVar);

[net,inVar,blockNum]=addConv(net,blockNum,inVar2,[3,3,128,128],[1,1],[2,2],1,[1,0]);%1/4
[net,inVar3,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar3,[3,3,128,128],[1,1],[1,1],1,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,128,128],[1,1],[1,1],1,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addSum(net,blockNum,{inVar,inVar2A});
[net,inVar,blockNum]=addConvt(net,blockNum,inVar,[2,2,128,128],0,2,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,128,128],[1,1],[1,1],1,[1,0]);
[net,inVar3U,blockNum]=addReLU(net,blockNum,inVar);

[net,inVar,blockNum]=addConv(net,blockNum,inVar3,[3,3,128,128],[1,1],[2,2],1,[1,0]);%1/8
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,128,128],[1,1],[1,1],1,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,128,256],[1,1],[1,1],1,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addSum(net,blockNum,{inVar,inVar3A});
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,256,128],[1,1],[1,1],1,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConvt(net,blockNum,inVar,[2,2,128,128],0,2,[1,0]);%
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,128,128],[1,1],[1,1],1,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConvt(net,blockNum,inVar,[2,2,128,128],0,2,[1,0]);%
[net,inVar4U,blockNum]=addReLU(net,blockNum,inVar);

%AG
[net,inVara,blockNum]=addConv(net,blockNum,inVar2,[1,1,128,128],[0,0],[1,1],1,[1,0]);
[net,inVarb,blockNum]=addConv(net,blockNum,inVar3U,[1,1,128,128],[0,0],[1,1],1,[1,0]);
[net,inVarc,blockNum]=addConv(net,blockNum,inVar4U,[1,1,128,128],[0,0],[1,1],1,[1,0]);
[net,inVar,blockNum]=addSum(net,blockNum,{inVara,inVarb,inVarc});

[net,inVar,blockNum]=addConv(net,blockNum,inVar,[1,1,128,128],[0,0],[1,1],1,[1,0]);
[net,inVarabc,blockNum]=addSigmoid(net,blockNum,inVar);
[net,inVar,blockNum]=addDot(net,blockNum,{inVarabc,inVar2});
[net,inVar,blockNum]=addConcat(net,blockNum,{inVar,inVar4U});
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,256,128],[1,1],[1,1],1,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConvt(net,blockNum,inVar,[2,2,128,128],0,2,[1,0]);
[net,inVar,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,128,64],[1,1],[1,1],1,[1,0]);
[net,inVarD,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addSum(net,blockNum,{inVarC,inVarD});
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,64,64],[1,1],[1,1],1,[1,0]);
[net,inVarE,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addSum(net,blockNum,{inVarB,inVarE});
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,64,64],[1,1],[1,1],1,[1,0]);
[net,inVarF,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addSum(net,blockNum,{inVarA,inVarF});
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,64,64],[1,1],[1,1],1,[1,0]);
[net,inVarG,blockNum]=addReLU(net,blockNum,inVar);
[net,inVar,blockNum]=addConcat(net,blockNum,{inVarE,inVarF,inVarG});
[net,inVar,blockNum]=addConv(net,blockNum,inVar,[3,3,192,1],[1,1],[1,1],1,[1,0]);

% sum
inVar = {inVar,'input'};
[net, inVar, blockNum] = addSum(net, blockNum, inVar);

outputName = 'prediction';
net.renameVar(inVar,outputName)

% loss
net.addLayer('loss', dagnn.Loss('loss','L2'), {'prediction','label'}, {'objective'},{});
net.vars(net.getVarIndex('prediction')).precious = 1;

end



     % Add a sum layer
function [net, inVar, blockNum] = addSum(net, blockNum, inVar)

outVar   = sprintf('sum%d', blockNum);
layerCur = sprintf('sum%d', blockNum);

block    = dagnn.Sum();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a relu layer
function [net, inVar, blockNum] = addReLU(net, blockNum, inVar)

outVar   = sprintf('relu%d', blockNum);
layerCur = sprintf('relu%d', blockNum);

block    = dagnn.ReLU('leak',0);
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end


% Add a bnorm layer
function [net, inVar, blockNum] = addBnorm(net, blockNum, inVar, n_ch)

trainMethod = 'adam';
outVar   = sprintf('bnorm%d', blockNum);
layerCur = sprintf('bnorm%d', blockNum);

params={[layerCur '_g'], [layerCur '_b'], [layerCur '_m']};
net.addLayer(layerCur, dagnn.BatchNorm('numChannels', n_ch), {inVar}, {outVar},params) ;

pidx = net.getParamIndex({[layerCur '_g'], [layerCur '_b'], [layerCur '_m']});
b_min                           = 0.025;
net.params(pidx(1)).value       = clipping(sqrt(2/(9*n_ch))*randn(n_ch,1,'single'),b_min);
net.params(pidx(1)).learningRate= 1;
net.params(pidx(1)).weightDecay = 0;
net.params(pidx(1)).trainMethod = trainMethod;

net.params(pidx(2)).value       = zeros(n_ch, 1, 'single');
net.params(pidx(2)).learningRate= 1;
net.params(pidx(2)).weightDecay = 0;
net.params(pidx(2)).trainMethod = trainMethod;

net.params(pidx(3)).value       = [zeros(n_ch,1,'single'), 0.01*ones(n_ch,1,'single')];
net.params(pidx(3)).learningRate= 1;
net.params(pidx(3)).weightDecay = 0;
net.params(pidx(3)).trainMethod = 'average';

inVar    = outVar;
blockNum = blockNum + 1;
end
% Add a sigmoid layer
function [net, inVar, blockNum] = addSigmoid(net, blockNum, inVar)

outVar   = sprintf('Sigmoid%d', blockNum);
layerCur = sprintf('Sigmoid%d', blockNum);

block    = dagnn.Sigmoid();
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end
% Add a swish layer
function [net, inVar, blockNum] = addSwish(net, blockNum, inVar)

outVar   = sprintf('Swish%d', blockNum);
layerCur = sprintf('Swish%d', blockNum);

block    = dagnn.Swish();
net.addLayer(layerCur, block, {inVar}, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end

% Add a Dot layer
function [net, inVar, blockNum] = addDot(net, blockNum, inVar)

outVar   = sprintf('Dot%d', blockNum);
layerCur = sprintf('Dot%d', blockNum);

block    = dagnn.Dot();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end
% add a ConvTranspose layer
function [net, inVar, blockNum] = addConvt(net, blockNum, inVar, dims, crop, upsample, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('convt%d', blockNum);

layerCur    = sprintf('convt%d', blockNum);

convBlock = dagnn.ConvTranspose('size', dims, 'crop', crop,'upsample', upsample, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f  = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*dims(4))) ; %improved Xavier
net.params(f).value        = sc*randn(dims, 'single');
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;
net.params(f).value        = zeros(dims(3), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end
% Add a pooling layer
function [net, inVar, blockNum] = addPooling(net, blockNum, inVar, method)

outVar   = sprintf('concat%d', blockNum);
layerCur = sprintf('concat%d', blockNum);
block = dagnn.Pooling('poolSize',[2,2], 'stride', 2, 'method', method);
% block = dagnn.Pooling('poolSize',[2,2], 'stride', 2);
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end
% add a Conv layer
function [net, inVar, blockNum] = addConv(net, blockNum, inVar, dims, pad, stride,dilate, lr)
opts.cudnnWorkspaceLimit = 1024*1024*1024*2; % 2GB
convOpts    = {'CudnnWorkspaceLimit', opts.cudnnWorkspaceLimit} ;
trainMethod = 'adam';

outVar      = sprintf('conv%d', blockNum);
layerCur    = sprintf('conv%d', blockNum);

convBlock   = dagnn.Conv('size', dims, 'pad', pad,'stride', stride,'dilate',dilate, ...
    'hasBias', true, 'opts', convOpts);

net.addLayer(layerCur, convBlock, {inVar}, {outVar},{[layerCur '_f'], [layerCur '_b']});

f = net.getParamIndex([layerCur '_f']) ;
sc = sqrt(2/(dims(1)*dims(2)*max(dims(3), dims(4)))) ; %improved Xavier
net.params(f).value        = sc*randn(dims, 'single') ;
net.params(f).learningRate = lr(1);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

f = net.getParamIndex([layerCur '_b']) ;

net.params(f).value        = zeros(dims(4), 1, 'single');
net.params(f).learningRate = lr(2);
net.params(f).weightDecay  = 1;
net.params(f).trainMethod  = trainMethod;

inVar    = outVar;
blockNum = blockNum + 1;
end
% Add a Concat layer
function [net, inVar, blockNum] = addConcat(net, blockNum, inVar)

outVar   = sprintf('concat%d', blockNum);
layerCur = sprintf('concat%d', blockNum);

block = dagnn.Concat('dim',3);
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar = outVar;
blockNum = blockNum + 1;
end
function [net, inVar, blockNum] = addPLM(net, blockNum, inVar)

outVar   = sprintf('plm%d', blockNum);
layerCur = sprintf('plm%d', blockNum);

block    = dagnn.PLM();
net.addLayer(layerCur, block, inVar, {outVar},{});

inVar    = outVar;
blockNum = blockNum + 1;
end

function A = clipping(A,b)
A(A>=0&A<b) = b;
A(A<0&A>-b) = -b;
end