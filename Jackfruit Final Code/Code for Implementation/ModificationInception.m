%% Modification Trial 1

close all
clear
clc


%% READ TRAINING IMAGES
allImages = imageDatastore('./patchDataset/PatchDatas','IncludeSubfolders',true,'LabelSource','foldernames');
% training_set = imageDatastore('./images/Training_Data','IncludeSubfolders',true,'LabelSource','foldernames');
% validation_set = imageDatastore('./images/Validation_Data','IncludeSubfolders',true,'LabelSource','foldernames');
% testing_set = imageDatastore('./images/Testing_Data','IncludeSubfolders',true,'LabelSource','foldernames');
[training_set, validation_set, testing_set] = splitEachLabel(allImages,.4,.3,.3);
training_set.ReadFcn = @readPathoImage_224; % read and resize images to 229
validation_set.ReadFcn = @readPathoImage_224;
testing_set.ReadFcn = @readPathoImage_224;
%% Modification 1:PREPARE AND MODIFY NEURAL NET
net = googlenet('Weights','places365');
layersTransfer = net.Layers(1:end-3);
categories(training_set.Labels)
numClasses = numel(categories(training_set.Labels));
lgraph = layerGraph(net);
fcLayer = fullyConnectedLayer(numClasses,'WeightLearnRateFactor',10,'BiasLearnRateFactor',10,'Name', 'FC_Layer');
DropLayer=dropoutLayer(0.7,'Name','NewDropoutLayer');
lgraph = replaceLayer(lgraph,'pool5-drop_7x7_s1',DropLayer);
clsLayer = classificationLayer('Name', 'OutputLayer');
lgraphNew = replaceLayer(lgraph,"loss3-classifier",fcLayer);
lgraphNew = replaceLayer(lgraphNew,"output",clsLayer);

%% Modification 2: Batch Normalization layer
larray = [batchNormalizationLayer('Name','BN1')
            leakyReluLayer('Name','leakyRelu_1','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'conv2-relu_3x3',larray);
larray1 = [batchNormalizationLayer('Name','BN2')
            leakyReluLayer('Name','leakyRelu_2','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_3a-relu_1x1',larray1);
larray2 = [batchNormalizationLayer('Name','BN3')
            leakyReluLayer('Name','leakyRelu_3','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_3a-relu_3x3',larray2);
larray3 = [batchNormalizationLayer('Name','BN31')
            leakyReluLayer('Name','leakyRelu_4','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_3a-relu_5x5',larray3);
larray4 = [batchNormalizationLayer('Name','BN4')
            leakyReluLayer('Name','leakyRelu_5','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_3a-relu_pool_proj',larray4);
larray5 = [batchNormalizationLayer('Name','BN5')
            leakyReluLayer('Name','leakyRelu_6','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_3b-relu_1x1',larray5);
larray6 = [batchNormalizationLayer('Name','BN6')
            leakyReluLayer('Name','leakyRelu_7','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_3b-relu_3x3',larray6);
larray7 = [batchNormalizationLayer('Name','BN7')
            leakyReluLayer('Name','leakyRelu_8','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_3b-relu_5x5',larray7);
larray8 = [batchNormalizationLayer('Name','BN8')
            leakyReluLayer('Name','leakyRelu_9','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_3b-relu_pool_proj',larray8);
larray9 = [batchNormalizationLayer('Name','BN9')
            leakyReluLayer('Name','leakyRelu_10','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4a-relu_1x1',larray9);
larray10 = [batchNormalizationLayer('Name','BN10')
            leakyReluLayer('Name','leakyRelu_11','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4a-relu_3x3',larray10);
larray11 = [batchNormalizationLayer('Name','BN11')
            leakyReluLayer('Name','leakyRelu_11p5','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4a-relu_5x5',larray11);
larray12 = [batchNormalizationLayer('Name','BN12')
            leakyReluLayer('Name','leakyRelu_12','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4a-relu_pool_proj',larray12);
larray13 = [batchNormalizationLayer('Name','BN13')
            leakyReluLayer('Name','leakyRelu_13','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4b-relu_1x1',larray13);
larray14 = [batchNormalizationLayer('Name','BN14')
            leakyReluLayer('Name','leakyRelu_14','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4b-relu_3x3',larray14);
larray15 = [batchNormalizationLayer('Name','BN15')
            leakyReluLayer('Name','leakyRelu_14p5','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4b-relu_5x5',larray15);
larray16 = [batchNormalizationLayer('Name','BN16')
            leakyReluLayer('Name','leakyRelu_15','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4b-relu_pool_proj',larray16);
larray17 = [batchNormalizationLayer('Name','BN17')
            leakyReluLayer('Name','leakyRelu_16','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4c-relu_1x1',larray17);
larray18 = [batchNormalizationLayer('Name','BN18')
            leakyReluLayer('Name','leakyRelu_17','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4c-relu_3x3',larray18);
larray19 = [batchNormalizationLayer('Name','BN19')
            leakyReluLayer('Name','leakyRelu_18','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4c-relu_5x5',larray19);
larray20 = [batchNormalizationLayer('Name','BN20')
            leakyReluLayer('Name','leakyRelu_19','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4c-relu_pool_proj',larray20);
larray21 = [batchNormalizationLayer('Name','BN21')
            leakyReluLayer('Name','leakyRelu_20','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4d-relu_1x1',larray21);
larray22 = [batchNormalizationLayer('Name','BN22')
            leakyReluLayer('Name','leakyRelu_22','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4d-relu_3x3',larray22);
larray23 = [batchNormalizationLayer('Name','BN23')
            leakyReluLayer('Name','leakyRelu_23','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4d-relu_5x5',larray23);
larray24 = [batchNormalizationLayer('Name','BN24')
            leakyReluLayer('Name','leakyRelu_24','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4d-relu_pool_proj',larray24);
larray25 = [batchNormalizationLayer('Name','BN25')
            leakyReluLayer('Name','leakyRelu_25','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4e-relu_1x1',larray25);
larray26 = [batchNormalizationLayer('Name','BN26')
            leakyReluLayer('Name','leakyRelu_26','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4e-relu_3x3',larray26);
larray27 = [batchNormalizationLayer('Name','BN27')
            leakyReluLayer('Name','leakyRelu_27','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4e-relu_5x5',larray27);
larray28 = [batchNormalizationLayer('Name','BN28')
            leakyReluLayer('Name','leakyRelu_28','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_4e-relu_pool_proj',larray28);
larray29 = [batchNormalizationLayer('Name','BN29')
            leakyReluLayer('Name','leakyRelu_29','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_5a-relu_1x1',larray29);
larray30 = [batchNormalizationLayer('Name','BN30')
            leakyReluLayer('Name','leakyRelu_30','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_5a-relu_3x3',larray30);
larray31 = [batchNormalizationLayer('Name','BN31p')
            leakyReluLayer('Name','leakyRelu_31','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_5a-relu_5x5',larray31);
larray32 = [batchNormalizationLayer('Name','BN32')
            leakyReluLayer('Name','leakyRelu_32','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_5a-relu_pool_proj',larray32);
larray33 = [batchNormalizationLayer('Name','BN33')
            leakyReluLayer('Name','leakyRelu_33','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_5b-relu_1x1',larray33);
larray34 = [batchNormalizationLayer('Name','BN34')
            leakyReluLayer('Name','leakyRelu_34','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_5b-relu_3x3',larray34);
larray35 = [batchNormalizationLayer('Name','BN35')
            leakyReluLayer('Name','leakyRelu_35','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_5b-relu_5x5',larray35);
larray36 = [batchNormalizationLayer('Name','BN36')
            leakyReluLayer('Name','leakyRelu_36','Scale',0.9)];
lgraphNew = replaceLayer(lgraphNew,'inception_5b-relu_pool_proj',larray36);
analyzeNetwork(lgraphNew)
plot(lgraphNew)
imageInputSize = net.Layers(1).InputSize;

%% Modification : 3 DATA AUGMENTATION FOR TRAINING
pixelRange = [-30 30];
imageAugmenter = imageDataAugmenter('RandRotation',[-20,20],'RandXReflection',true,'RandYReflection',true, ...
    'RandXTranslation',pixelRange,'RandYTranslation',pixelRange);
augmented_training_set = augmentedImageDatastore(imageInputSize,training_set,'DataAugmentation',imageAugmenter);
resized_validation_set = augmentedImageDatastore(imageInputSize,validation_set,'DataAugmentation',imageAugmenter);
resized_testing_set = augmentedImageDatastore(imageInputSize,testing_set,'DataAugmentation',imageAugmenter);
%% Modification : 4 Validation Frequency
miniBatchSize=32;
valFrequency = floor(numel(training_set.Files)/(miniBatchSize));

%% Modification : 5 TRAIN
opts = trainingOptions('sgdm', ...
    'MiniBatchSize', miniBatchSize,... % mini batch size, limited by GPU RAM, default 100 on Titan, 500 on P6000
    'InitialLearnRate', 1e-3,... % fixed learning rate
    'L2Regularization', 1e-5,... % optimization L2 constraint
    'MaxEpochs',15,... % max. epochs for training, default 3
    'GradientThresholdMethod','absolute-value',...
    'LearnRateDropFactor',0.2,...
    'LearnRateDropPeriod',5,...
    'Verbose',true,...
    'ValidationFrequency',valFrequency, ...
    'ExecutionEnvironment', 'auto',...% environment for training and classification, use a compatible GPU
    'ValidationData', validation_set,...
    'Plots', 'training-progress')
net = trainNetwork(augmented_training_set, lgraphNew, opts)
%% TESTING Process
[predLabels,predScores] = classify(net, testing_set, 'ExecutionEnvironment','auto');
figure,plotconfusion(testing_set.Labels, predLabels)
PerItemAccuracy = mean(predLabels == testing_set.Labels);
title(['overall per image accuracy ',num2str(round(100*PerItemAccuracy)),'%'])
%% -----------Code for ROC -----------------
cgt = double(testing_set.Labels); 
[X,Y,T,AUC,OPTROCPT,SUBY,SUBYNAMES] = perfcurve(cgt,predScores(:,1),1);
figure,plot(X,Y,'LineWidth',3);
xlabel('False positive rate')
ylabel('True positive rate')
title('ROC for Classification ')