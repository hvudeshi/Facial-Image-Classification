close all; clc; clear all;

% train and test images directory; change names when porting code to
% another machine
train=dir('resize_train');
test=dir('resize_test'); % test=dir('test_60');

test=test(3:numel(test));
train=train(3:numel(train));

no_test=numel(test); % to not count . & .. links
no_train=numel(train); % to not count . & .. links

% -------
labelForTrain=zeros(no_train,1);

for j=1:no_train
    labelForTrain(j)=str2double(strtok(train(j).name,'_'));
end
% -------

%%% precision-recall
classLabels=unique(labelForTrain);
confusionMatrix=zeros(numel(classLabels));
%%%

minEuclidean=zeros(no_test,1);
indexMatch=zeros(no_test,1);
labelForTest=zeros(no_test,1);

scores_individual=zeros(no_test,no_train);

for i=1:no_test
    I = imread(test(i).name); 
    
    if ndims(I)>2
        I = rgb2gray(I);
    end
    
    I = single(I); % Conversion to single is recommendedx
    test(i).name 
    [F1, D1] = vl_sift(I);
    
    for j=1:no_train    
        fprintf('i=%d ; j=%d\n', i,j);
        J = imread(train(j).name);
        
        if ndims(J)>2
            J = rgb2gray(J);
        end
%         J = rgb2gray(J);
        J = single(J); % in the documentation

%         [F1, D1] = vl_sift(I);
        [F2, D2] = vl_sift(J);

        % Where 1.5 = ratio between euclidean distance of NN2/NN1
        % returns squared euclidean distance between D1 & D2
        [matches, score] = vl_ubcmatch(D1,D2,1.5);

        scores_individual(i,j)=(sum(score)/size(score,2));
    end
    
    [minEuclidean(i), indexMatch(i)] = min(scores_individual(i,1:no_train));
    labelForTest(i)=labelForTrain(indexMatch(i)); % -------
    
%%% precision-recall
    originalLabel=str2double(strtok(test(i).name,'_'));
    col=find(classLabels==originalLabel);
    row=find(classLabels==labelForTest(i));
    confusionMatrix(row,col)=confusionMatrix(row,col)+1;
%%%
end

%%% precision-recall

finalConfusionMatrix=[0,classLabels';classLabels,confusionMatrix]
precision=zeros(numel(classLabels),1);
recall=zeros(numel(classLabels),1);
AvgPrecision=0;
for i=1:numel(classLabels)
    precision(i)=confusionMatrix(i,i)/sum(confusionMatrix(i,:)');
    recall(i)=confusionMatrix(i,i)/sum(confusionMatrix(:,i));
end
% print=[classLabels,precision,recall]

TotalDetected = 0;

for i=1:size(classLabels,1)
    TotalDetected = TotalDetected + confusionMatrix(i,i);
end
Accuracy = TotalDetected/no_test
% print=Accuracy
%%%