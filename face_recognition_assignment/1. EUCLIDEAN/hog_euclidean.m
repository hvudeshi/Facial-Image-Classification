clear all; close all; clc;

train=dir('train_60');
test=dir('test_60');
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

minEuclidean=zeros(no_test,1);
indexMatch=zeros(no_test,1);
labelForTest=zeros(no_test,1);

scores_individual=zeros(no_test,no_train);

for i=1:no_test
    I = imread(test(i).name);
    
    I = rgb2gray(I);
    I = single(I); % Conversion to single is recommendedx
    test(i).name 
    Ftest = extractHOGFeatures(I);
    
    for j=1:no_train
        
        fprintf('i=%d ; j=%d\n', i,j);
       
        J = imread(train(j).name);
        J = rgb2gray(J);
        J = single(J); % in the documentation

        
        Ftrain = extractHOGFeatures(J);

        testCol = size(Ftest,2);
        trainCol = size(Ftrain,2);
        
        if(testCol > trainCol)
            Ftrain = [Ftrain zeros(1, (testCol - trainCol))];
        else
            Ftest = [Ftest zeros(1, (trainCol - testCol))];
        end
        
        % find euclidean distance
        scores_individual(i,j)=sqrt(sum((Ftest - Ftrain).^2));
    end
    
    [minEuclidean(i), indexMatch(i)] = min(scores_individual(i,3:no_train));
    %indexMatch=indexMatch;
    
%     labelForTest(i)=str2double(strtok(train(indexMatch(i)).name,'_'));
    labelForTest(i)=labelForTrain(indexMatch(i)); % -------
    
%%% precision-recall    
     originalLabel=str2double(strtok(test(i).name,'_'));
    col=find(classLabels==originalLabel);
    row=find(classLabels==labelForTest(i));
    confusionMatrix(row,col)=confusionMatrix(row,col)+1;
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