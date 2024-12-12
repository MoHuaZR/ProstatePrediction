

function [acc_grid res_est res_acc] = mkl(featureA, featureB, featureC, label)




KFOLD=20;
sigma = 1;  %%%%0.001 100
acc_fold = zeros(KFOLD, 1);
acc_i = zeros(10, 1);
acc_grid = zeros(11, 11);
for row = 0:10
    for col = 0:10 - row
        c1 = row * 0.1;
        c2 = col * 0.1;
        c3 = 1 - c1 - c2;
        
        
        for fold = 1:length(label)
            
            
            
            tr_label = setdiff(1:length(label),fold);
            %% featureselection
            featureC_train=featureC(tr_label,:);
            featureA_train=featureA(tr_label,:);
            featureB_train=featureB(tr_label,:);
            Label_train=label(tr_label);
            [~ ,CI]=ttest2(featureC_train(Label_train==2,:),featureC_train(Label_train==1,:));
            [~ ,MI]=ttest2(featureA_train(Label_train==2,:),featureA_train(Label_train==1,:));
            [~ ,PI]=ttest2(featureB_train(Label_train==2,:),featureB_train(Label_train==1,:));

% 

           Funf = featureA(:,MI<0.01); % 
            GMf = featureC(:,CI<0.01); % 
            DTIf = featureB(:,PI<0.01);%
% 
% 

            
            
          %  Funf = featureA(:,MI<0.01); % 
%             GMf = featureC(:,CI<0.01); % 
%             DTIf = featureB(:,PI<0.01); %
% 
%             Funf = bsxfun(@rdivide, bsxfun(@minus, Funf, mean(Funf)), std(Funf));
%             % 行规整化
%             Funf = bsxfun(@rdivide, Funf, sqrt(sum(Funf.^2, 2)));
%             
%             GMf = bsxfun(@rdivide, bsxfun(@minus, GMf, mean(GMf)), std(GMf));
%             GMf = bsxfun(@rdivide, GMf, sqrt(sum(GMf.^2, 2)));
%             DTIf = bsxfun(@rdivide, bsxfun(@minus, DTIf, mean(DTIf)), std(DTIf));
%             DTIf = bsxfun(@rdivide, DTIf, sqrt(sum(DTIf.^2, 2)));
            te_label = fold;
            
            
            Y = label(tr_label);
            Y(Y == 2) = -1;
            Yt = label(te_label);
            Yt(Yt == 2) = -1;
            
            featureAK = calckernel('linear', sigma, Funf(tr_label, :));
            featureAKT = calckernel('linear', sigma, Funf(tr_label, :), Funf(te_label, :));
            
            featureCK = calckernel('linear', sigma, GMf(tr_label, :));
            featureCKT = calckernel('linear', sigma, GMf(tr_label, :), GMf(te_label, :));
            
            featureBK = calckernel('linear', sigma, DTIf(tr_label, :));
            featureBKT = calckernel('linear', sigma, DTIf(tr_label, :), DTIf(te_label, :));
            
            K =double( c1 * featureAK + c2 * featureCK + c3 * featureBK);
            KT = double(c1 * featureAKT + c2 * featureCKT + c3 * featureBKT);
            
            model = svmtrain(double(Y)', [(1:length(Y))', K], '-t 4');
            [predicted_label, accuracy, est] = svmpredict(double(Yt)', [(1:length(Yt))', KT], model);
            acc_fold(fold) = accuracy(1);
            estimates(row+1,col+1,fold)=est;
            acc_result(row+1,col+1,fold)=accuracy(1);
            prel(fold)=predicted_label;
        end
        
        acc_grid(row + 1, col + 1) = mean(acc_fold);
    end
end
res_est=estimates;
res_acc=acc_result;
end

