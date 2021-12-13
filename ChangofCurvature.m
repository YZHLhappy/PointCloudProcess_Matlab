function [result] = ChangofCurvature(pointKnn,k)
mx = mean(pointKnn(:,1));
my = mean(pointKnn(:,2));
mz = mean(pointKnn(:,3));

for i=1:k
    C1(i,1) = pointKnn(i,1)-mx;
    C1(i,2) = pointKnn(i,2)-my;
    C1(i,3) = pointKnn(i,3)-mz;
end
C=C1'*C1;
[~,D] = eig(C);
result = D(3,3)/(D(1,1)+D(2,2)+D(3,3));
end