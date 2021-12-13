function [result] = VarianceZ(pointKnn,k)

mz = mean(pointKnn(:,3));

for i=1:k
    C1(i,3) = pointKnn(i,3)-mz;
end
result = sum(C1(:,3))/k;
end