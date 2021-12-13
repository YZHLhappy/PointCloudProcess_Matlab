function [V1,V2,V3] = normalVector(pointKnn,k)
mx = mean(pointKnn(:,1));
my = mean(pointKnn(:,2));
mz = mean(pointKnn(:,3));

for i=1:k
    C1(i,1) = pointKnn(i,1)-mx;
    C1(i,2) = pointKnn(i,2)-my;
    C1(i,3) = pointKnn(i,3)-mz;
end
C=C1'*C1;
[V,D] = eig(C);
[~,idx] = min(diag(D));
% Normalize
V = V(:,idx)./norm(V(:,idx));
V1 = V(1);
V2 = V(2);
V3 = V(3);
end
