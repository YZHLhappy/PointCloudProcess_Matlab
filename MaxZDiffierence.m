function [result] = MaxZDiffierence(pointKnn)

result = max(pointKnn(:,3)) - min(pointKnn(:,3));
end