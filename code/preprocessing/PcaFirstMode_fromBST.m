% from bst_scout_value.m get the PCA function
function [F, explained] = PcaFirstMode_fromBST(F)
    % Remove average over time for each row
    Fmean = mean(F,2);
    F = bst_bsxfun(@minus, F, Fmean);
    % Signal decomposition
    [U,S,V] = svd(F, 'econ');
    S = diag(S);
    explained = S(1) / sum(S);
    %Find where the first component projects the most over original dimensions
    [tmp__, nmax] = max(abs(U(:,1))); 
    % What's the sign of absolute max amplitude along this dimension?
    [tmp__, i_omaxx] = max(abs(F(nmax,:)));
    sign_omaxx = sign(F(nmax,i_omaxx));
    % Sign of maximum in first component time series
    [Vmaxx, i_Vmaxx] = max(abs(V(:,1)));
    sign_Vmaxx = sign(V(i_Vmaxx,1));
    % Reconcile signs
    F = sign_Vmaxx * sign_omaxx * S(1) * V(:,1)';
    F = F + Fmean(nmax);
end