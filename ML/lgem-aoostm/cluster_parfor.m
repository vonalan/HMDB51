function [x_train, x_testa] = cluster_parfor(cline_train, stips_train, cline_testa, stips_testa, centroids)
    % global variables is not available in variable scope of subprocess;
    % global cline_train stips_train cline_testa stips_testa;
    
    K = size(centroids,1);
    
    idx_train = knnsearch(centroids, stips_train);
    idx_testa = knnsearch(centroids, stips_testa);
    
    x_train = build_hist(cline_train, idx_train, K);
    x_testa = build_hist(cline_testa, idx_testa, K);
end 


function [bovfs]  = build_hist(cline, xdata, nbins)
    bovfs = zeros(size(cline, 1), nbins); 
    
    cbegin = 0; 
    for i = 1:size(cline, 1)
        c = cline(i,1);
        if c~=0
            cend = cbegin + cline(i,1); 
            % x = xdata(cbegin+1:cend,:);
            % [i,c]
            hist = tabulate(xdata(cbegin+1:cend,:))'; 
            bovfs(i,hist(1,:)) = hist(2,:); 
            cbegin = cend;
        % else
        %     [i,c]
        end;
    end;
end