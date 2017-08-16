function [cline, label, bovfs] = cluster_light(cates, round, flag, C, centroids)
    K = size(centroids,1);
    
    cline = zeros(0,1); % cline may be helpful someday; 
    label = zeros(0,C); 
    bovfs = zeros(0,K); 
    
    % cates = {'brush_hair'};
    % splitdir = '../data';
    % stipdir = '../data'; 
    
    split_dir = '../testTrainMulti_7030_splits'; 
    stip_dir = '../hmdb51_org_stips';
    
    for j = 1 : size(cates,1)
        cate = strcat(cates(j,:),'');
        split_path = [split_dir,'/',cate,'_test_split',int2str(round),'.txt']; 
        % fprintf(split_path);
        % fprintf('\n'); 
        
        split_set = importdata(split_path); 
        vname_set = split_set.textdata;
        mask_set = split_set.data; 

        for k = 1 : size(mask_set,1)
            vname = vname_set{k,1};
            mask = mask_set(k,1); 

            if mask == flag
                stip_path = [stip_dir,'/',cate, '/', vname,'.txt']; 
                % stip_path = '../data/brushing_hair_brush_hair_f_nm_np2_ba_goo_2.avi.txt'; 
                %��fprintf(stip_path);
                % fprintf('\n');
                
                % [c,s] = read_stip_file(stip_path);
                stips_set = importdata(stip_path); 
                % stips_set = importdata('../hmdb51_org_stips/smile/show_your_smile_-)_smile_h_nm_np1_fr_med_0.avi.txt'); 
                stips = stips_set.data; 
                
                %cline
                s = zeros(1,K); 
                c = 0;
                
                %stips
                if size(stips,1) >= 0 && size(stips,2) == 169
                    c = size(stips,1);
                    
                    s = stips(:,8:end);
                    s = knnsearch(centroids, s); 
                    s = build_bow(c, s, K);
                end;
                
                % label
                l = zeros(1,C); 
                l(1,j) = 1;
                
                % need to be optimized!
                cline = [cline;c]; 
                label = [label;l];
                bovfs = [bovfs;s];
            end; 
            % break;
        end; 
        % break;
    end; 
    fprintf('%d, %d, %d',size(cline,1), size(label,1), size(bovfs,1)); 
    fprintf('\n');
end 


function [bovfs]  = build_bow(cline, xdata, nbins)
    bovfs = zeros(1, nbins); 
    hist = tabulate(xdata)'; % the upper bound of hist may not be nbins, but the maximum of xdata;
    bovfs(1,hist(1,:)) = hist(2,:); 
end 
