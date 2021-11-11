function gini_index = gini(array)
    
    psquares = 0;
    for each = array
        psquares = psquares + each ^ 2;
    end
    gini_index = 1 - psquares;
end