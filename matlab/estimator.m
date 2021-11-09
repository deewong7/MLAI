function my_var = estimator(array)
    u = mean(array);
    s = 0;
    for i = array
        s = s + ( (i - u)^2 );
    end
    my_var = s / (length(array) - 1);
end