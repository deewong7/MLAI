function my_var = impurity_variance(x)
    my_var = sum( (x - mean(x)) .^ 2) / ( length(x) - 1 );
end
