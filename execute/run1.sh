for method in fedavg fedprox feddyn fedrs
do
    for alpha in 0.0 0.01 0.1 1.0 100.0
    do
        for p in target untarget
        do
            python main.py --method $method --tsboard --c_frac 0.3 --alpha $alpha --p $p --quantity_skew --plugin
        done
    done
done