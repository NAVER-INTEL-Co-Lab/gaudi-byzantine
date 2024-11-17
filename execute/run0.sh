for method in fedavg fedprox feddyn fedrs krum trimmed_mean fang
do
    python main.py --method $method --tsboard --c_frac 0.0 --quantity_skew
done

for method in fedavg fedprox feddyn fedrs krum trimmed_mean fang
do
    for frac in 0.1 0.2 0.3
    do
        for p in target untarget
        do
            python main.py --method $method --tsboard --c_frac $frac --p $p --quantity_skew
        done
    done
done

for method in fedavg fedprox feddyn fedrs
do
    for frac in 0.1 0.2 0.3
    do
        for p in target untarget
        do
            python main.py --method $method --tsboard --c_frac $frac --p $p --quantity_skew --plugin
        done
    done
done