FUNCTION="ackley"
NUM_RUNS=10

for i in $(seq 1 $NUM_RUNS)
do
    python main.py --function $FUNCTION --run-idx $i
done
