for SETTING in 1,3 2,5 3,7, 3,8 4,7 4,8
do

IFS=","; set -- $SETTING;

echo "pure child"
python main.py --num_hidden $1 --num_observed $2 --mode purechild --nonlinear

echo "single source"
python main.py --num_hidden $1 --num_observed $2 --mode singlesource --nonlinear

done