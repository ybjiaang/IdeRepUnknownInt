for SETTING in 1,3 2,5 3,7, 3,8 4,7 4,8
do

IFS=","; set -- $SETTING;

echo "pure child"
python synthetic.py --num_hidden $1 --num_observed $2 --mode purechild

echo "single source"
python synthetic.py --num_hidden $1 --num_observed $2 --mode singlesource

done