TRAIN='/home/pataki/patho_scientificdata/modeling/fold1_train.csv'
TEST='/home/pataki/patho_scientificdata/modeling/fold1_test.csv'
LOG='/home/pataki/patho_scientificdata/modeling/fold1/'
mkdir -p $LOG
nohup python3 train.py --GPU_ID 1 --train_meta_file $TRAIN --test_meta_file $TEST --log_dir $LOG >> $LOG/logs.txt 2>&1 &

TRAIN='/home/pataki/patho_scientificdata/modeling/fold2_train.csv'
TEST='/home/pataki/patho_scientificdata/modeling/fold2_test.csv'
LOG='/home/pataki/patho_scientificdata/modeling/fold2/'
mkdir -p $LOG
nohup python3 train.py --GPU_ID 0 --train_meta_file $TRAIN --test_meta_file $TEST --log_dir $LOG >> $LOG/logs.txt 2>&1 &

TRAIN='/home/pataki/patho_scientificdata/modeling/fold3_train.csv'
TEST='/home/pataki/patho_scientificdata/modeling/fold3_test.csv'
LOG='/home/pataki/patho_scientificdata/modeling/fold3/'
mkdir -p $LOG
nohup python3 train.py --GPU_ID 2 --train_meta_file $TRAIN --test_meta_file $TEST --log_dir $LOG >> $LOG/logs.txt 2>&1 &

TRAIN='/home/pataki/patho_scientificdata/modeling/fold4_train.csv'
TEST='/home/pataki/patho_scientificdata/modeling/fold4_test.csv'
LOG='/home/pataki/patho_scientificdata/modeling/fold4/'
mkdir -p $LOG
nohup python3 train.py --GPU_ID 0 --train_meta_file $TRAIN --test_meta_file $TEST --log_dir $LOG >> $LOG/logs.txt 2>&1 &

TRAIN='/home/pataki/patho_scientificdata/modeling/fold5_train.csv'
TEST='/home/pataki/patho_scientificdata/modeling/fold5_test.csv'
LOG='/home/pataki/patho_scientificdata/modeling/fold5/'
mkdir -p $LOG
nohup python3 train.py --GPU_ID 1 --train_meta_file $TRAIN --test_meta_file $TEST --log_dir $LOG >> $LOG/logs.txt 2>&1 &
