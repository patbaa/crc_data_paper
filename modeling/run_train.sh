ZOOM_LEVEL=2

TRAIN="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold1_train.csv"
TEST="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold1_test.csv"
LOG="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold1/"
mkdir -p $LOG
nohup python3 train.py --GPU_ID 2 --train_meta_file $TRAIN --test_meta_file $TEST --log_dir $LOG >> $LOG/logs.txt 2>&1 &

TRAIN="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold2_train.csv"
TEST="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold2_test.csv"
LOG="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold2/"
mkdir -p $LOG
nohup python3 train.py --GPU_ID 0 --train_meta_file $TRAIN --test_meta_file $TEST --log_dir $LOG >> $LOG/logs.txt 2>&1 &

TRAIN="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold3_train.csv"
TEST="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold3_test.csv"
LOG="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold3/"
mkdir -p $LOG
nohup python3 train.py --GPU_ID 1 --train_meta_file $TRAIN --test_meta_file $TEST --log_dir $LOG >> $LOG/logs.txt 2>&1 &

TRAIN="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold4_train.csv"
TEST="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold4_test.csv"
LOG="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold4/"
mkdir -p $LOG
nohup python3 train.py --GPU_ID 0 --train_meta_file $TRAIN --test_meta_file $TEST --log_dir $LOG >> $LOG/logs.txt 2>&1 &

TRAIN="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold5_train.csv"
TEST="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold5_test.csv"
LOG="/home/pataki/crc_data_paper/modeling/zoom${ZOOM_LEVEL}/fold5/"
mkdir -p $LOG
nohup python3 train.py --GPU_ID 1 --train_meta_file $TRAIN --test_meta_file $TEST --log_dir $LOG >> $LOG/logs.txt 2>&1 &
