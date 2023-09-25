screen -dmS train
sleep 1
screen -r train -X stuff $'conda activate yolov5\n'
screen -r train -X stuff $'python train.py\n'

