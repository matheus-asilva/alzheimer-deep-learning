set PYTHONPATH=. &&
wandb login 825e234f13b80286db418be6ccf93e748593dce7 &&
for net in vgg16 vgg19 xception resnet152v2 resnet50v2 mobilenet
do
    echo "############################################################################################"
    echo "Training $net..."
    python training\\run_experiment.py --save "{\"dataset\": \"AlzheimerT2SmallDataset\", \"dataset_args\": {\"types\": [\"CN\", \"MCI\", \"AD\"]}, \"model\": \"AlzheimerCNN\", \"network\": \"$net\"}"
    echo " "
done