set PYTHONPATH=. &&
wandb login 825e234f13b80286db418be6ccf93e748593dce7 &&
# já foi mobilenet
for net in vgg16 vgg19 xception resnet152v2 resnet50v2
do
    echo "############################################################################################"
    echo "Training $net..."
    python training\\run_experiment.py --save "{\"dataset\": \"AlzheimerT2StarSmallDataset\", \"dataset_args\": {\"types\": [\"CN\", \"AD\"]}, \"model\": \"AlzheimerCNN\", \"network\": \"$net\", \"train_args\": {\"batch_size\": 16, \"epochs\": 100}, \"opt_args\": {\"lr\": 1e-3, \"decay\": 1e-5}}"
    echo " "
done