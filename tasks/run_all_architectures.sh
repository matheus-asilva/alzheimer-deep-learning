set PYTHONPATH=. &&
wandb login 825e234f13b80286db418be6ccf93e748593dce7 &&
# for net in mobilenet vgg16 vgg19 xception resnet152v2 resnet50v2
for net in full_mobilenet
do
    echo "###################################################################################################################################################"
    echo "Training $net..."
    python training\\run_experiment.py --save "{\"dataset\": \"AlzheimerT2StarSmallDataset\", \"dataset_args\": {\"types\": [\"CN\", \"AD\"]}, \"model\": \"AlzheimerCNN\", \"network\": \"full_mobilenet\", \"train_args\": {\"batch_size\": 32, \"epochs\": 50}, \"opt_args\": {\"learning_rate\": 1e-4, \"decay\": 1e-5}}"
    echo " "
done