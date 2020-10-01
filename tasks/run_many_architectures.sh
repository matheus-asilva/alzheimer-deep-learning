export PYTHONPATH=. &&
wandb login 825e234f13b80286db418be6ccf93e748593dce7 &&
# for net in mobilenet vgg16 vgg19 xception resnet152v2 resnet50v2
for net in full_mobilenet
do
    echo "Training $net..."
    python training/run_experiment.py --nowandb --save "{\"dataset\": \"AlzheimerMPRageNoDeep\", \"dataset_args\": {\"types\": [\"CN\", \"AD\"]}, \"model\": \"AlzheimerCNN\", \"network\": \"$net\", \"train_args\": {\"batch_size\": 4, \"epochs\": 100}, \"opt_args\": {\"learning_rate\": 1e-3}}"
    echo " "
done

sleep 5
