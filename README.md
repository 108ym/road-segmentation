
## Guide to use
  - Download the files
  - Python 3.10.14, pip install -r requirements.txt to install all the required dependencies.
  - Open Config.py and adjust it according to your setup and adjust related file paths in it.
  - To start training run U-Net/train.py
  - To test model run U-Net/test.py (test on 5 images), the output in 'results' directory
  - To see segmentation results for all 31 images, checkout 'results_all' directory. 
    You can also run and get the same results by changing path in test.py. See comments in test.py #to evaluate for all images...

## Dataset
The dataset has 21 Training images, 5 Validation images and 5 Test images

## Training Setup
Following configurations were used for final model training.
  - Batch Size: 1
  - Learning rate: 0.0008
  - Optimizer: SGD
  - Loss: Focal Loss
  - Metric: IoU

## Directories guide
 -Dataset used consist of input images(train, val, test) and corresponding masks(train_labels, val_labels, test_labels). Complete dataset can be found in 'all' and 'all_labels'
 -Best segmentation test result in 'results', segmentation result for complete dataset in 'results_all'
 -results140 & model/state_dict140.pt (using lr=0.0001 and epoch = 40), results150 & model/state_dict150.pt (using lr=0.0001 and epoch = 50), results840 & model/state_dict840.pt (using lr=0.000 and epoch = 40)

