# Dog breed classifier (CNN)
Creating a CNN from scratch and then using ResNet50 (transfer learning) with PyTorch to classify dog breeds. If given dog image, it would identify the dog breed. But if given human image, it would tell what dog breed it resembles the most. This makes this project really interesting.

Preprocessing: The dataset was augmented by the torchvision.transforms techniques like RandomHorizontalFlip, RandomRotation.Image file was finally converted to tensor and normalized.

## CNN from scratch using PyTorch
3 Convulational layer and 2 linear layer CNN was designed. Relu activation function used with pooling of 2x. 1st layer gets input image of 224 * 224 * 3 dimensions and has kernel size of (3,3), and 32 filtered images as output. ReLu activation function is used and pooling would downsample dimension to 112 * 112 * 32. 2nd layer would take 32 images, with output of 64. Relu function is used and pooling would downsize dim to 56 * 56 * 64. 3rd layer would take 64 as input and 128 as output, and relu function and pooling would downsize dimension to 28 * 28 * 128. This would be input for fully connected layer 1 with output as 500 classes. 2nd fully connected layer would taske these 500 as inputs and would give 133 classes as output using relu function.

Accuracy: 11% after 50 epochs.
When ResNet50 transfer learning was used, accuracy increased to 70%. 
Pretrained models has right weights and so, it leads to good results.

## Three possible areas for improvement:
1. More datapoints for each dog class would improve accuracy
2. Better transfer learning models with more layers
3. More epochs would increase score but compuationally expensive
