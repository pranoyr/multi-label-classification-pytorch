import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
import torch.nn.functional as F

# chanDim = -1
# model = Sequential()
# model.add(Conv2D(32, (3, 3), padding="same", input_shape=(50,50,3)))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(3, 3)))
# model.add(Dropout(0.25))

# # (CONV => RELU) * 2 => POOL
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(Conv2D(64, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# # (CONV => RELU) * 2 => POOL
# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(Conv2D(128, (3, 3), padding="same"))
# model.add(Activation("relu"))
# model.add(BatchNormalization(axis=chanDim))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))

# # first (and only) set of FC => RELU layers
# model.add(Flatten())
# model.add(Dense(1024))
# model.add(Activation("relu"))
# model.add(BatchNormalization())
# model.add(Dropout(0.5))
# # use a *softmax* activation for single-label classification
# # and *sigmoid* activation for multi-label classification
# model.add(Dense(params['n_classes']))
# model.add(Activation('sigmoid'))


class MultiLabelModel(nn.Module):
    def __init__(self, num_classes=25):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(MultiLabelModel, self).__init__()

        # chanDim = -1
        # model = Sequential()
        # model.add(Conv2D(32, (3, 3), padding="same", input_shape=(50,50,3)))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(3, 3)))
        # model.add(Dropout(0.25))
        self.conv1 = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),#padding 5
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(32),
        nn.MaxPool2d(kernel_size=3, stride=1),
        nn.Dropout(0.25))



        # # (CONV => RELU) * 2 => POOL
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(64, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        self.conv2 = nn.Sequential(
        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),#padding 5,
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),#padding 5
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(kernel_size=2, stride=1),
        nn.Dropout(0.25))



        # # (CONV => RELU) * 2 => POOL
        # model.add(Conv2D(128, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(Conv2D(128, (3, 3), padding="same"))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization(axis=chanDim))
        # model.add(MaxPooling2D(pool_size=(2, 2)))
        # model.add(Dropout(0.25))
        self.conv3 = nn.Sequential(
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),#padding 5,
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),#padding 5
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(kernel_size=2, stride=1),
        nn.Dropout(0.25))


        # # first (and only) set of FC => RELU layers
        # model.add(Flatten())
        # model.add(Dense(1024))
        # model.add(Activation("relu"))
        # model.add(BatchNormalization())
        # model.add(Dropout(0.5))
        # # use a *softmax* activation for single-label classification
        # # and *sigmoid* activation for multi-label classification
        # model.add(Dense(params['n_classes']))
        # model.add(Activation('sigmoid'))

        self.fc = nn.Sequential(
        nn.Linear(128 * 96 * 96, 1024),
        nn.ReLU(inplace=True),
        nn.BatchNorm1d(1024),
        nn.Dropout(0.5),
        nn.Linear(1024, 25))

        #self.fc = nn.Linear(128 * 96 * 96, 1024)

    
    def forward(self, x):
       
        x = self.conv1(x) 
        x = self.conv2(x) 
        x = self.conv3(x)
        x = x.view(-1, 128 * 96 * 96)
        x = self.fc(x)
        return x

if (__name__=='__main__'):
    net = MultiLabelModel()
    x  = torch.randn(32, 3,64,64)
    output = net(x)
    print(output.size())