import mindspore
from mindspore import Tensor
from mindspore import Parameter
from mindspore.common import dtype as mstype
from mindspore import nn
from mindspore import ops
from mindspore.ops import operations as P
from mindspore.ops import functional as F
from mindspore.ops import composite as C
from mindspore import compression
from mindspore import context
from mindspore.common.initializer import initializer
from mindspore.common.initializer import Normal
from mindspore.train.model import Model
from mindspore.nn import Momentum
from mindspore.nn import learning_rate_schedule
import numpy as np
import pytest
import os

class Net(nn.Cell):
    # define the operator required
    def __init__(self, num_class=10, num_channel=1):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(num_channel, 6, 5, pad_mode='valid')
        self.conv2 = nn.Conv2d(6, 16, 5, pad_mode='valid')
        self.fc1 = nn.Dense(16 * 5 * 5, 120, weight_init='ones')
        self.fc2 = nn.Dense(120, 84, weight_init='ones')
        self.fc3 = nn.Dense(84, num_class, weight_init='ones')
        self.relu = nn.ReLU()
        self.max_pool2d = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()

    # use the preceding operators to construct networks
    def construct(self, x):
        x = self.max_pool2d(self.relu(self.conv1(x)))
        x = self.max_pool2d(self.relu(self.conv2(x)))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


@pytest.fixture(autouse=True)
def add_um(doctest_namespace):
    doctest_namespace["P"] = P
    doctest_namespace["ops"] = ops
    doctest_namespace["Parameter"] = Parameter
    doctest_namespace["Tensor"] = Tensor
    doctest_namespace["mstype"] = mstype
    doctest_namespace["nn"] = nn
    doctest_namespace["mindspore"] = mindspore
    doctest_namespace["Momentum"] = Momentum
    doctest_namespace["learning_rate_schedule"] = learning_rate_schedule
    doctest_namespace["F"] = F
    doctest_namespace["C"] = C
    doctest_namespace["np"] = np
    doctest_namespace["context"] = context
    doctest_namespace["os"] = os
    doctest_namespace["initializer"] = initializer
    doctest_namespace["Normal"] = Normal
    doctest_namespace["Model"] = Model
    doctest_namespace["ms"] = mindspore
    doctest_namespace["Net"] = Net
    doctest_namespace["compression"] = compression
    return doctest_namespace
