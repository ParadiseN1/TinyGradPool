from tinygrad.tensor import Tensor
from tinygrad.nn import Conv2d
from tinygrad.nn import Linear


class AlexNet:
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        #the single-column model has 64, 192, 384, 384, 256 filters in the five convolutional layers, respectively
        self.l1 = Conv2d(3, 64, (11,11), stride=4, padding=2)
        self.l2 = Conv2d(64, 192, (5,5), padding=2)
        self.l3 = Conv2d(192, 384, (3,3))
        self.l4 = Conv2d(384, 384, (3,3))
        self.l5 = Conv2d(384, 256, (3,3))

        self.fc1=Linear(256*3*3, 4096)
        self.fc2=Linear(4096, 4096)
        self.fc3=Linear(4096, 1000)

    def local_response_norm(self, input, n=5, k=2, alpha=10e-4, beta=0.75):
        div = input.mul(input)
        div = div.unsqueeze(1)
        div = div.pad2d((0, 0,n//2,n//2)).avg_pool2d((n, 1), stride=1).squeeze(1)
        div = div.mul(alpha).add(k).pow(beta)

        return input/div
    
    def forward(self, x):
        ## Convolutional part
        out = self.local_response_norm(self.l1(x).relu()).max_pool2d((3,3), stride=2)
        out = self.local_response_norm(self.l2(out).relu()).max_pool2d((3,3), stride=2)
        out = self.l3(out).relu()
        out = self.l4(out).relu()
        out = self.l5(out).relu().max_pool2d((3,3), stride=2)
        ## Fully connected part
        out = self.fc1(out.flatten(1)).relu().dropout(0.5)
        out = self.fc2(out).relu().dropout(0.5)
        out = self.fc3(out).relu()
        return out.softmax()