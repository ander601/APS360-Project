class MyNet(nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()

        self.pad = nn.ReflectionPad2d(3)
        self.pool = nn.MaxPool2d(2, 2)
        self.cv_1 = nn.Conv2d(3, 32, kernel_size=7, padding=0, bias=False)
        self.cv_2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.cv_3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.fc_1 = nn.Linear(128*224*224, 64)
        self.fc_2 = nn.Linear(64, 16)
        self.fc_3 = nn.Linear(16, 2)


    def forward(self, x):

        x = self.pad(x)
        x1 = self.pool(F.relu(self.cv_1(x)))
        x2 = self.pool(F.relu(self.cv_2(x1)))
        x3 = self.pool(F.relu(self.cv_3(x2)))

        bs = x3.shape[0]
        x3 = x3.view(bs, -1)
        x4 = F.tanh(self.fc_1(x3))
        x5 = F.tanh(self.fc_2(x4))
        x6 = F.tanh(self.fc_3(x5))

        return x6