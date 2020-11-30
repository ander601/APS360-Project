import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import cv2
from PIL import Image

class nn_after_feature(nn.Module):
  def __init__(self, hidden_one, hidden_two):
      super(nn_after_feature, self).__init__()
      self.name = "nn_after_feature"
      self.fc1 = nn.Linear(256*6*6, hidden_one)
      self.fc2 = nn.Linear(hidden_one, hidden_two)
      self.fc3 = nn.Linear(hidden_two, 2)

  def forward(self, x):
      x = x.view(-1, 256*6*6)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x


class mask_analysis_system:
    net_feature_extracter=""
    net_classifier=""

    def __init__(self):

        self.net_classifier=nn_after_feature(50, 100)
        if torch.cuda.is_available():
            state = torch.load("checkpoint128_0.001.pth")
            self.net_classifier.load_state_dict(state)
            self.net_classifier=self.net_classifier.cuda()
            self.net_feature_extracter = torchvision.models.alexnet(pretrained=True)
            self.net_feature_extracter=self.net_feature_extracter.cuda()
        else:
            state = torch.load("checkpoint128_0.001.pth", map_location=torch.device('cpu'))
            self.net_classifier.load_state_dict(state)
            self.net_feature_extracter = torchvision.models.alexnet(pretrained=True)


    def analysis(self, img_list):
        judgement_list=[]
        transformer = transforms.Compose([transforms.Resize([224, 224]), transforms.ToTensor()])
        for img in img_list:
            img=Image.fromarray(img)
            img=transformer(img)
            if torch.cuda.is_available():
                img=img.float().unsqueeze(0).cuda()
            else:
                img=img.float().unsqueeze(0)
            feature=self.net_feature_extracter.features(img)
            feature=feature.contiguous()
            out=self.net_classifier(feature)
            out=out.max(1, keepdim=True)[1][0]
            if out == 0:
                judgement_list.append(True)
            else:
                judgement_list.append(False)
        return judgement_list


def test_analysis():
    analyzer = mask_analysis_system()
    '''
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    image_loaded=torchvision.datasets.ImageFolder("Photos_Directory/Train", transform=transform)
    train_loader = torch.utils.data.DataLoader(image_loaded, batch_size=64,
                                                   num_workers=1)
    '''
    img = cv2.imread('Photos_Directory/Train/withoutMask/1.jpg.jpg')
    print(img.shape)
    out=analyzer.analysis([img])
    print(out)

if __name__ == '__main__':
    test_analysis()
