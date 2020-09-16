import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.backends.cudnn as cudnn

import torchvision

import numpy as np
import torchvision.transforms as transforms
from torch.autograd import Variable,Function
from FaceNet.DataPro import TripletFaceDataset
#from sklearn import neighbors



class PairwiseDistance(Function):
    '''
        compute distance of the embedding features, p is norm, when p is 2, then return L2-norm distance
    '''
    def __init__(self, p):
        super(PairwiseDistance, self).__init__()
        self.norm = p

    def forward(self, x1, x2):
        eps = 1e-6  # in case of zeros
        diff = torch.abs(x1 - x2)     # subtraction
        out = torch.pow(diff, self.norm).sum(dim=1) # square
        return torch.pow(out + eps, 1. / self.norm) # L-p norm


class TripletLoss(Function):
    '''
       Triplet loss function.
       这里的margin就相当于是公式里的α
       loss = max(diatance(a,p) - distance(a,n) + margin, 0)
       forward method:
           args:
                anchor, positive, negative
           return:
                triplet loss
    '''
    def __init__(self, margin, num_classes=10):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.pdist = PairwiseDistance(2) # to calculate distance

    def forward(self, anchor, positive, negative):
        d_p = self.pdist.forward(anchor, positive) # distance of anchor and positive
        d_n = self.pdist.forward(anchor, negative) # distance of anchor and negative

        dist_hinge = torch.clamp(self.margin + d_p - d_n, min=0.0) # ensure loss is no less than zero
        loss = torch.mean(dist_hinge)
        return loss


class BasicBlock(nn.Module):
    '''
        resnet basic block.
        one block includes two conv layer and one residual
    '''
    expansion = 1
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)


        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class FaceNet_Base(nn.Module):
    def __init__(self, block, num_blocks, embedding_size=256, num_classes=10):
        super(FaceNet_Base, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        # feature map size 32x32
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # feature map size 32x32
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # feature map size 16x16
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        # feature map size 8x8
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        # feature map size 4x4
        # as we use resnet basic block, the expansion is 1
        self.linear = nn.Linear(512 * block.expansion, embedding_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def l2_norm(self, input):
        input_size = input.size()
        buffer = torch.pow(input, 2)
        normp = torch.sum(buffer, 1).add_(1e-10)
        norm = torch.sqrt(normp)
        _output = torch.div(input, norm.view(-1, 1).expand_as(input))
        output = _output.view(input_size)
        return output

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out
        '''# normalize the features, then we set margin easily
        self.features = self.l2_norm(out)
        # multiply by alpha = 10 as suggested in https://arxiv.org/pdf/1703.09507.pdf
        alpha = 10
        self.features = self.features * alpha
        # here we get the 256-d features, next we use those features to make prediction
        return self.features
        '''


class FaceNet(nn.Module):
    def __init__(self,embedding_size=256, num_classes=10):
        super(FaceNet, self).__init__()
        self.model = FaceNet_Base(BasicBlock, [2, 2, 2, 2], embedding_size, num_classes)
        self.device = "cuda"

    def forward(self, x):
        return self.model(x)

    def train_facenet(self, root_dir, csv_name, epoch, optimizer, margin, num_triplets):
        self.model.train()
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=np.array([0.4914, 0.4822, 0.4465]), std=np.array([0.2023, 0.1994, 0.2010])),
        ])

        train_set = TripletFaceDataset(root_dir, csv_name, num_triplets=num_triplets,transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True, num_workers=8, pin_memory=True)

        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            '''data[0], target[0] = data[0].cuda(), target[0].cuda()  # anchor to cuda
            data[1], target[1] = data[1].cuda(), target[1].cuda()  # positive to cuda
            data[2], target[2] = data[2].cuda(), target[2].cuda()  # negative to cuda
            data[0], target[0] = Variable(data[0]), Variable(target[0])  # anchor
            data[1], target[1] = Variable(data[1]), Variable(target[1])  # positive
            data[2], target[2] = Variable(data[2]), Variable(target[2])  # negative'''

            optimizer.zero_grad()

            face_inputs_0 = data[0].to(self.device)
            face_inputs_1 = data[1].to(self.device)
            face_inputs_2 = data[2].to(self.device)
            anchor = self.model.forward(face_inputs_0)
            positive = self.model.forward(face_inputs_1)
            negative = self.model.forward(face_inputs_2)

            loss = TripletLoss(margin=margin, num_classes=10).forward(anchor, positive, negative)  # get triplet loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        context = 'Train Epoch: {} [{}/{}], Average loss: {:.4f}'\
            .format(epoch, len(train_loader.dataset), len(train_loader.dataset), total_loss / len(train_loader))
        print(context)


    def test_facenet(self, epoch, model, clf, test=True):
        model.eval()
        transform = transforms.Compose([transforms.Resize(32),transforms.ToTensor(),
                                        transforms.Normalize(mean=np.array([0.485, 0.456, 0.406]),std=np.array([0.229, 0.224, 0.225])), ])

        # prepare dataset by ImageFolder, data should be classified by directory
        test_set = torchvision.datasets.ImageFolder(root='./mnist/test' if test else './mnist/train',
                                                    transform=transform)
        test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

        correct, total = 0, 0
        for i, (data, target) in enumerate(test_loader):
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model.forward(data)
            predicted = clf.predict(output.data.cpu().numpy())

            correct += (torch.tensor(predicted) == target.data.cpu()).sum()
            total += target.size(0)

        context = 'Accuracy of model in ' + ('test' if test else 'train') + \
                  ' set is {}/{}({:.2f}%)'.format(correct, total, 100. * float(correct) / float(total))
        print(context)

'''def KNN_classifier(model, epoch, n_neighbors):
       
        #use all train set data to make KNN classifier
        
        model.eval()
        # preprocessing function for image
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=np.array([0.485, 0.456, 0.406]),
                std=np.array([0.229, 0.224, 0.225])),
        ])
        # prepare dataset by ImageFolder, data should be classified by directory
        train_set = torchvision.datasets.ImageFolder(root='./mnist/train', transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=False)

        features, labels = [], []  # store features and labels
        for i, (data, target) in enumerate(train_loader):
            #  load data to gpu
            data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            # forward
            output = model(data)
            # get features and labels to make knn classifier，extend的含义是给列表追加；列表
            features.extend(output.data.cpu().numpy())
            labels.extend(target.data.cpu().numpy())

        # n_neighbor is adjustable
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(features, labels)
        return clf
'''




