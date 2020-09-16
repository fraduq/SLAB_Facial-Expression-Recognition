import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
from FaceNet import FaceNet
from FaceNet.DataPro import PrePareCSVData

class_ID = {"anger": 0,
                "disgust": 1,
                "fear": 2,
                "happiness": 3,
                "neutral": 4,
                "sadness": 5,
                "surprise": 6
                }

def trainer(root_dir, csv_name, lr, embedding_size,num_epochs,num_class):
    margin = 1
    num_triplets = 1000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FaceNet(embedding_size, num_class).to(device)
    if device == 'cuda':
        model = torch.nn.DataParallel(model)
        cudnn.benchmark = True
    optimizer = optim.SGD(model.parameters(), lr = lr, momentum = 0.9, weight_decay = 3e-4)
    print('start training ......')

    for epoch in range(num_epochs):
        model.train_facenet(root_dir, csv_name, epoch, optimizer, margin, num_triplets)
        '''clf = KNN_classifier(model, epoch, n_neighbors)  # get knn classifier
        test_facenet(epoch, model, clf, False)  # validate train set
        test_facenet(epoch, model, clf, True, epoch == num_epochs - 1)  # validate test set'''
        if (epoch + 1) % 5 == 0:
            lr = lr / 1.1
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
    return model


def main():
    CTrainCSV = PrePareCSVData("./data/images/train", "train", class_ID)
    CValCSV = PrePareCSVData("./data/images/test", "test", class_ID)

    lr = 0.012
    embedding_size = 128
    num_epochs = 14
    num_clase = 7
    trainer( './data/images/train', './data/images/train.csv', lr, embedding_size,num_epochs, num_clase)

if __name__ == '__main__':
    main()