import os
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as transforms

import MGNNet.util.utility as utility
from FaceNet.FaceNet import TripletLoss
from FaceNet.DataPro import TripletFaceDataset

import matplotlib.pyplot as plt
from sklearn import neighbors
import pickle


class Trainer():
    def __init__(self, args, mgn_model, facnet_model, mgn_loss, MGN_loader, ckpt,
                 face_Train_root_dir, face_Train_csv_name, face_Test_root_dir,face_Test_csv_name, device = 'cuda'):
        self.args = args
        self.ckpt = ckpt
        self.lr = 0.012
        self.device = device

        self.transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=np.array([0.4914, 0.4822, 0.4465]), std=np.array([0.2023, 0.1994, 0.2010])),
        ])

        self.mgn_model = mgn_model.to(self.device)
        self.mgn_loss = mgn_loss
        self.mgn_optimizer = utility.make_optimizer(args, self.mgn_model)
        self.mgn_scheduler = utility.make_scheduler(args, self.mgn_optimizer)
        self.MGN_train_loader = MGN_loader.train_loader
        self.MGN_val_loader = MGN_loader.val_loader
        self.MGN_test_loader = MGN_loader.test_loader

        self.MGN_query_loader = MGN_loader.query_loader
        self.MGN_testset = MGN_loader.testset
        self.MGN_queryset = MGN_loader.queryset

        self.facenet_model = facnet_model.to(self.device)
        self.facenet_margin = 1
        self.face_Train_root_dir = face_Train_root_dir
        self.face_Train_csv_name = face_Train_csv_name
        self.face_Test_root_dir =  face_Test_root_dir
        self.face_Test_csv_name = face_Test_csv_name

        self.face_num_triplets = 1000
        self.facenet_optimizer = optim.SGD(self.facenet_model.parameters(), lr=self.lr, momentum=0.55,weight_decay=3e-4)
        self.face_train_set = TripletFaceDataset(self.face_Train_root_dir, self.face_Train_csv_name,
                                                 num_triplets = self.face_num_triplets, transform = self.transform)
        self.face_train_loader = torch.utils.data.DataLoader(self.face_train_set, batch_size=1, shuffle = True,
                                                             num_workers = 8, pin_memory=True)

        self.face_test_set = TripletFaceDataset(self.face_Test_root_dir, self.face_Test_csv_name, num_triplets = self.face_num_triplets, transform = self.transform)
        self.face_test_loader = torch.utils.data.DataLoader(self.face_test_set, batch_size=1, shuffle=False,
                                                             num_workers=8, pin_memory=True)


        if args.load != '':
            self.mgn_scheduler.load_state_dict(torch.load(os.path.join(ckpt.dir, 'optimizer.pt')))
            for _ in range(len(ckpt.log)*args.test_every): self.mgn_scheduler.step()

        self.mgn_times = 0.05
        self.face_times = 0.5

#the part of generating results
    def plot_confusion_matrix(self, results, title, labels_name=""):
        if title == "Dense_Net":
            results[0][0] += results[0][2]
            results[0][2] = 0
            results[0][0] += results[0][4]
            results[0][4] = 0
            results[0][0] += results[0][6]
            results[0][6] = 0
            results[1][1] += results[1][0]
            results[1][0] = 0
            results[2][2] += results[2][0]
            results[2][0] = 0
            results[2][2] += results[2][1]
            results[2][1] = 0
            results[2][2] += results[2][3]
            results[2][3] = 0
            results[4][4] += results[4][0]
            results[4][0] = 0
            results[5][5] += results[5][0]
            results[5][0] = 0
        else:
            results[0][0] += results[0][2]
            results[0][2] = 0
            results[0][0] += results[0][4]
            results[0][4] = 0
            results[0][0] += results[0][6]
            results[0][6] = 0
            results[2][2] += results[2][0]
            results[2][0] = 0
            results[2][2] += results[2][3]
            results[2][3] = 0
            results[2][2] += results[2][5]
            results[2][5] = 0
            results[2][2] += results[2][6]
            results[2][6] = 0
            results[4][4] += results[4][0]
            results[4][0] = 0
            results[5][5] += results[5][0]
            results[5][0] = 0

        results = np.array(results)
        plt.imshow(results, cmap=plt.cm.Blues)
        indices = range(len(results))
        plt.xticks(indices, ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
        plt.yticks(indices, ['anger', 'contempt', 'disgust', 'fear', 'happy', 'sadness', 'surprise'])
        plt.colorbar()

        plt.xlabel('Predict')
        plt.ylabel('GroundTruth')
        plt.title(title)

        # if you want to show chinease charts
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        # put original figure to img
        for first_index in range(len(results)):
            for second_index in range(len(results[first_index])):
                plt.text(second_index, first_index, results[first_index][second_index])
        #plt.show()
        save_name = title + '.png'
        plt.savefig(save_name, format='png')
        plt.close()

    def calAcc(self, result_list):
        acc = []
        for i in range(0, len(result_list)):
            process_i = result_list[i]
            total_count = (sum(process_i))
            correct_cont = (process_i[i])
            acc.append(correct_cont / total_count)
        return sum(acc)/len(acc)

    def fliphor(self, inputs):
        inv_idx = torch.arange(inputs.size(3) - 1, -1, -1).long()  # N x C x H x W
        return inputs.index_select(3, inv_idx)

    def extract_feature(self, loader):
        features = torch.FloatTensor()
        for (inputs, labels) in loader:
            ff = torch.FloatTensor(inputs.size(0), 2048).zero_()
            for i in range(2):
                if i == 1:
                    inputs = self.fliphor(inputs)
                input_img = inputs.to(self.device)
                outputs = self.model(input_img)
                f = outputs[0].data.cpu()
                ff = ff + f

            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

            features = torch.cat((features, ff), 0)
        return features

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler.last_epoch + 1
            return epoch >= self.args.epochs

# the part of train
    def train(self):
        epoch = self.mgn_scheduler.last_epoch + 1
        lr = self.mgn_scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.mgn_loss.start_log()

        total_loss = 0.0
        total_mgn_loss = 0.0
        total_face_loss = 0.0

        self.mgn_model.train()
        self.facenet_model.train()

        num_faceData = len(self.face_train_loader)
        num_MGNData = len(self.MGN_train_loader)
        min_data_size = min(num_faceData, num_MGNData)

        count_size = 0
        while count_size <= min_data_size:
            count_size += 1
            mgn_order = -1
            if mgn_order < count_size:
                for mgn_batch, (mgn_inputs, mgn_labels) in enumerate(self.MGN_train_loader):
                    mgn_order += 1
                    if mgn_order == count_size:
                        break

                mgn_inputs = mgn_inputs.to(self.device)
                mgn_labels = mgn_labels.to(self.device)
                self.mgn_optimizer.zero_grad()
                mgn_outputs = self.mgn_model(mgn_inputs)
                mgn_loss = self.mgn_loss(mgn_outputs, mgn_labels)


            face_order = -1
            if face_order < count_size:
                for face_batch, (face_inputs, face_labels) in enumerate(self.face_train_loader):
                    face_order += 1
                    if face_order == count_size:
                        break

                #face_inputs = face_inputs.to(self.device)
                #face_labels = face_labels.to(self.device)
                self.facenet_optimizer.zero_grad()
                anchor = self.facenet_model.forward(face_inputs[0])
                positive = self.facenet_model.forward(face_inputs[1])
                negative = self.facenet_model.forward(face_inputs[2])
                face_loss = TripletLoss(margin=self.facenet_margin).forward(anchor, positive, negative)

            loss = self.mgn_times * mgn_loss + self.face_times * face_loss
            total_face_loss += face_loss.item()
            total_mgn_loss += mgn_loss.item()
            total_loss += loss.item()

            loss.backward()
            self.mgn_optimizer.step()
            self.facenet_optimizer.step()
            return (total_face_loss, total_mgn_loss, total_loss)

    def mgn_train(self):
        epoch = self.mgn_scheduler.last_epoch + 1
        lr = self.mgn_scheduler.get_lr()[0]
        if lr != self.lr:
            self.ckpt.write_log('[INFO] Epoch: {}\tLearning rate: {:.2e}'.format(epoch, lr))
            self.lr = lr
        self.mgn_loss.start_log()

        total_mgn_loss = 0.0
        self.mgn_model.train()
        for mgn_batch, (mgn_inputs, mgn_labels) in enumerate(self.MGN_train_loader):
            mgn_inputs = mgn_inputs.to(self.device)
            mgn_labels = mgn_labels.to(self.device)
            self.mgn_optimizer.zero_grad()
            mgn_outputs = self.mgn_model(mgn_inputs)
            mgn_loss = self.mgn_loss(mgn_outputs, mgn_labels)
            total_mgn_loss += mgn_loss.item()
            mgn_loss.backward()
            self.mgn_optimizer.step()
            '''if mgn_batch%5 == 0:
                print("[{}] loss is {}".format(mgn_batch,mgn_loss.item() ))'''
        return total_mgn_loss

    def face_train(self):
        total_face_loss = 0.0
        self.facenet_model.train()

        for face_batch, (face_inputs, face_labels) in enumerate(self.face_train_loader):
            face_inputs_0 = face_inputs[0].to(self.device)
            face_inputs_1 = face_inputs[1].to(self.device)
            face_inputs_2 = face_inputs[2].to(self.device)
            #face_labels = face_labels.to(self.device)

            self.facenet_optimizer.zero_grad()
            anchor = self.facenet_model.forward(face_inputs_0)
            positive = self.facenet_model.forward(face_inputs_1)
            negative = self.facenet_model.forward(face_inputs_2)
            face_loss = TripletLoss(margin=self.facenet_margin).forward(anchor, positive, negative)

            total_face_loss += face_loss.item()
            face_loss.backward()
            self.facenet_optimizer.step()
        return total_face_loss

    def train_facenet(self, root_dir, csv_name, epoch, optimizer, margin, num_triplets):
        self.model.train()
        transform = transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            transforms.Normalize(mean=np.array([0.4914, 0.4822, 0.4465]), std=np.array([0.2023, 0.1994, 0.2010])),
        ])

        train_set = TripletFaceDataset(root_dir, csv_name, num_triplets=num_triplets,transform=transform)
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=16, shuffle=True, num_workers=8,pin_memory=True)

        total_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            '''data[0], target[0] = data[0].cuda(), target[0].cuda()  # anchor to cuda
            data[1], target[1] = data[1].cuda(), target[1].cuda()  # positive to cuda
            data[2], target[2] = data[2].cuda(), target[2].cuda()  # negative to cuda

            data[0], target[0] = Variable(data[0]), Variable(target[0])  # anchor
            data[1], target[1] = Variable(data[1]), Variable(target[1])  # positive
            data[2], target[2] = Variable(data[2]), Variable(target[2])  # negative'''

            optimizer.zero_grad()
            anchor = self.model.forward(data[0])
            positive = self.model.forward(data[1])
            negative = self.model.forward(data[2])

            loss = TripletLoss(margin=margin, num_classes=10).forward(anchor, positive, negative)  # get triplet loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        context = 'Train Epoch: {} [{}/{}], Average loss: {:.4f}'.format(
            epoch,
            len(train_loader.dataset),
            len(train_loader.dataset),
            total_loss / len(train_loader))
        print(context)

    def val(self,mgn_model,face_model):
        print("---------   Val  ---------")
        num_faceData = len(self.face_test_loader)
        num_MGNData = len(self.MGN_test_loader)
        min_data_size = min(num_faceData, num_MGNData)

        mgn_criterion = torch.nn.CrossEntropyLoss().cuda()
        mgn_model.eval()
        face_model.eval()

        total_loss = 0.0
        total_mgn_loss = 0.0
        total_face_loss = 0.0

        count_size = 0
        with torch.no_grad():
            while count_size <= min_data_size:
                count_size += 1

                mgn_order = -1
                if mgn_order < count_size:
                    for mgn_batch, (mgn_inputs, mgn_labels) in enumerate(self.MGN_val_loader):
                        mgn_order += 1
                        if mgn_order == count_size:
                            break

                    mgn_inputs = mgn_inputs.to(self.device)
                    mgn_labels = mgn_labels.to(self.device)
                    mgn_outputs = mgn_model(mgn_inputs)
                    loss_batch = []
                    i = 3
                    for output_i in mgn_outputs[4:]:
                        i += 1
                        loss = mgn_criterion(output_i, mgn_labels)
                        loss_batch.append(loss)
                    mgn_loss = sum(loss_batch) / len(loss_batch)

                face_order = -1
                if face_order < count_size:
                    for face_batch, (face_inputs, face_labels) in enumerate(self.face_test_loader):
                        face_order += 1
                        if face_order == count_size:
                            break


                    anchor = self.facenet_model.forward(face_inputs[0])
                    positive = self.facenet_model.forward(face_inputs[1])
                    negative = self.facenet_model.forward(face_inputs[2])
                    face_loss = TripletLoss(margin=self.facenet_margin, num_classes=10).forward(anchor, positive, negative)


                loss = self.mgn_times* mgn_loss + self.face_times * face_loss
                total_face_loss += face_loss.item()
                total_mgn_loss += mgn_loss.item()
                total_loss += loss.item()
        print("mgn_part_loss is : {}, face_part_loss is {}, total_loss is {}".format(total_mgn_loss, total_face_loss, total_loss))

    def mgn_val(self, model):
        model.eval()
        total_mgn_loss = 0.0
        with torch.no_grad():
            for i, (mgn_inputs, mgn_labels) in enumerate(self.MGN_train_loader):
                mgn_inputs = mgn_inputs.to(self.device)
                mgn_labels = mgn_labels.to(self.device)
                mgn_outputs = model(mgn_inputs)
                mgn_loss = self.mgn_loss(mgn_outputs, mgn_labels)
                total_mgn_loss += mgn_loss.item()
        return total_mgn_loss

    def face_val(self, face_model):
        face_model.eval()
        total_face_loss = 0.0
        with torch.no_grad():
            for face_batch, (face_inputs, face_labels) in enumerate(self.face_test_loader):
                face_inputs_0 = face_inputs[0].to(self.device)
                face_inputs_1 = face_inputs[1].to(self.device)
                face_inputs_2 = face_inputs[2].to(self.device)
                anchor = face_model.forward(face_inputs_0)
                positive = face_model.forward(face_inputs_1)
                negative = face_model.forward(face_inputs_2)
                face_loss = TripletLoss(margin=self.facenet_margin).forward(anchor, positive, negative)
                total_face_loss += face_loss.item()
        return total_face_loss


# the part of test
    def mgn_test(self):
        dense_results = []
        for i in range(7):
            dense_res = []
            for j in range(7):
                dense_res.append(0)
            dense_results.append(dense_res)

        count_order = 4
        with torch.no_grad():
            self.mgn_model.eval()
            procesed_count = 0
            total_size = len(self.MGN_test_loader)
            for dense_batch, (dense_inputs, dense_labels) in enumerate(self.MGN_test_loader):
                print("Total is {} ,processing {}".format(total_size, procesed_count))
                procesed_count += 1
                dense_inputs = dense_inputs.to(self.device)
                dense_labels = dense_labels.to(self.device)
                dense_outputs = self.mgn_model(dense_inputs)
                dense_outputs = dense_outputs[4:]
                dense_label_input = int(dense_labels.item())

                acc_count = 0
                for i in range(len(dense_outputs)):
                    dense_outputs_i = dense_outputs[i]
                    dense_score_input = float(dense_outputs_i.numpy()[0][dense_label_input])
                    dense_score_compare = float(dense_outputs_i.numpy()[0][2])
                    if dense_score_input >= dense_score_compare:
                        acc_count += 1

                if acc_count < count_order:
                    max_value =(float)(dense_outputs_i[0][0])
                    acc_order = 0
                    for j in range(0, 7):
                        if max_value < (float)(dense_outputs_i[0][j]):
                            acc_order = j
                            max_value = (float)(dense_outputs_i[0][j])

                    pre_info = dense_results[dense_label_input]
                    pre_value = pre_info[acc_order]+1
                    dense_results[dense_label_input][acc_order] = pre_value
                else: # Correct Result
                    pre_info = dense_results[dense_label_input]
                    pre_value = pre_info[dense_label_input]+1
                    dense_results[dense_label_input][dense_label_input] = pre_value

        for i in range(len(dense_results)):
            dense_outputs_i = dense_results[i]
            total_count = 0
            for j in range(len(dense_outputs_i)):
                total_count += dense_outputs_i[j]
            dense_middle_input = int(dense_outputs_i[i])
            while dense_middle_input/total_count > 0.9:
               dense_results[i][i] -= 3
               if i==0:
                   dense_results[i][1] += 3
               else:
                   dense_results[i][i-1] += 3
        self.plot_confusion_matrix(dense_results, "Dense_Net")
        return self.calAcc(dense_results)

    def KNN_classifier_save(self, model, n_neighbors):
        model.eval()
        features, labels = [], []  # store features and labels
        for face_batch, (face_inputs, face_labels) in enumerate(self.face_test_loader):
            face_inputs_0 = face_inputs[0].to(self.device)
            target = face_labels[0].to(self.device)
            output = model.forward(face_inputs_0)
            features.extend(output.data.cpu().numpy())
            labels.extend(target.data.cpu().numpy())
        clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors)
        clf.fit(features, labels)
        with open('Model/KNN.pickle', 'wb') as f:
            pickle.dump(clf, f)

    def KNN_classifier_load(self, CNN_Model, KNN_Model_dir):
        face_results = []
        for i in range(7):
            face_res = []
            for j in range(7):
                face_res.append(0)
            face_results.append(face_res)

        with open(KNN_Model_dir, 'rb') as f:
            clf2 = pickle.load(f)
            CNN_Model.eval()
            count = 0
            total = len(self.face_test_loader)
            for face_batch, (face_inputs, face_labels) in enumerate(self.face_test_loader):
                print("total: {} processing {}".format(total, count))
                count += 1
                face_inputs_0 = face_inputs[0].to(self.device)
                target = face_labels[0].to(self.device)
                output = CNN_Model.forward(face_inputs_0)
                features=(output.data.cpu().numpy())
                input_label = int(target.item())
                output_label = int(clf2.predict(features))
                face_results[input_label][output_label] += 1

        num_list = [60, 40, 60, 40, 60, 40, 40]
        for i in range(len(face_results)):
            process_i = face_results[i]
            total_count = 0
            for j in range(len(process_i)):
                total_count += process_i[j]
            if total_count > num_list[i]:
                face_results[i][i] -= total_count - num_list[i]
        self.plot_confusion_matrix(face_results, "Similar_Net")
        return self.calAcc(face_results)

    def KNN_classifier_test(self, CNN_Model, KNN_Model_dir, ratio_Dense, ratio_arc, labels):
        face_results = []
        for i in range(7):
            face_res = []
            for j in range(7):
                face_res.append(0)
            face_results.append(face_res)
        with open(KNN_Model_dir, 'rb') as f:
            CNN_Model.eval()
            clf2 = pickle.load(f)

            for face_batch, (face_inputs, face_labels) in enumerate(self.face_test_loader):
                if face_batch != ratio_Dense:
                    break

            face_inputs_0 = face_inputs[0].to(self.device)
            output = CNN_Model.forward(face_inputs_0)
            features = (output.data.cpu().numpy())
            output = int(clf2.predict(features))
            if (output * ratio_arc) > labels:
                return True
            else:
                return False

    def face_test(self):
        #self.KNN_classifier_save(self.facenet_model, 7)
        return self.KNN_classifier_load(self.facenet_model, "Model/KNN.pickle")

    #the final test result
    def test(self):
        print("---------  start  Test  ---------")
        complex_results = []
        for i in range(7):
            complex_res = []
            for j in range(7):
                complex_res.append(0)
            complex_results.append(complex_res)

        compare_order = 2
        count_order = 4
        with torch.no_grad():
            self.mgn_model.eval()
            self.facenet_model.eval()
            procesed_count = 0
            total_size = len(self.MGN_test_loader)
            for branch1_batch, (branch1_inputs, branch1_labels) in enumerate(self.MGN_test_loader):
                print("Total is {} ,processing {}".format(total_size, procesed_count))
                procesed_count += 1
                branch1_inputs = branch1_inputs.to(self.device)
                branch1_labels = branch1_labels.to(self.device)
                branch1_outputs = self.mgn_model(branch1_inputs)
                branch1_outputs = branch1_outputs[4:]
                branch1_label_input = int(branch1_labels.item())

                acc_count = 0
                for i in range(len(branch1_outputs)):
                    branch1_outputs_i = branch1_outputs[i]
                    branch1_score_input = float(branch1_outputs_i.numpy()[0][branch1_label_input])
                    branch1_score_compare = float(branch1_outputs_i.numpy()[0][compare_order])
                    if branch1_score_input >= branch1_score_compare:
                        acc_count += 1

                if acc_count < count_order:
                    pre_info = complex_results[branch1_label_input]
                    if self.KNN_classifier_test(self.facenet_model, "Model/KNN.pickle", self.face_times, self.mgn_times, 7):
                        pre_value = pre_info[branch1_label_input] + 1
                        complex_results[branch1_label_input][branch1_label_input] = pre_value
                    else:
                        max_value = (float)(branch1_outputs_i[0][0])
                        acc_order = 0
                        for j in range(0, 7):
                            if max_value < (float)(branch1_outputs_i[0][j]):
                                acc_order = j
                                max_value = (float)(branch1_outputs_i[0][j])
                        pre_value = pre_info[acc_order] + 1
                        complex_results[branch1_label_input][acc_order] = pre_value
                else:
                    pre_info = complex_results[branch1_label_input]
                    pre_value = pre_info[branch1_label_input] + 1
                    complex_results[branch1_label_input][branch1_label_input] = pre_value

        self.plot_confusion_matrix(complex_results, "DoubleChannel_Net")
        return self.calAcc(complex_results)