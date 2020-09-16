import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
import os
import csv
from tqdm import tqdm


class TripletFaceDataset(Dataset):
    def __init__(self, root_dir, csv_name, num_triplets, transform=None):
        '''
        randomly select triplet,which means anchor,positive and negative are all selected randomly.
        args:
            root_dir : dir of data set
            csv_name : dir of train.csv
            num_triplets: total number of triplets
        '''
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_name)
        self.num_triplets = num_triplets
        self.transform = transform
        self.training_triplets = self.generate_triplets(self.df, self.num_triplets)

    @staticmethod
    def generate_triplets(df, num_triplets):
        def make_dictionary_for_pic_class(df):
            '''
                make csv to the format that we want
              - pic_classes = {'class0': [class0_id0, ...], 'class1': [class1_id0, ...], ...}
              建立类名与id一一对应的一个字典
            '''
            pic_classes = dict()
            for idx, label in enumerate(df['class']):
                if label not in pic_classes:
                    pic_classes[label] = []
                pic_classes[label].append(df.iloc[idx, 0])
            return pic_classes

        triplets = []
        classes = df['class'].unique()
        pic_classes = make_dictionary_for_pic_class(df)

        for _ in range(num_triplets):

            '''
              - randomly choose anchor, positive and negative images for triplet loss
              - anchor and positive images in pos_class
              - negative image in neg_class
              - at least, two images needed for anchor and positive images in pos_class
              - negative image should have different class as anchor and positive images by definition
            '''

            pos_class = np.random.choice(classes)  # random choose positive class
            neg_class = np.random.choice(classes)  # random choose negative class
            while len(pic_classes[pos_class]) < 2:
                pos_class = np.random.choice(classes)
            while pos_class == neg_class:
                neg_class = np.random.choice(classes)

            pos_name = df.loc[df['class'] == pos_class, 'name'].values[0]  # get positive class's name
            neg_name = df.loc[df['class'] == neg_class, 'name'].values[0]  # get negative class's name

            if len(pic_classes[pos_class]) == 2:
                ianc, ipos = np.random.choice(2, size=2, replace=False)
            else:
                ianc = np.random.randint(0, len(pic_classes[pos_class]))  # random choose anchor
                ipos = np.random.randint(0, len(pic_classes[pos_class]))  # random choose positive
                while ianc == ipos:
                    ipos = np.random.randint(0, len(pic_classes[pos_class]))
            ineg = np.random.randint(0, len(pic_classes[neg_class]))  # random choose negative

            triplets.append([pic_classes[pos_class][ianc], pic_classes[pos_class][ipos], pic_classes[neg_class][ineg],
                             pos_class, neg_class, pos_name, neg_name])

        return triplets

    def __getitem__(self, idx):

        anc_id, pos_id, neg_id, pos_class, neg_class, pos_name, neg_name = self.training_triplets[idx]

        anc_img = os.path.join(self.root_dir, str(pos_name), str(anc_id) + '.jpg')  # join the path of anchor
        pos_img = os.path.join(self.root_dir, str(pos_name), str(pos_id) + '.jpg')  # join the path of positive
        neg_img = os.path.join(self.root_dir, str(neg_name), str(neg_id) + '.jpg')  # join the path of nagetive

        anc_img = Image.open(anc_img).convert('RGB')  # open the anchor image
        pos_img = Image.open(pos_img).convert('RGB')  # open the positive image
        neg_img = Image.open(neg_img).convert('RGB')  # open the negative image

        pos_class = torch.from_numpy(np.array([pos_class]).astype('long'))  # make label transform the type we want
        neg_class = torch.from_numpy(np.array([neg_class]).astype('long'))  # make label transform the type we want

        data = [anc_img, pos_img, neg_img]
        label = [pos_class, pos_class, neg_class]
        if self.transform:
            data = [self.transform(img) for img in data]

        return data, label

    def __len__(self):

        return len(self.training_triplets)


class PrePareCSVData():
    def __init__(self, file_dir, model, class_ID):
        csv_save_dir = file_dir[:file_dir.rfind("/")]
        if model == "train":
            CSV_out = open(os.path.join(csv_save_dir, "train.csv"), "w", newline='')
        elif model == "test":
            CSV_out = open(os.path.join(csv_save_dir, "test.csv"), "w", newline='')
        csv_writer = csv.writer(CSV_out, dialect="excel")
        title_list = []
        title_list.append("id")
        title_list.append("name")
        title_list.append("class")
        csv_writer.writerow(title_list)

        for root, dirs, files in os.walk(file_dir):
            for file_name in dirs:
                test_dir_name = os.path.join(root, file_name)
                for jpg_name in tqdm(os.listdir(test_dir_name)):
                    ID = class_ID[file_name]
                    jpeg_info = []
                    jpeg_info.append(jpg_name[:jpg_name.find(".")])
                    jpeg_info.append(file_name)
                    jpeg_info.append(ID)
                    csv_writer.writerow(jpeg_info)