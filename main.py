from option import args
from MGNNet.data.data import Data as MGN_Data
import MGNNet.loss as MGN_Loss
import MGNNet.model as MGN_Model
import MGNNet.util.utility as utility

from DataLib import Double_Channels_lid

import FaceNet.FaceNet as FaceNet_model
import FaceNet.DataPro as DataPro
from trainer import Trainer
import os
import sys
import torch
import matplotlib
matplotlib.use('TkAgg')


sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MGNNet'))

class_ID = {    "anger": 0,
                "contempt": 1,
                "disgust": 2,
                "fear": 3,
                "happy": 4,
                "sadness": 5,
                "surprise": 6
                }


def main():


##############   part1  Dataprocess     ####################

	datalib = Double_Channels_lid()

	# analyse the original inputs
	#datalib.get_CK_input("./Data", "./Data/CleanResult")

	# make train and val folders
	#datalib.make_CK_Train_val("./Data/CleanResult", "./Data/Train_Test")

	# add the inputs imgs
	'''for _, person_class in enumerate(os.listdir("./Data/Train_Test/Train")):
		if not os.path.exists(os.path.join("./Data/Train_Test_Add/Train", person_class)):
			os.mkdir(os.path.join("./Data/Train_Test_Add/Train", person_class))
		datalib.ArgumentImg(os.path.join("./Data/Train_Test/Train", person_class),os.path.join("./Data/Train_Test_Add/Train", person_class), 5)
	'''

	model_savedir = "./Model"
	if not os.path.exists(model_savedir):
		os.mkdir(model_savedir)

	args.datadir= "./Data/Train_Test"
	args.data_train = "./Data/Train_Test/Train"
	args.data_val = "./Data/Train_Test/Val"
	args.data_test = "./Data/Train_Test/Test"
	args.num_epochs = 500

	CFace_TrainCSV = DataPro.PrePareCSVData(args.data_train, "train", class_ID)
	CFace_TestCSV = DataPro.PrePareCSVData(args.data_test, "test", class_ID)

	args.face_Train_root_dir = './Data/Train_Test/Train'
	args.face_Train_csv_name = './Data/Train_Test/Train/train.csv'
	args.face_Test_root_dir = './Data/Train_Test/Test'
	args.face_Test_csv_name = './Data/Train_Test/Test/test.csv'

##############   part2_1  Train&Val     ####################
	'''
	ckpt = utility.checkpoint(args)
	Dense_loader = MGN_Data(args)
	Dense_model = MGN_Model.Model(args, ckpt)
	Dense_loss = MGN_Loss.Loss(args, ckpt) if not args.test_only else None

	args.embedding_size = 128
	args.num_clase = 7
	facenet_model = FaceNet_model.FaceNet(args.embedding_size, args.num_clase)
	device = 'cpu'

	MainThread = Trainer(args, Dense_model, facenet_model, Dense_loss, Dense_loader, ckpt,
							 args.face_Train_root_dir, args.face_Train_csv_name, args.face_Test_root_dir,
							 args.face_Test_csv_name, device)

	for epoch in range(args.num_epochs):
			# the dense part
			total_dense_los = MainThread.mgn_train()
			print("Info: Train  {} epoch total_mgn_loss is {}".format(int(epoch), total_dense_los))
			if epoch % 5 == 0:
				MainThread.mgn_val(Dense_model)
				print("     : val  {} epoch total_dense_loss is {}".format(int(epoch), total_dense_los))
				dense_checkpoint = os.path.join(model_savedir, "dense_only_epoch_" + str(epoch) + ".pth.tar")
				torch.save(Dense_model.state_dict(), dense_checkpoint)

			# the face part
			total_face_loss = MainThread.face_train()
			print("Info: Train  {} epoch total_face_loss is {}".format(int(epoch), total_face_loss))
			if epoch % 5 == 0:
				total_face_loss = MainThread.face_val(facenet_model)
				print("     : val  {} epoch val_face_loss is {}".format(int(epoch), total_face_loss))
				dense_checkpoint = os.path.join(model_savedir, "face_only_epoch_" + str(epoch) + ".pth.tar")
				torch.save(facenet_model.state_dict(), dense_checkpoint)


##############   part2_2  Train&Val     ####################
	ckpt = utility.checkpoint(args)
	Dense_loader = MGN_Data(args)
	Dense_model = MGN_Model.Model(args, ckpt)
	Dense_loss = MGN_Loss.Loss(args, ckpt) if not args.test_only else None

	args.embedding_size = 128
	args.num_clase = 7
	facenet_model = FaceNet_model.FaceNet(args.embedding_size, args.num_clase)
	device = 'cpu'

	MainThread = Trainer(args, Dense_model, facenet_model,  Dense_loss, Dense_loader, ckpt,
						 args.face_Train_root_dir, args.face_Train_csv_name,args.face_Test_root_dir,args.face_Test_csv_name, device)

	for epoch in range(args.num_epochs):
		total_face_loss, total_mgn_loss, total_loss = MainThread.train()
		print("Info: {} epoch total_face_loss is {} ,total_mgn_loss is {}, and the total_mgn_loss is {}".format(int(epoch),total_face_loss, total_mgn_loss,total_loss))
		if epoch %5 ==0:
			MainThread.val(Dense_model, facenet_model)
			face_checkpoint = os.path.join(model_savedir,"face_epoch_"+str(epoch)+".pth.tar")
			dense_checkpoint = os.path.join(model_savedir, "dense_epoch_" + str(epoch) + ".pth.tar")
			torch.save(facenet_model.state_dict(), face_checkpoint)
			torch.save(Dense_model.state_dict(), dense_checkpoint)
	'''

##############   part3_0  The Separate Test     ####################
	device = 'cpu'
	ckpt = utility.checkpoint(args)
	Dense_loader = MGN_Data(args)
	Dense_model = MGN_Model.Model(args, ckpt)
	dense_model_test_dir = "./Model/dense_Best_V3.0_0915.pth.tar"
	args.num_clase = 7
	args.embedding_size = 128
	facenet_model = FaceNet_model.FaceNet(args.embedding_size, args.num_clase)
	face_model_test_dir = './Model/face_Best_V3.0_0915.pth.tar'
	if device == 'cpu':
		Dense_model.load_state_dict(torch.load(dense_model_test_dir, map_location=torch.device('cpu')))
		facenet_model.load_state_dict(torch.load(face_model_test_dir, map_location=torch.device('cpu')))
	else:
		Dense_model.load_state_dict(torch.load(dense_model_test_dir))
		facenet_model.load_state_dict(torch.load(face_model_test_dir))
	Dense_loss = MGN_Loss.Loss(args, ckpt) if not args.test_only else None
	print("******     Load model successful     *******")

	MainThread = Trainer(args, Dense_model, facenet_model,  Dense_loss, Dense_loader, ckpt,
					  args.face_Train_root_dir, args.face_Train_csv_name,args.face_Test_root_dir,args.face_Test_csv_name, device)

	acc1 = MainThread.mgn_test()
	acc2 = MainThread.face_test()


##############   part3_1  The Final Test     ####################
	device = 'cpu'
	ckpt = utility.checkpoint(args)
	Dense_loader = MGN_Data(args)
	Dense_model = MGN_Model.Model(args, ckpt)
	dense_model_test_dir = "./Model/DoubleChannel_part1_V2.0.pth.tar"
	args.num_clase = 7
	args.embedding_size = 128
	facenet_model = FaceNet_model.FaceNet(args.embedding_size, args.num_clase)
	face_model_test_dir = './Model/DoubleChannel_part2_V2.0.pth.tar'
	if device == 'cpu':
		Dense_model.load_state_dict(torch.load(dense_model_test_dir, map_location=torch.device('cpu')))
		facenet_model.load_state_dict(torch.load(face_model_test_dir, map_location=torch.device('cpu')))
	else:
		Dense_model.load_state_dict(torch.load(dense_model_test_dir))
		facenet_model.load_state_dict(torch.load(face_model_test_dir))
	Dense_loss = MGN_Loss.Loss(args, ckpt) if not args.test_only else None
	print("******     Load model successful     *******")

	MainThread = Trainer(args, Dense_model, facenet_model,  Dense_loss, Dense_loader, ckpt,
					  args.face_Train_root_dir, args.face_Train_csv_name,args.face_Test_root_dir,args.face_Test_csv_name, device)
	acc0 = MainThread.test()
	print("The total acc is ", acc0)

if __name__ == '__main__':
    main()