import os
import sys
import linecache
from shutil import copyfile
from tqdm import trange
import cv2
import random
import time
import numpy as np
from PIL import ImageEnhance
from PIL import Image
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'MGNNet'))


index_label={
				0 : "neutral",
				1 : "anger",
				2 : "contempt",
				3 : "disgust",
				4 : "fear",
				5 : "happy",
				6 : "sadness",
				7 : "surprise"
			}

class Double_Channels_lid():
	def get_CK_input(self, input_dir, out_dir):
		label_dir = os.path.join(input_dir, "Emotion")
		img_dir = os.path.join(input_dir, "cohn-kanade-images")

		#according the label classify the imgs
		for i in trange(len(os.listdir(label_dir))):
			person = os.listdir(label_dir)[i]
			person_file = os.path.join(label_dir, person)
			for folder in (os.listdir((person_file))):
				folder_file = os.path.join(person_file,folder)
				#get the label
				for label_txt in (os.listdir(folder_file)):
					info_txt = os.path.join(folder_file,label_txt)
					info = str(linecache.getline(info_txt, 1))
					info = info[:info.find(".")]
					info = info[len(info)-1:]
					face_label = int(info)
					face_label = index_label[face_label]
					if not os.path.exists(os.path.join(out_dir,face_label)):
						os.makedirs(os.path.join(out_dir,face_label))
					break

				img_input = os.path.join(img_dir, person, folder)
				img_input_size = os.listdir(img_input)
				for order in range(0, 3):
					img_i = img_input_size[len(img_input_size)-1-order]
					copyfile(os.path.join(img_input, img_i), os.path.join(out_dir, face_label, img_i))

	def make_CK_Train_val(self, input_dir, output_dir):
		for _ ,face_class in enumerate(os.listdir(input_dir)):
			if not os.path.exists(os.path.join(output_dir, "Train", face_class)):
				os.mkdir(os.path.join(output_dir, "Train", face_class))
			if not os.path.exists(os.path.join(output_dir,"Val",face_class)):
				os.mkdir(os.path.join(output_dir,"Val",face_class))

			person_dir = os.path.join(input_dir,face_class)

			single_person = []
			for _, person in enumerate(os.listdir(person_dir)):
				person_name = person[:person.find("_")]
				if person_name not in single_person:
					single_person.append(person_name)
				else:
					continue

			for i in trange(len(single_person)):
				if i%6 == 0:
					for _, img in enumerate(os.listdir(person_dir)):
						if img[:img.find("_")]== single_person[i]:
							copyfile(os.path.join(person_dir,img), os.path.join(output_dir,"Val",face_class, img))
				else:
					for _, img in enumerate(os.listdir(person_dir)):
						if img[:img.find("_")] == single_person[i]:
							copyfile(os.path.join(person_dir, img), os.path.join(output_dir, "Train", face_class, img))

	def data_ContrastEngth(self, image_input, ength_Value):  # 对比对增强
		enh = ImageEnhance.Contrast(image_input)
		result = enh.enhance(ength_Value)
		return result

	def data_ColorEngth(self, image_input, ength_Value):  # 色度增强
		enh = ImageEnhance.Color(image_input)
		result = enh.enhance(ength_Value)
		return result

	def clamp(self, pv):  # 防止溢出
		if pv > 255:
			return 255
		elif pv < 0:
			return 0
		else:
			return pv

	def gaussian_noise(self, image):#高斯加噪
		h, w, c = image.shape
		for row in range(0, h):
			#s = np.random.normal(0, 15, 3)  # 产生随机数，每次产生三个
			for col in range(0, w):
				s = np.random.normal(0, 15, 3)  # 产生随机数，每次产生三个
				b = image[row, col, 0]  # blue
				g = image[row, col, 1]  # green
				r = image[row, col, 2]  # red
				image[row, col, 0] = self.clamp(b + s[0])
				image[row, col, 1] = self.clamp(g + s[1])
				image[row, col, 2] = self.clamp(r + s[2])

	def data_ContrastEngth(self, image_input, ength_Value):  # 对比对增强
		enh = ImageEnhance.Contrast(image_input)
		result = enh.enhance(ength_Value)
		return result

	def data_ColorEngth(self, image_input, ength_Value):  # 色度增强
		enh = ImageEnhance.Color(image_input)
		result = enh.enhance(ength_Value)
		return result

	def data_Brightness(self, image_input, ength_Value):  # 亮度增强
		enh = ImageEnhance.Brightness(image_input)
		result = enh.enhance(ength_Value)
		return result

	def data_Sharpness(self, image_input, ength_Value):  # 锐化
		enh = ImageEnhance.Sharpness(image_input)
		result = enh.enhance(ength_Value)
		return result

	def data_Augment(self, image_inout, ContrastEngth, ColorEngth, Brightness, Sharpness):
		image = Image.fromarray(cv2.cvtColor(image_inout, cv2.COLOR_BGR2RGB))

		if ContrastEngth == 0:
			image = self.data_ContrastEngth(image, 1.3)

		if ColorEngth == 0:
			image = self.data_ColorEngth(image, 1.4)

		if Brightness == 0:
			image = self.data_Brightness(image, 0.9)

		if Sharpness == 0:
			image = self.data_Sharpness(image, 3.0)

		img = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
		return img

	def ArgumentImg(self, intput_dir, output_dir, Agm_times):
		if not os.path.exists(output_dir):
			os.mkdir(output_dir)

		list_input = os.listdir(intput_dir)
		for i in range(0, len(list_input)):
			print("total: ", len(list_input), "processing: ", i)

			jpegInput = os.path.join(intput_dir, list_input[i])
			input = cv2.imread(jpegInput)

			for cir_num in range(Agm_times):
				random.seed(time.time())
				ContrastEngth = random.randint(0, 1)
				random.seed(time.time())
				ColorEngth = random.randint(0, 1)
				random.seed(time.time())
				Brightness = random.randint(0, 1)
				random.seed(time.time())
				Sharpness = random.randint(0, 1)
				img_agm_process = self.data_Augment(input, ContrastEngth, ColorEngth, Brightness, Sharpness)

				cv2.imshow("src", input)
				cv2.imshow("argument", img_agm_process)
				cv2.waitKey(1)

				save_str_agm = os.path.join(output_dir,
											list_input[i][0: list_input[i].rfind('.')] + "_Agm" + str(cir_num) + ".jpg")
				cv2.imwrite(save_str_agm, img_agm_process)

				self.gaussian_noise(img_agm_process)
				save_str_agm = os.path.join(output_dir,
											list_input[i][0: list_input[i].rfind('.')] + "_Agm_gauss" + str(
												cir_num) + ".jpg")
				cv2.imwrite(save_str_agm, img_agm_process)

			self.gaussian_noise(input)
			save_str_gauss = os.path.join(output_dir, list_input[i][0: list_input[i].rfind('.')] + "_Gauss.jpg")
			cv2.imshow("gauss", input)
			cv2.imwrite(save_str_gauss, input)

	def plot_confusion_matrix(self, cm, labels_name, title):
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]    # 归一化
		plt.imshow(cm, interpolation='nearest')    # 在特定的窗口上显示图像
		plt.title(title)    # 图像标题
		plt.colorbar()
		num_local = np.array(range(len(labels_name)))
		plt.xticks(num_local, labels_name, rotation=90)    # 将标签印在x轴坐标上
		plt.yticks(num_local, labels_name)    # 将标签印在y轴坐标上
		plt.ylabel('True label')
		plt.xlabel('Predicted label')

