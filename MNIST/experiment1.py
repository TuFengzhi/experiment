from keras.datasets import mnist
from keras.layers import Input
from operator import itemgetter
from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
import numpy as np
import os
import retrain
import re
import cv2

def eachFile(filepath):
    file = []
    pathDir =  os.listdir(filepath)
    for allDir in pathDir:
        child = os.path.join('%s%s' % (filepath, allDir))
        file.append(child)
    return file

transformations = ['light', 'occl', 'blackout']
target_models = [0]

img_rows, img_cols = 28, 28
(_, _), (_, y_test) = mnist.load_data()
input_shape = (img_rows, img_cols, 1)
input_tensor = Input(shape=input_shape)

model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
model3 = Model3(input_tensor=input_tensor)

for target_model in target_models:
    test_paths = ['./result/Model' + str(target_model + 1) + '/']
    for i in range(len(test_paths)):
        print('Model', target_model, 'Experiment', i)
        test_path = test_paths[i]
        total_origin = 0
        total_update = 0
        total = 0
        invalid = 0
        for transformation in transformations:
            current_origin = 0
            current_update = 0
            current = 0
            for file in eachFile(test_path + transformation + '/'):
                if os.path.basename(file) == '.DS_Store':
                    continue
                valid = True
                data = re.split('[_.]', os.path.basename(file))
                index = int(data[0])

                img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
                img2 = cv2.blur(img, (3, 3))
                # hist,bins = np.histogram(img.flatten(),256,[0,256])
                # cdf = hist.cumsum()
                # cdf_normalized = cdf * hist.max()/ cdf.max()
                # cdf_m = np.ma.masked_equal(cdf,0)
                # cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
                # cdf = np.ma.filled(cdf_m,0).astype('uint8')
                # img2 = cdf[img]

                img = img / 255.0
                img2 = img2 / 255.0

                # cv2.imshow('Color input image', img)
                # cv2.imshow('Histogram equalized', img2)
                # cv2.waitKey(0)
                
                img2 = np.expand_dims(img2.reshape(img_rows, img_cols, 1), axis=0)

                gen_img = np.expand_dims(img.reshape(img_rows, img_cols, 1), axis=0)
                orig_img = gen_img.copy()
                result = [np.argmax(model1.predict(gen_img)[0]), np.argmax(model2.predict(gen_img)[0]), np.argmax(model3.predict(gen_img)[0])]
                result_update = [np.argmax(model1.predict(img2)[0]), np.argmax(model2.predict(img2)[0]), np.argmax(model3.predict(img2)[0])]

                if i == 0:
                    if result[0] == result[1] == result[2] and result[0] == y_test[index]:
                            invalid += 1
                            valid = False

                if valid:
                    if result[target_model] == y_test[index]:
                        total_origin += 1
                        current_origin += 1

                    if result_update[target_model] == y_test[index]:
                        current_update += 1
                        total_update += 1

                    # result_tmp = []
                    # for constant in [-0.4, -0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3, 0.4]:
                    #     test_img = gen_img + constant
                    #     test_img[test_img > 1] = 1
                    #     test_img[test_img < 0] = 0
                    #     result_tmp.append((np.argmax(model1.predict(test_img)[0]), model1.predict(test_img)[0][np.argmax(model1.predict(test_img)[0])]))
                    # result_update = max(result_tmp, key=itemgetter(1))[0]
                    # if y_test[index] == result_update:
                    #     current_update += 1
                    #     total_update += 1
                    
                    total += 1
                    current += 1

            print(current_origin, current_update, current, current_origin / current, current_update / current)
        print('#Valid', total, '#True Negative(Original)', total - total_origin, '#True Negativa(Enhanced)', total - total_update, '#Invalid', invalid)
        print('Original Accuracy', total_origin * 1.0 / total, 'Enhanced Accuracy', total_update * 1.0 / total)
