from __future__ import print_function

from keras.datasets import mnist
from keras.layers import Input
from scipy.misc import imsave

from Model1 import Model1
from Model2 import Model2
from Model3 import Model3
from configs import bcolors
from utils import *

import retrain

transformations = ['light', 'occl', 'blackout']
weight_diff = 1 # weight hyperparm to control differential behavior
weight_nc = 1 # weight hyperparm to control neuron coverage
step = 1 # step size of gradient descent
seeds = 1000 # number of seeds of input
grad_iterations = 100 # number of iterations of gradient descent
threshold = 0.5 # threshold for determining neuron activated
target_model = 2 # target model that we want it predicts differently, choices=[0, 1, 2]
start_point = (0, 0)# occlusion upper left corner coordinate
occlusion_size = (10, 10)# occlusion size

# input image dimensions
img_rows, img_cols = 28, 28
input_shape = (img_rows, img_cols, 1)
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
model1 = Model1(input_tensor=input_tensor)
model2 = Model2(input_tensor=input_tensor)
# model3 = Model3(input_tensor=input_tensor)
# model1 = retrain.Model1(input_tensor=input_tensor)
# model2 = retrain.Model2(input_tensor=input_tensor)
model3 = retrain.Model3(input_tensor=input_tensor)

# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

total_origin = 0
# ==============================================================================================
# start gen inputs
for transformation in transformations:
    (_, _), (x_test, _) = mnist.load_data()
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    x_test = x_test.astype('float32')
    x_test /= 255
    for i in range(seeds):
        gen_img = np.expand_dims(x_test[i], axis=0)
        orig_img = gen_img.copy()
        
        # first check if input already induces differences
        label1, label2, label3 = np.argmax(model1.predict(gen_img)[0]), np.argmax(model2.predict(gen_img)[0]), np.argmax(
            model3.predict(gen_img)[0])
        print('Image Index', i, ':', label1, label2, label3)
        if not label1 == label2 == label3:
            print(bcolors.OKGREEN + 'input already causes different outputs: {}, {}, {}'.format(label1, label2, label3) + bcolors.ENDC)

            update_coverage(gen_img, model1, model_layer_dict1, threshold)
            update_coverage(gen_img, model2, model_layer_dict2, threshold)
            update_coverage(gen_img, model3, model_layer_dict3, threshold)

            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                     neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                           neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            gen_img_deprocessed = deprocess_image(gen_img)

            # save the result to disk
            # imsave('./generated_inputs/' + str(i) + '_' + 'already_differ_' + str(label1) + '_' + str(label2) + '_' + str(label3) + '.png', gen_img_deprocessed)
            continue

        # if all label agrees
        orig_label = label1
        layer_name1, index1 = neuron_to_cover(model_layer_dict1)
        layer_name2, index2 = neuron_to_cover(model_layer_dict2)
        layer_name3, index3 = neuron_to_cover(model_layer_dict3)

        # construct joint loss function
        if target_model == 0:
            loss1 = -weight_diff * K.mean(model1.get_layer('before_softmax').output[..., orig_label])
            loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
            loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
        elif target_model == 1:
            loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
            loss2 = -weight_diff * K.mean(model2.get_layer('before_softmax').output[..., orig_label])
            loss3 = K.mean(model3.get_layer('before_softmax').output[..., orig_label])
        elif target_model == 2:
            loss1 = K.mean(model1.get_layer('before_softmax').output[..., orig_label])
            loss2 = K.mean(model2.get_layer('before_softmax').output[..., orig_label])
            loss3 = -weight_diff * K.mean(model3.get_layer('before_softmax').output[..., orig_label])
        loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
        loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
        loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
        layer_output = (loss1 + loss2 + loss3) + weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)

        # for adversarial image generation
        final_loss = K.mean(layer_output)

        # we compute the gradient of the input picture wrt this loss
        grads = normalize(K.gradients(final_loss, input_tensor)[0])

        # this function returns the loss and grads given the input picture
        iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])

        # we run gradient ascent for 20 steps
        for iters in range(grad_iterations):
            loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate([gen_img])
            if transformation == 'light':
                grads_value = constraint_light(grads_value)  # constraint the gradients value
            elif transformation == 'occl':
                grads_value = constraint_occl(grads_value, start_point, occlusion_size)  # constraint the gradients value
            elif transformation == 'blackout':
                grads_value = constraint_black(grads_value)  # constraint the gradients value

            gen_img += grads_value * step
            predictions1 = np.argmax(model1.predict(gen_img)[0])
            predictions2 = np.argmax(model2.predict(gen_img)[0])
            predictions3 = np.argmax(model3.predict(gen_img)[0])

            if not predictions1 == predictions2 == predictions3:
                update_coverage(gen_img, model1, model_layer_dict1, threshold)
                update_coverage(gen_img, model2, model_layer_dict2, threshold)
                update_coverage(gen_img, model3, model_layer_dict3, threshold)

                print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f' % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2), neuron_covered(model_layer_dict2)[2], len(model_layer_dict3), neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
                averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] + neuron_covered(model_layer_dict3)[0]) / float(neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] + neuron_covered(model_layer_dict3)[1])
                print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

                gen_img_deprocessed = deprocess_image(gen_img)
                orig_img_deprocessed = deprocess_image(orig_img)

                # save the result to disk
                imsave('./' + transformation + '/' + str(i) + '_' + transformation + '_' + str(predictions1) + '_' + str(predictions2) + '_' + str(predictions3) +'.png', gen_img_deprocessed)
                # imsave('./' + transformation + '/' + str(i) + '_' + transformation + '_' + str(predictions1) + '_' + str(predictions2) + '_' + str(predictions3) +'_orig.png', orig_img_deprocessed)
                
                total_origin += 1
                break
