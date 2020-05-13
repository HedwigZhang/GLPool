import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model
import pickle, os, time, cv2
import numpy as np
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
from shuffleNet_v2 import ShuffleNet_V2
from imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def train():
    # load network
    model = ShuffleNet_V2()
    ######## load h5 file
    model.load_weights(r".../cifar100/shuffleNet-v2/shufflenetv2_glp.h5")
    ###### load data
    img_path = '.../cifar100/shuffleNet-v2/bird.png'
    #img = image.load_img(img_path, target_size=(32, 32))
    img = cv2.imread(img_path) # 用cv2加载原始图像
    img = cv2.resize(img, (32,32), interpolation=cv2.INTER_NEAREST)
    img1 = img / 255
    # cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # cv2.waitKey (0) 
    # cv2.destroyAllWindows()
    x = image.img_to_array(img1)
    x = np.expand_dims(x, axis=0)
    #x = tf.transpose(x, [0, 3, 1, 2])
    #x = preprocess_input(x)
    #test_pic = load_data()
    ######## You should known the layers name
    # base_model = Model(inputs = model.input, outputs = model.get_layer('dw1_dwconv').output)
    # f1 = base_model.predict(x)
    preds = model.predict(x)
    # #print('Predicted:', decode_predictions(preds, top=3)[0])
    class_idx = np.argmax(preds[0])
    print(class_idx)
    
    class_output = model.output[:, class_idx]
    last_conv_layer = model.get_layer("last_conv")
    # gap_weights = model.get_layer("glp_dwconv")
 
    # grads = K.gradients(class_output, gap_weights.output)[0]
    # iterate = K.function([model.input],[grads,last_conv_layer.output[0]])
    # pooled_grads_value, conv_layer_output_value = iterate([x])
    # pooled_grads_value = np.squeeze(pooled_grads_value,axis=0)
    # for i in range(1024):
    #     conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
 
    # heatmap = np.mean(conv_layer_output_value, axis=-1)
    # heatmap = np.maximum(heatmap,0)#relu激活。
    # heatmap /= np.max(heatmap)
    # #
    # # img = cv2.imread(img_path)
    # # img = cv2.resize(img, dsize=(32,32),interpolation=cv2.INTER_NEAREST)
    # # img = img_to_array(image)
    # heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
    # heatmap = np.uint8(255 * heatmap)
    # heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    # superimposed_img = cv2.addWeighted(img,0.6,heatmap,0.4,0)
    # cv2.imwrite(".../CAM/panda/panda_cam.jpg", superimposed_img)
    # cv2.waitKey (0) 
    # cv2.destroyAllWindows()

    grads = K.gradients(class_output,last_conv_layer.output)[0]
    pooled_grads = K.mean(grads,axis=(0,1,2))
    iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate([x])
    for i in range(1024):
        conv_layer_output_value[:,:,i] *= pooled_grads_value[i]
 
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap,0)
    heatmap /= np.max(heatmap)
    cv2.imwrite(".../cifar100/shuffleNet-v2/cam/bird_heatmap.jpg", heatmap, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.waitKey (0) 
    cv2.destroyAllWindows()
 
    img = cv2.imread(img_path)
    img = cv2.resize(img,dsize=(32,32),interpolation=cv2.INTER_NEAREST)
    #  img = img_to_array(image)
    heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(img,0.6,heatmap,0.4,0)
    cv2.imwrite(".../cifar100/shuffleNet-v2/cam/bird_cam.jpg", superimposed_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    cv2.waitKey (0) 
    cv2.destroyAllWindows()

    # last_conv_layer = model.get_layer("glp_bn")
    # grads = K.gradients(predict_output, last_conv_layer.output)[0]
    # pooled_grads = K.mean(grads, axis=(0, 1, 2))
    # iterate = K.function([model.layers[0].input], [pooled_grads, last_conv_layer.output[0]])
    # pooled_grads_value, conv_layer_output_value = iterate([x])

    # for i in range(1024):
    #     conv_layer_output_value[:, :, i] *= pooled_grads_value[i] # 将特征图数组的每个通道乘以这个通道对大象类别重要程度
 
    # heatmap = np.mean(conv_layer_output_value, axis=-1)
    # heatmap = np.maximum(heatmap, 0) # heatmap与0比较，取其大者
    # hea_max = np.max(heatmap)
    # print(hea_max)
    # heatmap = heatmap / hea_max

    # # plt.matshow(heatmap)
    # # plt.show()
    # plt.imshow(heatmap, cmap='gray')
    #     #plt.imshow(show_img, cmap='colormap')
    #     # 图片存储地址
    # plt.savefig(".../CAM/panda/heatmap.jpg")
    # plt.close()
    # # cv2.imwrite(".../panda_glp_bn/fea_"+ str(i)+".jpg", show_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    # # cv2.waitKey (0) 
    # # cv2.destroyAllWindows()
    # img = cv2.imread(img_path) # 用cv2加载原始图像
    # img = cv2.resize(img, (32,32))
    # heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0])) # 将热力图的大小调整为与原始图像相同
    # heatmap = np.uint8(255 * heatmap) # 将热力图转换为RGB格式
    # heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # 将热力图应用于原始图像
    # superimposed_img = heatmap * 0.4 + img # 这里的0.4是热力图强度因子
    # cv2.imwrite(".../test_google/glp/CAM/panda/panda_cam.jpg", superimposed_img)
    # cv2.waitKey (0) 
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    train()
