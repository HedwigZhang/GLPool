##############################
# the main function, which will show the training process
##############################
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler, History
from tensorflow.contrib.tpu.python.tpu import keras_support

from shuffleNet_v2 import ShuffleNet_V2
from load_data import get_train_data, get_valid_data
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.backend import set_session
import pickle, os, time

graph = tf.get_default_graph()
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def lr_scheduler(epoch):
    lr = 0.01
    if epoch >= 20: lr /= 10.0
    if epoch >= 30: lr /= 10.0
    if epoch >= 40: lr /= 10.0
    return lr

# def plot_confusion_matrix(cm, classes,
#                           title='Confusion matrix',
#                           cmap=plt.cm.jet):
#     cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
#     plt.imshow(cm, interpolation='nearest', cmap=cmap)
#     plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(classes))
#     plt.xticks(tick_marks, classes, rotation=45)
#     plt.yticks(tick_marks, classes)
#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, '{:.2f}'.format(cm[i, j]), horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")
#     plt.tight_layout()
#     plt.ylabel('True label')
#     plt.xlabel('Predicted label')
#     plt.show()

# # 显示混淆矩阵
# def plot_confuse(model, x_val, y_val):
#     predictions = model.predict_classes(x_val)
#     truelabel = y_val.argmax(axis=-1)   # 将one-hot转化为label
#     conf_mat = confusion_matrix(y_true=truelabel, y_pred=predictions)
#     plt.figure()
#     plot_confusion_matrix(conf_mat, range(np.max(truelabel)+1))

def train():
    X_train, y_train = get_train_data()
    X_test, y_test = get_valid_data()
    # data generater
    train_gen = ImageDataGenerator(rescale=1.0/255, horizontal_flip=True, width_shift_range=4.0/32.0, height_shift_range=4.0/32.0)
    test_gen = ImageDataGenerator(rescale=1.0/255)
    # train_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, zca_whitening=True, horizontal_flip=True,)
    # test_gen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True, zca_whitening=True)
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    # load network
    model = ShuffleNet_V2()
    #model.compile(Adam(0.001), "categorical_crossentropy", ["accuracy"])
    model.compile(SGD(0.01, momentum = 0.9, nesterov=True), "categorical_crossentropy", ["acc"])
    #model.compile(SGD(0.01, momentum = 0.9), "categorical_crossentropy", ["acc", "top_k_categorical_accuracy"])
    model.summary()
    
    # set GPU
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    session = tf.Session(config = config)
    #session = tf.Session()
    set_session(session)

    # set
    batch_size = 128
    scheduler = LearningRateScheduler(lr_scheduler)
    hist = History()

    start_time = time.time()
    #model.fit_generator(train_gen.flow(X_train, y_train, batch_size, shuffle=True),
    #                    steps_per_epoch=X_train.shape[0]//batch_size,
    #                    validation_data=test_gen.flow(X_test, y_test, batch_size, shuffle=False),
    #                    validation_steps=X_test.shape[0]//batch_size,
    #                    callbacks=[scheduler, hist], max_queue_size=5, epochs=100)
    model.fit_generator(train_gen.flow(X_train, y_train, batch_size, shuffle=True),
                        steps_per_epoch=X_train.shape[0]//batch_size,
                        validation_data=test_gen.flow(X_test, y_test, batch_size, shuffle=False),
                        validation_steps=X_test.shape[0]//batch_size,
                        callbacks=[scheduler, hist], max_queue_size=5, epochs=50)

    elapsed = time.time() - start_time
    print('training time', elapsed)
    
    history = hist.history
    history["elapsed"] = elapsed

    with open("shuffle_v2_002_glp.pkl", "wb") as fp:
        pickle.dump(history, fp)

    #model.save('shufflenetv2_Pdw.h5')
    
if __name__ == "__main__":
    train()