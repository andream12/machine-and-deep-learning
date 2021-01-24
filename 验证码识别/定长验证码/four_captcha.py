import tensorflow as tf
import pandas as pd
import random, os
import numpy as np
from PIL import Image
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt

# 定义一些常量
image_shape = (60, 120)
learning_rate = 0.001
batch_size = 256
number = list(map(lambda x: str(x), range(10)))
alphabet = list(map(lambda x: chr(x), range(ord('a'), ord('z') + 1)))
ALPHABET = list(map(lambda x: chr(x), range(ord('A'), ord('Z') + 1)))
alphabet = number + alphabet + ALPHABET
alphabet_num = len(alphabet)

# 生成定长验证码的文本，即标签
def random_captcha_text(char_set=None, captcha_size=4):
    text = []
    if char_set is None:
        char_set = alphabet
    for i in range(captcha_size):
        c = random.choice(char_set)
        text.append(c)
    return text

# 生成验证码图片,返回图片和标签
def gen_img_and_text(width=120, height=60, char_set=None):
    image = ImageCaptcha(width=width, height=height)
    captha_text = random_captcha_text(char_set)  # 生成验证码的文本
    captha_text = "".join(captha_text)  # 拼接文本
    # 随机字符和背景的颜色
    w1, w2, w3 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    g1, g2, g3 = random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)
    # 生成验证码
    captha_image = image.create_captcha_image(captha_text, (w1, w2, w3), (g1, g2, g3))
    captha_image = np.array(captha_image)
    return captha_image, captha_text


# 生成一个训练batch
def get_next_batch(batch_size=batch_size):
    # (batch_size,120,60)
    inputs = np.zeros([batch_size, image_shape[0], image_shape[1], 3])
    label = np.zeros([batch_size, 4, alphabet_num])
    label_text = []
    for i in range(batch_size):
        # 生成不定长度的字串
        captha_image, captha_text = gen_img_and_text(char_set=alphabet)
        inputs[i, :] = captha_image
        # 标签
        for j, c in enumerate(captha_text):
            label[i, j][alphabet.index(c)] = 1
        label_text.append(captha_text)
    label_text = np.array(label_text)
    return inputs, label, label_text

# 标签转化为文本
def label2text(label):
    text = []
    for l in label:
        index = np.argmax(l)
        text.append(alphabet[index])
    return "".join(text)


# 模型
def crack_captcha_cnn():
    model = tf.keras.Sequential()

    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(5, 5), activation="relu", padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation="relu", padding="same"))
    model.add(tf.keras.layers.Conv2D(filters=128, kernel_size=(5, 5), activation="relu", padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D((2, 2), strides=2))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4 * 62))
    model.add(tf.keras.layers.Reshape([4, 62]))

    model.add(tf.keras.layers.Softmax())
    model.build(input_shape=[None, 60, 120, 3])

    return model

# 训练
def train():
    try:
        model = tf.keras.models.load_model('captcha_model.h5')
    except Exception as e:
        model = crack_captcha_cnn()
    # 装配
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3), metrics=['accuracy'], loss='categorical_crossentropy')
    for i in range(2000):
        print(f"========================={i + 1}========================")
        batch_x, batch_y, text = get_next_batch(512)
        val_x, val_y, text = get_next_batch(50)
        model.fit(batch_x, batch_y, epochs=4, validation_data=(val_x, val_y), validation_freq=1)
        # 预测新样本集
        test_num, count = 50, 0
        test_x, test_y, test_text = get_next_batch(test_num)
        pred = model.predict(test_x)
        for j in range(pred.shape[0]):
            content = label2text(pred[j])
            if content == test_text[j]:
                count += 1
            print("预测值为：", content, "====>", "真实值为：", test_text[j], content == test_text[j])
        print('准确率为：', count / test_num)
        # 每50次保存一次模型
        if i % 50 == 0 and i > 0:
            model.save('captcha_model.h5')
            print('模型保存成功！！！')

def predict(test_num=200):
    model = tf.keras.models.load_model('captcha_model.h5')
    # 预测新样本集
    test_num, count = test_num, 0
    test_x, test_y, test_text = get_next_batch(test_num)
    pred = model.predict(test_x)
    for j in range(pred.shape[0]):
        content = label2text(pred[j])
        if content == test_text[j]:
            count += 1
        print("预测值为：", content, "====>", "真实值为：", test_text[j], content == test_text[j])
    print('共破解{}个，其中正确{}个，正确率为{}%'.format(test_num, count, count / test_num * 100))

if __name__ == '__main__':
    # train()
    predict()
