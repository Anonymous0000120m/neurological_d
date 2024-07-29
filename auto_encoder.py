import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import logging
import cv2
import os

# 设置日志记录
logging.basicConfig(filename='gan_autoencoder.log', level=logging.INFO, 
                    format='%(asctime)s:%(levelname)s:%(message)s')

# 定义 Autoencoder 类
class Autoencoder:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.autoencoder = self.build_model()

    def build_model(self):
        input_img = layers.Input(shape=self.input_shape)
        x = layers.Flatten()(input_img)
        x = layers.Dense(128, activation='relu')(x)
        encoded = layers.Dense(64, activation='relu')(x)

        x = layers.Dense(128, activation='relu')(encoded)
        x = layers.Dense(np.prod(self.input_shape), activation='sigmoid')(x)
        decoded = layers.Reshape(self.input_shape)(x)

        autoencoder = models.Model(input_img, decoded)
        autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
        logging.info("自编码器模型创建成功")
        return autoencoder

    def train(self, x_train, epochs=50, batch_size=256):
        logging.info("开始训练自编码器模型")
        self.autoencoder.fit(x_train, x_train, epochs=epochs, batch_size=batch_size, shuffle=True)

    def save_model(self, model_file='autoencoder_model.h5'):
        self.autoencoder.save(model_file)
        logging.info(f"模型已保存到 {model_file}")

# 定义 GAN 类
class GAN:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
        self.gan = self.build_gan()

    def build_generator(self):
        model = models.Sequential()
        model.add(layers.Dense(128, activation='relu', input_dim=100))
        model.add(layers.Dense(np.prod(self.input_shape), activation='sigmoid'))
        model.add(layers.Reshape(self.input_shape))
        logging.info("生成器模型创建成功")
        return model

    def build_discriminator(self):
        model = models.Sequential()
        model.add(layers.Flatten(input_shape=self.input_shape))
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(1, activation='sigmoid'))
        logging.info("判别器模型创建成功")
        return model

    def build_gan(self):
        self.discriminator.trainable = False
        model = models.Sequential([self.generator, self.discriminator])
        model.compile(optimizer='adam', loss='binary_crossentropy')
        logging.info("GAN 模型创建成功")
        return model

    def train(self, x_train, epochs=10000, batch_size=128):
        for epoch in range(epochs):
            idx = np.random.randint(0, x_train.shape[0], batch_size)
            real_imgs = x_train[idx]
            noise = np.random.normal(0, 1, (batch_size, 100))
            fake_imgs = self.generator.predict(noise)

            real_labels = np.ones((batch_size, 1))
            fake_labels = np.zeros((batch_size, 1))

            d_loss_real = self.discriminator.train_on_batch(real_imgs, real_labels)
            d_loss_fake = self.discriminator.train_on_batch(fake_imgs, fake_labels)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = self.gan.train_on_batch(noise, real_labels)

            logging.info(f"Epoch: {epoch}, D Loss: {d_loss[0]}, G Loss: {g_loss}")

# 定义 CNN 类
class CNN:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = self.build_model()

    def build_model(self):
        model = models.Sequential()
        model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(128, activation='relu'))
        model.add(layers.Dense(self.num_classes, activation='softmax'))
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        logging.info("CNN 模型创建成功")
        return model

    def train(self, x_train, y_train, epochs=10, batch_size=32):
        self.model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    def predict(self, x_test):
        return self.model.predict(x_test)

# 图像处理函数
def load_and_preprocess_images(image_paths, target_size=(28, 28)):
    images = []
    for img_path in image_paths:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, target_size)
        img = img.astype('float32') / 255.0
        images.append(img)
    return np.expand_dims(np.array(images), axis=-1)

# 示例使用
if __name__ == "__main__":
    # 图像文件夹路径
    image_folder = "path_to_your_images"
    image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder) if fname.endswith('.png')]

    # 加载和预处理图像
    x_train = load_and_preprocess_images(image_paths)

    # 1. 训练自编码器
    autoencoder = Autoencoder(input_shape=(28, 28, 1))
    autoencoder.train(x_train, epochs=10)
    autoencoder.save_model()

    # 2. 训练 GAN
    gan = GAN(input_shape=(28, 28, 1))
    gan.train(x_train, epochs=1000, batch_size=128)

    # 3. 训练 CNN
    num_classes = 10  # 根据实际数据集进行调整
    cnn = CNN(input_shape=(28, 28, 1), num_classes=num_classes)
    # 需要分割 y_train，如果 Image folder 不含标签则需要另行准备
    y_train = np.random.randint(num_classes, size=x_train.shape[0])  # 示例标签
    cnn.train(x_train, y_train, epochs=10)

    # 4. 预测
    predictions = cnn.predict(x_train)
    predicted_classes = np.argmax(predictions, axis=1)
    print("预测的类别：", predicted_classes)
