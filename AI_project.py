#Необходимые библиотеки.
from keras.utils import to_categorical
from keras import layers
from keras import models
from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from keras.preprocessing.image import ImageDataGenerator
#

#Количество эпох
EPOCH = 15;
#

#Загрузим изображения mnist с 60000 изображений в тренировочном датасете и 10000 в тестовом. 
(X_train, Y_train), (X_test, Y_test) = mnist.load_data();
#

#Отобразим структуру
print('X_train: ' + str(X_train.shape))
print('Y_train: ' + str(Y_train.shape))
print('X_test:  '  + str(X_test.shape))
print('Y_test:  '  + str(Y_test.shape))
#


#Преобразование датасетов в тензоры. Получили представление каждого изображения в виде массива 784 чисел от 0 до 255 - коэффициент серого. И изменение типа.
X_train = X_train.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
#


#Датагены
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(X_train)
#


                                                                #ОБЫЧНАЯ МНОГОСЛОЙНАЯ НЕЙРОННАЯ СЕТЬ
model0 = models.Sequential()
model0.add(layers.Flatten())
model0.add(layers.Dense(100, input_dim=784, activation='relu'))
model0.add(layers.Dense(200, activation='relu'))
model0.add(layers.Dense(10, activation='softmax'))
model0.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
hstr0 = model0.fit(datagen.flow(X_train, Y_train, batch_size=30), steps_per_epoch = len(X_train) // 30, epochs= EPOCH, validation_data=datagen.flow(X_test, Y_test, batch_size=30), validation_steps = len(X_test) / 30);
model0.save('C:\\Users\\user\\Downloads\\my_model0')





                                                                                    #СВЁРТОЧНАЯ НЕЙРОННАЯ СЕТЬ
model = models.Sequential()

#32 - глубина карты активации, ядра размера 3x3. На выходе из одного изображения размером 28x28 получаем 32 фильтра размерами 26x26. Шаг окна - 1.
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
#

#Слой пулинга для уменьшения разрешения карты активации в два раза. Входит - 26x26x32, выходит - 13x13x32. Пулинг максимальный (из окна размером 2x2
#выбирают максимальное значение)
model.add(layers.MaxPooling2D((2, 2)))
#

#Снова используем комбинацию свёртки и пулинга. Входит - 13x13x32, выходит - 5x5x64.
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
#

#Ещё один слой свёртки. Выход - 3x3x64
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#

#Преобразуем 3x3x64 в строку (1, 576)
model.add(layers.Flatten())
#

#Два дополнительных слоя.
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
#



#Обучение
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
hstr = model.fit(datagen.flow(X_train, Y_train, batch_size=30), steps_per_epoch = len(X_train) // 30, epochs= EPOCH, validation_data=datagen.flow(X_test, Y_test, batch_size=30), validation_steps = len(X_test) / 30);
#



                                                                #СВЁРТОЧНАЯ НЕЙРОННАЯ СЕТЬ С ИСПОЛЬЗОВАНИЕМ AVERAGE-ПУЛИНГА
model1 = models.Sequential()
model1.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model1.add(layers.AveragePooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.AveragePooling2D((2, 2)))
model1.add(layers.Conv2D(64, (3, 3), activation='relu'))
model1.add(layers.Flatten())
model1.add(layers.Dense(64, activation='relu'))
model1.add(layers.Dense(10, activation='softmax'))
model1.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
hstr1 = model1.fit(datagen.flow(X_train, Y_train, batch_size=30), steps_per_epoch = len(X_train) // 30, epochs= EPOCH, validation_data=datagen.flow(X_test, Y_test, batch_size=30), validation_steps = len(X_test) / 30);
#


#Сведения о модели
model.summary()
#

#Сохранение весов
model.save_weights('C:\\Users\\user\\Downloads\\my_model')
model.save_weights('C:\\Users\\user\\Downloads\\my_model1')
#


#Вывод каналов активации первого слоя
layer_outputs = [layer.output for layer in model.layers[:5]]
activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
activations = activation_model.predict(X_test)[0]
first_layer_activation = activations[0]
second_layer_activasion = activations[2]
third_layer_activasion = activations[4]

plt.figure()
plt.title('Каналы активации первого слоя для тестового изображения')
for i in range(32):  
    plt.subplot(4, 8, i+1)
    plt.imshow(first_layer_activation[:, :, i], cmap='viridis')
    plt.xlabel(i+1)
    plt.axis('off') 
#


plt.figure()
#Графики точности на осях accuracy/epochs
test_acc = hstr.history['val_accuracy']
test_acc0 = hstr0.history['val_accuracy']
test_acc1 = hstr1.history['val_accuracy']
loss = hstr.history['val_loss']
loss0 = hstr0.history['val_loss']
loss1 = hstr1.history['val_loss']
epochs = range(1, len(test_acc) + 1)

plt.plot(epochs, test_acc, 'bo', label='Сверточная сеть')
plt.plot(epochs, test_acc0, 'b', label='Обычная многослойная нейронная')
plt.plot(epochs, test_acc1, 'bv', label='Свёрточная с Average-Пулингом')
plt.title('Точность на тестовых выборках')
plt.legend()
#

#Графики потерь на осях accuracy/epochs
plt.figure()
plt.plot(epochs, loss, 'bo', label='Сверточная сеть')
plt.plot(epochs, loss0, 'b', label='Обычная многослойная нейронная')
plt.plot(epochs, loss1, 'bv', label='Свёрточная с Average-Пулингом')
plt.title('Потери на тестовых выборках')
plt.legend()
#


#Вывод цифр для примера
predictions = model.predict(X_test)
plt.figure()
for i in range(9):  
    plt.subplot(3, 3, i+1)
    plt.imshow(X_test[100+i], cmap=plt.get_cmap('gray'))
    plt.xlabel(str(np.argmax(predictions[i+100])))
plt.show()


#Загрузка весов
#model.load_weights('C:\\Users\\user\\Downloads\\my_model')
#



















