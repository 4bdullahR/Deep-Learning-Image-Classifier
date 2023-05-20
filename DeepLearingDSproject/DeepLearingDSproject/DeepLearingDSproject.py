from tensorflow import keras
from keras import datasets, models, layers
import numpy as np 
import cv2
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import matplotlib.pyplot as plt
import os.path

(traing_images, traing_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
traing_images, testing_images = traing_images / 255, testing_images / 255

class_names = ['PLANE', 'CAR', 'BIRD', 'CAT', 'DEER', 'DOG', 'FROG', 'HORSE', 'SHIP', 'TRUCK']

if not os.path.isdir('neural_network_image_classifier'):
    print("Pretrained model file doesn't exists, traiing starts now")

    My_Image_classifcation_model = models.Sequential()
    My_Image_classifcation_model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)))
    My_Image_classifcation_model.add(layers.MaxPool2D((2,2)))
    My_Image_classifcation_model.add(layers.Conv2D(64, (3,3), activation='relu'))
    My_Image_classifcation_model.add(layers.MaxPool2D((2,2)))
    My_Image_classifcation_model.add(layers.Conv2D(64,(3,3), activation='relu'))
    My_Image_classifcation_model.add(layers.Flatten())
    My_Image_classifcation_model.add(layers.Dense(64,activation='relu'))
    My_Image_classifcation_model.add(layers.Dense(10, activation='softmax'))

    My_Image_classifcation_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    My_Image_classifcation_model.fit(traing_images, traing_labels, epochs=10, validation_data=(testing_images, testing_labels))

    loss , accuracy = My_Image_classifcation_model.evaluate(testing_images,testing_labels)
    print(f"Loss = {loss} \nAccracy: {accuracy}")

    My_Image_classifcation_model.save('neural_network_image_classifier')



My_Image_classifcation_model = models.load_model('neural_network_image_classifier')

from tkinter import *
top = Tk()
top.title('Test the model')
top.geometry('300x750')
Label(top,text='Test these samples from the Internet!',font=('Helvatical bold',13)).pack()

buttons = []
for i in class_names:
    button = Button(top, text=i, height=4, width=20, command=lambda i=i: testModel(class_names.index(i)))
    button.pack()
    buttons.append(button)
    
def testModel(user_input):

    img = cv2.imread(f'{class_names[user_input].lower()}.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.imshow(img, cmap=plt.cm.binary)

    predection = My_Image_classifcation_model.predict(np.array([img]) / 255)
    index = np.argmax(predection)

    print('\n-------------------------------------------------')
    print(f'IT IS A {class_names[index]}, no? \nThe neurel network model predicts that the given photo is a {class_names[index]}')
    print('-------------------------------------------------\n')

    plt.gcf().canvas.set_window_title('neurel network model prediction')
    plt.xlabel(f"it is a {class_names[index]}, isn't it?")
    plt.show()

mainloop()