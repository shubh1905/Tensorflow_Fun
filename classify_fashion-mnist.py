from __future__ import absolute_import, division, print_function,unicode_literals

#tensrorflow and tf.keras

import tensorflow as tf
import keras
from tensorflow.keras import layers
import keras.backend as k
k.clear_session()

#helper libraries
import numpy as np
import matplotlib.pyplot as plt 

class MyModel(tf.keras.Model):
    def __init__(self,num_classes=10,image_shape=(28,28)):
        super(MyModel,self).__init__(name='my_model')
        self.num_classes=num_classes
        #define layers
        self.flatten_1=layers.Flatten(input_shape=image_shape)
        self.dense_1=layers.Dense(128,activation='relu')
        self.dense_2=layers.Dense(num_classes,activation='softmax')
        
    def call(self,inputs):
        f1=self.flatten_1(inputs)
        f2=self.dense_1(f1)
        f3=self.dense_2(f2)
        return f3
    
    
    def compute_output_shape(self,input_shape):
        shape=tf.TensorShape(input_shape).as_list()
        shape[-1]=self.num_classes
        return tf.TensorShape(shape)
    



print(tf.VERSION)
print(tf.keras.__version__)
#getting the dataset
fashion_mnist=keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)=fashion_mnist.load_data()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print(train_images.shape)
len(train_labels)
print(test_images.shape)
len(test_images)

#preprocess data
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
#plt.show()

train_images=train_images/255.0
test_images=test_images/255.0

plt.figure(figsize=(10,10))
for i in range(25):
	plt.subplot(5,5,i+1)
	plt.xticks([])
	plt.yticks([])
	plt.grid(False)
	plt.imshow(train_images[i],cmap=plt.cm.binary)
	plt.xlabel(class_names[train_labels[i]])
#plt.show()

#building the model-higher api
'''
model=tf.keras.Sequential([
	layers.Flatten(input_shape=(28,28)),
	layers.Dense(128,activation=tf.keras.activations.relu),
	layers.Dense(10,activation=tf.keras.activations.softmax)
])
'''

#fucntional api    
inputs=tf.keras.Input(shape=(28,28))
l1=layers.Flatten(input_shape=(28,28))(inputs)
l2=layers.Dense(128,activation='relu')(l1)
l3=layers.Dense(10,activation='softmax')(l2)

#instantiate the model
model=tf.keras.Model(inputs=inputs,outputs=l3)
#model from class file
#model=MyModel(num_classes=10)
#compiling the model
model.compile(optimizer=tf.train.RMSPropOptimizer(0.01),loss=tf.keras.losses.sparse_categorical_crossentropy,metrics=['accuracy'])

#training the model
model.fit(train_images,train_labels,epochs=5,verbose=1,batch_size=32)

#evaluating the model
#model.evaluate(test_images,test_labels)

#predicting with model
'''result=model.predict(train_images)
for i in range(20):
    plt.figure(i)
    plt.imshow(train_images[i])
    plt.xlabel(train_labels[i]+"--"+class_names[np.argmax(result[i])])
'''