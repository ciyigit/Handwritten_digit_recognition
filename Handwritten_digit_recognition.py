import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
from PIL import Image

mnist = input_data.read_data_sets("./MNIST", one_hot=True)#one_hot有且只有一个1（0,0,1,0,0,0,0,0,0,0）表示2
#input_layer
xs=tf.placeholder(dtype=tf.float32,shape=[None,784],name='xs')
#lable
ys=tf.placeholder(dtype=tf.float32,shape=[None,10],name='ys')


def read_data(path):
    image = cv2.imread(path,cv2.IMREAD_GRAYSCALE)

    process_image =cv2.resize(image,dsize=(28,28))
    process_image = np.resize(process_image/255.0,new_shape=(1,784))

    return image , process_image

def predict(image_path,sess,output):
    image , process_image =read_data(image_path)
    #预测结果
    result = sess.run(output,feed_dict={xs:process_image})
    result = np.argmax(result,1)
    print('the prection is',result)
    cv2.putText(image,'the prediction is {}'.format(result),(25,100),cv2.FONT_HERSHEY_COMPLEX,2,(255,0,255),2)
    plt.imshow(image, cmap='gray')
    plt.show()
    


def add_layer(inputs, in_size, out_size, activation_function=None ):
    
    Weights = tf.Variable(tf.truncated_normal([in_size, out_size]))
    tf.add_to_collection('losses',tf.contrib.layers.l2_regularizer(0.001)(Weights))
    biases = tf.Variable(tf.zeros([1, out_size])+0.01 )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def reconstruct_image():
    #create 10 folders
    for i in range(10):
        if not os.path.exists('./{}'.format(i)):
            os.makedirs('./{}'.format(i))

    for i in range(int(mnist.test.num_examples/1)):
        #x_data[0]:[[784]]
        x_data,y_data  =mnist.test.next_batch(1)

        img=Image.fromarray(np.reshape(np.array(x_data[0]*255,dtype='uint8'),newshape=(28,28)))
        dir = np.argmax(y_data[0])
        img.save('./{}/{}.bmp'.format(dir,i))
# t
#reconstruct_image()
#start

hidden_layer1 = add_layer(xs, 784, 300, activation_function=tf.nn.relu)
hidden_layer2 =add_layer(hidden_layer1,300,150,activation_function=tf.nn.sigmoid)
out_data = add_layer(hidden_layer2, 150, 10, activation_function=None)
cem = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=ys,logits=out_data))
loss=cem + tf.add_n(tf.get_collection('losses'))
prediction = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(ys,1),tf.argmax(out_data,1)),tf.float32))
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
saver = tf.train.Saver()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    if not os.path.exists('checkpoint'):
        for i in range(100):
            for _ in range(int(mnist.train.num_examples / 200)):
                X_train,y_train=mnist.train.next_batch(200)
                sess.run(train_step, feed_dict={xs: X_train, ys: y_train})
            print(i,'loss value is:',i,sess.run(loss,feed_dict={xs: X_train, ys: y_train}))
            print('prediction value is::',sess.run(prediction,feed_dict={xs:mnist.test.images,ys:mnist.test.labels}))
        saver.save(sess,'./mnist-ckpt')
    else:
        saver.restore(sess,'./mnist-ckpt')
        predict('./88.png',sess,out_data)    
