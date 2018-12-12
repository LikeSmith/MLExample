"""
MNISTIntegerRecognition.py

Example code for network to recognize MNIST image set as integers
"""

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import gzip
import matplotlib.pyplot as plt

# Network Specifications:
h1_size = 20
h2_size = 15

# Training Specificatiions:
learning_rate = 0.0001
n_epochs = 50
batch_size = 128

if __name__ == '__main__':
    # Load MNIST Data set
    print('Load Training Data...')
    with gzip.open('../../data/MNIST/MNIST_train_imgs.idx3-ubyte.gz', 'rb') as file:
        mn = int.from_bytes(file.read(4), byteorder='big')
        if mn != 2051:
            print('Trainiing Data file corrupt, Magic Number is %d'%mn)
            exit(1)
        
        train_size = int.from_bytes(file.read(4), byteorder='big')
        train_n_rows = int.from_bytes(file.read(4), byteorder='big')
        train_n_cols = int.from_bytes(file.read(4), byteorder='big')
                
        train_dat = np.zeros((train_size, train_n_rows, train_n_cols))
                
        for i in tqdm(range(train_size)):
            for j in range(train_n_rows):
                for k in range(train_n_cols):
                    train_dat[i, j, k] = float(int.from_bytes(file.read(1), byteorder='big'))/255.0
    
    print('Loading Training Labels...')
    with gzip.open('../../data/MNIST/MNIST_train_labels.idx1-ubyte.gz', 'rb') as file:
        mn = int.from_bytes(file.read(4), byteorder='big')
        if mn != 2049:
            print('Training Labels file corrupt, Magic Number is %d'%mn)
            exit(1)
                    
        train_size_l = int.from_bytes(file.read(4), byteorder='big')
        train_labels = np.zeros((train_size_l, 10))
                
        for i in tqdm(range(train_size_l)):
            train_labels[i, int.from_bytes(file.read(1), byteorder='big')] = 1.0
                            
    print('Loading Validation Data...')   
    with gzip.open('../../data/MNIST/MNIST_valid_imgs.idx3-ubyte.gz', 'rb') as file:
        mn = int.from_bytes(file.read(4), byteorder='big')
        if mn != 2051:
            print('Validation Data file corrupt, Magic Number is %d'%mn)
            exit(1)
                
        valid_size = int.from_bytes(file.read(4), byteorder='big')
        valid_n_rows = int.from_bytes(file.read(4), byteorder='big')
        valid_n_cols = int.from_bytes(file.read(4), byteorder='big')
                
        valid_dat = np.zeros((valid_size, valid_n_rows, valid_n_cols))
                
        for i in tqdm(range(valid_size)):
            for j in range(valid_n_rows):
                for k in range(valid_n_cols):
                    valid_dat[i, j, k] = float(int.from_bytes(file.read(1), byteorder='big'))/255.0
                            
    print('Loading Validation Labels')
    with gzip.open('../../data/MNIST/MNIST_valid_labels.idx1-ubyte.gz', 'rb') as file:
        mn = int.from_bytes(file.read(4), byteorder='big')
        if mn != 2049:
            print('Validation Labels file corrupt, Magic Number is %d'%mn)
            exit(1)
                    
        valid_size_l = int.from_bytes(file.read(4), byteorder='big')
        valid_labels = np.zeros((valid_size_l, 10))
                
        for i in tqdm(range(valid_size_l)):
            valid_labels[i, int.from_bytes(file.read(1), byteorder='big')] = 1.0
                            
    assert valid_size == valid_size_l
    assert train_size == train_size_l
    assert valid_n_rows == train_n_rows
    assert valid_n_cols == train_n_cols
    
    n_cols = train_n_cols
    n_rows = train_n_rows
    
    # Build Compute graph
    with tf.variable_scope('MNIST_Integer_Recognition') as scope:
        # Specify inputs
        img_in = tf.placeholder('float', [None, n_rows, n_cols], 'input_img')
        label_true = tf.placeholder('float', [None, 10], 'true_label')
        
        # Build network
        img_flat = tf.layers.flatten(img_in, 'flatten')
        h1 = tf.layers.dense(img_flat, h1_size, tf.nn.relu, name='h1')
        h2 = tf.layers.dense(h1, h2_size, tf.nn.relu, name='h2')
        label_pred = tf.layers.dense(h2, 10, tf.nn.softmax, name='label_pred')
        
        # Calculate loss
        loss = tf.reduce_mean(tf.squared_difference(label_pred, label_true))
        
        # Setup Optimizer
        trainer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = trainer.minimize(loss)
    
    # Train Network:
    n_batches = int(np.ceil(train_size/batch_size))
    losses = np.zeros((n_epochs, 2))
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        print('Training...')
        for epoch in range(n_epochs):
            print('Epoch: %d'%(epoch+1))
            
            # shuffle data
            perm = np.random.permutation(train_size)
            
            batch_bar = tqdm(range(n_batches))
            batch_bar.set_description('Ave. Loss: N/A')
            for batch in batch_bar:
                # get batch
                batch_imgs = train_dat[perm[batch*batch_size:min((batch+1)*batch_size, train_size)], :, :]
                batch_labels = train_labels[perm[batch*batch_size:min((batch+1)*batch_size, train_size)], :]
                
                # apply training step, calculate loss
                res = sess.run([train_op, loss], feed_dict={img_in:batch_imgs, label_true:batch_labels})
                
                losses[epoch, 0] += res[1]
                
                batch_bar.set_description('Ave. Loss: %f'%(losses[epoch, 0]/(batch+1)))
                    
            # calculate validation
            losses[epoch, 1] = sess.run(loss, feed_dict={img_in:valid_dat, label_true:valid_labels})
            losses[epoch, 0] /= n_batches
            
            print('Epoch %d done.  Ave. Loss: %f, Val. Loss: %f'%(epoch, losses[epoch, 0], losses[epoch, 1]))
            
        # get predictions for validation set
        pred = sess.run(label_pred, feed_dict={img_in:valid_dat})
    
    # evaluate results
    plt.figure()
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Average Loss', 'Validation Loss'])
    
    best = np.argmax(pred, axis=0)
    
    plt.figure()
    for i in range(10):
        plt.subplot(2, 5, i+1)
        plt.imshow(valid_dat[best[i], :, :], cmap='gray')
        plt.title('Pred: %d'%(i,))
        
    plt.show()
    
    print('done!')