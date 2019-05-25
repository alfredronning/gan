import tensorflow as tf
import numpy as np
import datetime
import matplotlib.pyplot as plt
from tensorflow.contrib.learn.python.learn.datasets import mnist


tf.logging.set_verbosity(tf.logging.ERROR)
mnist = mnist.read_data_sets("MNIST_data/", one_hot=True)


def discriminator(x_image, reuse=False):
    if reuse:
        tf.get_variable_scope().reuse_variables()

    """first conv layer"""
    d_w1 = tf.get_variable("d_w1", shape=[3, 3, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.05))
    d_b1 = tf.get_variable("d_b1", shape=[32], initializer=tf.constant_initializer(0))
    d1 = tf.nn.conv2d(input=x_image, filter=d_w1, strides=[1, 1, 1, 1], padding='SAME')
    d1 = d1 + d_b1
    d1 = tf.nn.relu(d1)
    d1 = tf.nn.avg_pool(d1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    """second conv layer"""
    d_w2 = tf.get_variable("d_w2", shape=[3, 3, 32, 64], initializer=tf.truncated_normal_initializer(stddev=0.05))
    d_b2 = tf.get_variable("d_b2", shape=[64], initializer=tf.constant_initializer(0))
    d2 = tf.nn.conv2d(input=d1, filter=d_w2, strides=[1, 1, 1, 1], padding='SAME')
    d2 = d2 + d_b2
    d2 = tf.nn.relu(d2)
    d2 = tf.nn.avg_pool(d2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    """first fully connected layer"""
    d_w3 = tf.get_variable("d_w3", shape=[7 * 7 * 64, 1024], initializer=tf.truncated_normal_initializer(stddev=0.05))
    d_b3 = tf.get_variable("d_b3", shape=[1024], initializer=tf.constant_initializer(0))
    d3 = tf.reshape(d2, [-1, 7 * 7 * 64])
    d3 = tf.matmul(d3, d_w3) + d_b3
    d3 = tf.nn.relu(d3)

    """final output layer"""
    d_w4 = tf.get_variable("d_w4", shape=[1024, 1], initializer=tf.truncated_normal_initializer(stddev=0.05))
    d_b4 = tf.get_variable("d_b4", shape=[1], initializer=tf.constant_initializer(0))
    d4 = tf.matmul(d3, d_w4) + d_b4
    return d4


def generator(batch_size, z_dim):
    """start with random noice vector of length z_dim to generate non reterministic output images"""
    z = tf.truncated_normal([batch_size, z_dim], mean=0, stddev=1, name="z")
    g_w1 = tf.get_variable("g_w1", [z_dim, 3136], initializer=tf.truncated_normal_initializer(stddev=0.05))
    g_b1 = tf.get_variable("g_b1", [3136], initializer=tf.constant_initializer(0))
    g1 = tf.matmul(z, g_w1) + g_b1
    g1 = tf.reshape(g1, [-1, 56, 56, 1])
    g1 = tf.contrib.layers.batch_norm(g1, epsilon=0.0001, scope='bn1')
    g1 = tf.nn.relu(g1)

    """first conv layer"""
    g_w2 = tf.get_variable("g_w2", [3, 3, 1, 32], initializer=tf.truncated_normal_initializer(stddev=0.05))
    g_b2 = tf.get_variable("g_b2", [32], initializer=tf.constant_initializer(0))
    g2 = tf.nn.conv2d(g1, filter=g_w2, strides=[1, 1, 1, 1], padding='SAME')
    g2 = g2 + g_b2
    g2 = tf.contrib.layers.batch_norm(g2, epsilon=0.0001, scope='bn2')
    g2 = tf.nn.relu(g2)

    """second conv layer"""
    g_w3 = tf.get_variable("g_w3", [3, 3, 32, 16], initializer=tf.truncated_normal_initializer(stddev=0.05))
    g_b3 = tf.get_variable("g_b3", [16], initializer=tf.constant_initializer(0))
    g3 = tf.nn.conv2d(g2, filter=g_w3, strides=[1, 1, 1, 1], padding='SAME')
    g3 = g3 + g_b3
    g3 = tf.contrib.layers.batch_norm(g3, epsilon=0.0001, scope='bn3')
    g3 = tf.nn.relu(g3)

    """final conv layer generating the output"""
    g_w4 = tf.get_variable("g_w4", [3, 3, 16, 1], initializer=tf.truncated_normal_initializer(stddev=0.05))
    g_b4 = tf.get_variable("b_w4", [1], initializer=tf.constant_initializer(0))
    g4 = tf.nn.conv2d(g3, filter=g_w4, strides=[1, 2, 2, 1], padding='SAME')
    g4 = g4 + g_b4

    g4 = tf.nn.sigmoid(g4)
    return g4


def main(testing_model=False, display_images=False):
    sess = tf.Session()

    batch_size = 50
    z_dimensions = 100
    learning_rate = 0.0001
    epochs = 50000

    x_placeholder = tf.placeholder("float", shape=[None, 28, 28, 1], name='r_placeholder')

    """generated images"""
    g = generator(batch_size, z_dimensions)
    """the discriminator prediction for real and fake images"""
    dr = discriminator(x_placeholder)
    df = discriminator(g, reuse=True)
    """the calculated loss for the generator and the discriminator """
    g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=df, labels=tf.ones_like(df)))
    d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=dr, labels=tf.ones_like(dr)))
    d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=df, labels=tf.zeros_like(df)))


    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if 'd_' in var.name]
    g_vars = [var for var in t_vars if 'g_' in var.name]

    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        d_trainer_real = tf.train.AdamOptimizer(learning_rate).minimize(d_loss_real, var_list=d_vars)
        d_trainer_fake = tf.train.AdamOptimizer(learning_rate).minimize(d_loss_fake, var_list=d_vars)
        g_trainer = tf.train.AdamOptimizer(learning_rate).minimize(g_loss, var_list=g_vars)

    saver = tf.train.Saver()
    if testing_model:
        pass

    else:
        """running the session"""

        sess.run(tf.global_variables_initializer())

        generator_loss = 0
        discriminator_loss_real, discriminator_loss_fake = 1, 1

        g_count, d_real_count, d_fake_count = 0, 0, 0
        for i in range(epochs):
            real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
            if discriminator_loss_fake > 0.6:
                """train the discriminator on fake images"""
                _, discriminator_loss_fake, discriminator_loss_real, generator_loss = \
                    sess.run([d_trainer_fake, d_loss_fake, d_loss_real, g_loss], {x_placeholder: real_image_batch})
                d_fake_count += 1
            if generator_loss > 0.5:
                """train the generator"""
                _, discriminator_loss_fake, discriminator_loss_real, generator_loss = \
                    sess.run([g_trainer, d_loss_fake, d_loss_real, g_loss], {x_placeholder: real_image_batch})
                g_count += 1
            if discriminator_loss_real > 0.45:
                """train the discriminator on real images"""
                _, discriminator_loss_fake, discriminator_loss_real, generator_loss = \
                    sess.run([d_trainer_real, d_loss_fake, d_loss_real, g_loss], {x_placeholder: real_image_batch})
                d_real_count += 1

            if i%10 == 0:
                real_image_batch = mnist.train.next_batch(batch_size)[0].reshape([batch_size, 28, 28, 1])
                g_count, d_real_count, d_fake_count = 0, 0, 0

            if i%100 == 0:
                print("Epoch:", i)

            if i%5000 == 0:
                saver.save(sess, 'saves/model.ckpt', global_step=i)
                if display_images:
                    images = sess.run(generator(3, z_dimensions))
                    discriminator_prediction = sess.run(discriminator(x_placeholder), {x_placeholder: images})
                    for j in range(3):
                        print("Discriminator result: ", discriminator_prediction[j][0])
                        img = images[j, :, :, 0]
                        plt.imshow(img.reshape([28, 28]), cmap='Greys')
                        plt.show()


if __name__ == '__main__':
    main()

