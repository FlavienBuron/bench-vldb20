import argparse
import time

import numpy as np
import tensorflow as tf

import data_loader
import model

SHUFFLE = True
BATCH_SIZE = 2
LOSS_WEIGHT = 0.05

parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str)
parser.add_argument('--output', type=str)
parser.add_argument('--runtime', default=0, type=int)
args = parser.parse_args()

def validation(generator, X_val, mask):

    generator_imputation = generator(X_val)
    val_loss = tf.reduce_mean(tf.math.abs(tf.math.multiply(generator_imputation, mask) - tf.math.multiply(X_val, mask)))
    generator_imputation = X_val * mask + (1 - mask) * generator_imputation
    print('\t\tValidation Loss: {}\n'.format(val_loss))\

    return generator_imputation


# @tf.function
def train(
        generator, discriminator, generator_optimizer, discriminator_optimizer, cross_entropy,
        X_batch, Y_batch, real_example, real_example_mask, mask, w):

    with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
        generator_imputation = generator(X_batch)
        x_hat = tf.math.multiply(mask, X_batch) + tf.math.multiply((1-mask), generator_imputation)

        discriminator_guess_fakes = discriminator(x_hat)
        discriminator_guess_reals = discriminator(real_example)

        # Generator loss. Original compared imputed missing value to their real values
        # Since we don't have the real values, we compare imputed non-missing values to their original values
        g_loss_mae = tf.reduce_sum(tf.math.abs(
            tf.math.multiply(generator_imputation, mask) - tf.math.multiply(Y_batch, mask)
        ))/tf.reduce_sum(mask + 1e-8)
        # Generator Discriminator loss over the imputed real value ?
        # g_loss is BCE of generated data over real data
        g_loss_gan = cross_entropy(tf.ones_like(discriminator_guess_fakes), discriminator_guess_fakes)
        g_loss_gan = tf.expand_dims(g_loss_gan, 2)
        g_loss_gan = tf.reduce_sum((1-mask)*g_loss_gan)/tf.reduce_sum((1-mask)+ 1e-8)

        # magnitude of GAN loss to be adjusted
        generator_current_loss = g_loss_mae + (g_loss_gan * w)

        # Discriminator loss - label smoothing
        loss_fakes = cross_entropy(
            tf.random.uniform(shape=tf.shape(discriminator_guess_fakes), minval=0.0, maxval=0.2),
            discriminator_guess_fakes
        )
        loss_reals = cross_entropy(
            tf.random.uniform(shape=tf.shape(discriminator_guess_reals), minval=0.8, maxval=1),
            discriminator_guess_reals
        )
        real_example_bce = tf.reduce_sum(tf.expand_dims(
            cross_entropy(tf.ones_like(discriminator_guess_fakes), discriminator_guess_fakes), 2)*mask)/tf.reduce_sum(mask + 1e-8)

        fake_bce = tf.reduce_sum(tf.expand_dims(
            cross_entropy(tf.zeros_like(discriminator_guess_fakes), discriminator_guess_fakes), 2)*(1-mask))/tf.reduce_sum((1-mask)+ 1e-8)

        discriminator_current_loss = real_example_bce + fake_bce


    generator_gradient = generator_tape.gradient(generator_current_loss, generator.trainable_variables)
    discriminator_gradient = discriminator_tape.gradient(discriminator_current_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradient, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradient, discriminator.trainable_variables))

    return generator_current_loss, discriminator_current_loss


def run(input, output, runtime):

    dataX, dataM = data_loader.data_loader(input)

    _, seq_len, ndims = dataX.shape
    BATCH_SIZE = dataX.shape[0]//1
    print("Batch size: ", BATCH_SIZE)

    Imputer, Discriminator = model.build_GAN(seq_len, ndims)
    X_val = tf.cast(dataX, tf.float32)
    M_val = dataM

    # this works for both G and D
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits = False, reduction=tf.keras.losses.Reduction.NONE)

    generator_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)
    discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate = 1e-3)

    start_time = time.time()

    for epoch in range(100):
        # Shuffle data by shuffling row index
        if SHUFFLE:
            shuffle_idx = np.random.choice(len(dataX), len(dataX), replace=False)
            dataX = dataX[ shuffle_idx ]
            dataM = dataM[ shuffle_idx ]

        for iteration in range(dataX.shape[0] // BATCH_SIZE):

            start =iteration*BATCH_SIZE
            X_batch = tf.convert_to_tensor(dataX[start:start+BATCH_SIZE], dtype=tf.float32)
            Y_batch = tf.identity(X_batch)
            M_batch = tf.convert_to_tensor(dataM[start:start+BATCH_SIZE], dtype=tf.float32)

            # take data that is not in the batch as real example for discriminator training
            real_example = np.concatenate([dataX[0:start], dataX[start+BATCH_SIZE:]])
            real_example_mask = np.concatenate([dataM[0:start], dataM[start+BATCH_SIZE:]])
            idx = np.random.choice(len(real_example), len(real_example), replace=False)
            real_example = real_example[idx]
            real_example_mask = real_example_mask[idx]
            real_example = tf.convert_to_tensor(real_example[:BATCH_SIZE],dtype=tf.float32)
            real_example_mask = tf.convert_to_tensor(real_example_mask[:BATCH_SIZE], dtype=tf.float32)

            generator_current_loss, discriminator_current_loss = train(
                Imputer, Discriminator, generator_optimizer, discriminator_optimizer, cross_entropy,
                X_batch, Y_batch, real_example, real_example_mask, M_batch, LOSS_WEIGHT)

            if iteration % 50 == 0:

                # Check Imputer's plain loss on training example

                generator_imputation = Imputer(X_batch)
                x_hat = tf.math.multiply(M_batch, X_batch) + tf.math.multiply((1-M_batch), generator_imputation)
                train_loss = tf.reduce_sum(
                    tf.math.abs(tf.math.multiply(generator_imputation, M_batch)- tf.math.multiply(Y_batch, M_batch)))/tf.reduce_sum(M_batch)
                discriminator_guess_reals = Discriminator(real_example)
                discriminator_guess_fakes = Discriminator(x_hat)
                d_acc_real = tf.reduce_sum(tf.expand_dims(
                            tf.keras.metrics.binary_accuracy(
                                tf.ones_like(discriminator_guess_fakes),
                                discriminator_guess_fakes),2)*M_batch)/tf.reduce_sum((M_batch)+ 1e-8)
                d_acc_fake = tf.reduce_mean(tf.expand_dims(
                            tf.keras.metrics.binary_accuracy(
                                tf.zeros_like(discriminator_guess_fakes),
                                discriminator_guess_fakes),2)*(1-M_batch))/tf.reduce_sum((1-M_batch)+ 1e-8)


                print(
                    f'\r{epoch}.{iteration} Generator Loss: {generator_current_loss:.8f}, '
                    f'Discriminator Loss: {discriminator_current_loss:.8f}, '
                    f'Discriminator Accuracy (reals, fakes): ({d_acc_real:.8f}, {d_acc_fake:.8f}),'
                    f' Imputation Loss: {train_loss:.8f}',
                    end='\r')

    out_matrix = validation(Imputer, X_val, M_val)

    end_time = time.time()
    exec_time = (end_time - start_time) * 1000 * 1000

    print("Time pGAN: ", exec_time)

    if runtime > 0:
        np.savetxt(output, np.array([exec_time]))
    else:
        np.savetxt(output, np.squeeze(out_matrix, 0))


if __name__ == '__main__':
    input = args.input
    output = args.output
    runtime = args.runtime
    run(input, output, runtime)