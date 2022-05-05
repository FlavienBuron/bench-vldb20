"""
Modified from Ivan Bongiorni,  https://github.com/IvanBongiorni
2022-03-17
"""

from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, RepeatVector, Conv1D, BatchNormalization,
    Concatenate, TimeDistributed, Dense, Flatten
)


USE_BATCHNORM = False


def build_vanilla_seq2seq(seq_len):
    """
    Implements a seq2seq RNN with Convolutional self attention. It keeps a canonical
    Encoder-Decoder structure: an Embedding layers receives the sequence of chars and
    learns a representation. This series is received by two different layers at the same time.
    First, an LSTM Encoder layer, whose output is repeated and sent to the Decoder. Second, a
    block of 1D Conv layers. Their kernel filters work as multi-head self attention layers.
    All their scores are pushed through a TanH gate that scales each score in the [-1,1] range.
    Both LSTM and Conv outputs are concatenated and sent to an LSTM Decoder, that processes
    the signal and sents it to Dense layers, performing the prediction for each step of the
    output series.
    """

    ## ENCODER
    encoder_input = Input((seq_len, 1))

    #LSTM block
    encoder_lstm = LSTM(units=64)(encoder_input)
    # RepeatVector repeats its input n times
    output_lstm = RepeatVector(seq_len)(encoder_lstm)

    # Conv block
    conv_1 = Conv1D(
        filters = 32,
        kernel_size = 3,
        activation = 'elu',
        kernel_initializer = 'he_normal',
        padding = 'same'
    )(encoder_input)
    if USE_BATCHNORM:
        conv_1 = BatchNormalization()(conv_1)

    conv_2 = Conv1D(
        filters = 32,
        kernel_size = 3,
        activation = 'elu',
        kernel_initializer = 'he_normal',
        padding = 'same'
    )(conv_1)
    if USE_BATCHNORM:
        conv_2 = BatchNormalization()(conv_2)

    conv_3 = Conv1D(
        filters = 32,
        kernel_size = 3,
        activation = 'elu',
        kernel_initializer = 'he_normal',
        padding = 'same'
    )(conv_2)
    if USE_BATCHNORM:
        conv_3 = BatchNormalization()(conv_3)

    conv_4 = Conv1D(
        filters = 32,
        kernel_size = 3,
        activation = 'elu',
        kernel_initializer = 'he_normal',
        padding = 'same'
    )(conv_3)
    if USE_BATCHNORM:
        conv_4 = BatchNormalization()(conv_4)

    # Concatenate LSTM and COnv Encoder outputs for Decoder LSTM layer
    encoder_output = Concatenate(axis = -1)([output_lstm, conv_2])

    decoder_lstm = LSTM(64, return_sequences = True)(encoder_output)

    decoder_output = TimeDistributed(
        Dense(units = 1,
                activation = 'linear',
                kernel_initializer= 'he_normal')
    )(decoder_lstm)

    seq2seq = Model(inputs = [encoder_input], outputs=[decoder_output])

    return seq2seq


def build_discriminator(seq_len):
    '''
    Discriminator is based on the Vanilla seq2seq Encoder. The Decoder is removed
    and a Dense layer is left instead to perform binary classification.
    '''

    ## ENCODER
    encoder_input = Input((seq_len, 1))

    #LSTM block
    encoder_lstm = LSTM(units=64)(encoder_input)
    # RepeatVector repeats its input n times
    output_lstm = RepeatVector(seq_len)(encoder_lstm)

    # Conv block
    conv_1 = Conv1D(
        filters = 32,
        kernel_size = 3,
        activation = 'elu',
        kernel_initializer = 'he_normal',
        padding = 'same'
    )(encoder_input)
    if USE_BATCHNORM:
        conv_1 = BatchNormalization()(conv_1)

    conv_2 = Conv1D(
        filters = 32,
        kernel_size = 3,
        activation = 'elu',
        kernel_initializer = 'he_normal',
        padding = 'same'
    )(conv_1)
    if USE_BATCHNORM:
        conv_2 = BatchNormalization()(conv_2)

    # Concatenate LSTM and Conv Encoder outputs and Flatten for Decoder LSTM layer
    encoder_output = Concatenate(axis=-1)([output_lstm, conv_2])
    # encoder_output = Flatten()(encoder_output)

    #Final layer for binary classification (real/fake)

    discriminator_output = TimeDistributed(
        Dense(
        units = 1,
        activation = 'sigmoid',
        kernel_initializer = 'he_normal'
    ))(encoder_output)

    Discriminator = Model(inputs=[encoder_input], outputs=[discriminator_output])

    return Discriminator


def build_GAN(seq_len):
    '''
    This is just a wrapper in case the model is trained as a GAN. It calls the vanilla
    seq2seq Generator, and build_discriminator() for the Discriminator model.
    '''

    generator = build_vanilla_seq2seq(seq_len)
    discriminator = build_discriminator(seq_len)
    return generator, discriminator