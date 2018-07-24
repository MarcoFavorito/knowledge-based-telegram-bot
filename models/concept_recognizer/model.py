from keras import Input
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LambdaCallback
from keras.engine import Model
from keras.layers import Embedding, Bidirectional, LSTM, TimeDistributed, Dense, Dropout, Activation, add
from keras.optimizers import RMSprop
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split


def get_model():
    main_input = Input(shape=(self.MAX_LENGTH,), name="main_input")

    embedded = Embedding(len(self.w2idx), self.embedding_size, mask_zero=True)(main_input)

    relation_input = Input(shape=(len(c.RELATIONS),), name="relation_input")
    relation_matrix = Dense(self.embedding_size, input_shape=(None, len(c.RELATIONS)))(relation_input)

    merge_l = add([embedded, relation_matrix])

    blstm_l1 = Bidirectional(LSTM(
        self.lstm_size,
        return_sequences=True)
    )(merge_l)

    blstm_l2 = Bidirectional(LSTM(
        self.lstm_size,
        return_sequences=True)
    )(blstm_l1)

    output_l = TimeDistributed(Dense(len(c.entity_tags), use_bias=True))(blstm_l2)
    dropout_out_l = Dropout(self.dropout)(output_l)
    activation = Activation(activation="softmax")(dropout_out_l)

    model = Model(inputs=[main_input, relation_input], outputs=[activation])

    # learning rate = 0.001
    opt = RMSprop()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.summary()
    return model
