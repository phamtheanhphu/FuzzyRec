from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Multiply, Concatenate, Embedding, Reshape, LSTM
from tensorflow.keras.optimizers import Adam

from models import FuzzyLayer


def prepare_model(input_len, input_shape, num_classes, item_size, user_size):

    membership_layer_units = False
    dr_layer_1_units = 100
    dr_layer_2_units = 100
    fusion_dr_layer_units = 100
    learning_rate = 10 ** -3
    fuzz_membership_layer = []
    model_inputs = []

    for vector in range(input_len):
        model_inputs.append(Input(shape=(input_shape,)))
        # Membership Function layer
        fuzz_membership_layer.append(FuzzyLayer(membership_layer_units)(model_inputs[vector]))

    # Fuzzy Rule Layer
    rule_layer = Multiply()(fuzz_membership_layer)

    inp = Concatenate()(model_inputs)

    # Input DR Layers
    dr_layer_1 = Dense(dr_layer_1_units, activation='sigmoid')(inp)
    dr_layer_2 = Dense(dr_layer_2_units, activation='sigmoid')(dr_layer_1)

    # Input Embedding Layers

    item_emb_input = Input(shape=(input_shape,))
    model_inputs.append(item_emb_input)
    item_emb_layer_1 = Embedding(item_size, input_shape, mask_zero=True, name='item_embedding')(item_emb_input)
    item_emb_layer_2 = LSTM(input_shape)(item_emb_layer_1)
    item_emb_layer_3 = Reshape(target_shape=(input_shape,))(item_emb_layer_2)

    user_emb_input = Input(shape=(input_shape,))
    model_inputs.append(user_emb_input)
    user_emb_layer_1 = Embedding(user_size, input_shape, mask_zero=True, name='user_embedding')(user_emb_input)
    user_emb_layer_2 = LSTM(input_shape)(user_emb_layer_1)
    user_emb_layer_3 = Reshape(target_shape=(input_shape,))(user_emb_layer_2)

    emb_layer = Multiply()([item_emb_layer_3, user_emb_layer_3])

    # Fusion Layer
    fusion_layer = Concatenate()([rule_layer, dr_layer_2])

    # Fusion DR Layer
    fusion_dr_layer = Dense(fusion_dr_layer_units, activation='sigmoid')(fusion_layer)

    # Fusion Embedding Layer
    fusion_emb_layer = Dense(fusion_dr_layer_units, activation='sigmoid')(emb_layer)

    final_fusion_layer = Concatenate()([fusion_dr_layer, fusion_emb_layer])

    # Task Driven Layer
    out = Dense(num_classes, activation='softmax')(final_fusion_layer)

    model = Model(model_inputs, out)

    # compile model
    opt = Adam(learning_rate=learning_rate)
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])
    return model