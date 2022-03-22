from keras.preprocessing.sequence import pad_sequences


def get_padded_data(data, pad_len):
    pd = pad_sequences(data, maxlen=pad_len, dtype="long", truncating="post", padding="post")
    return pd
