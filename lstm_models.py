from keras.models import load_model
from pathlib import Path
import numpy as np
import json

# constants
maxlen = 40

def process_string(input_sentence, chars):
    """
    Removes all invalid chars, lowercases, and pads with space

    Processes input strings given to the API get request and ensures it only
    contains characters that the text-gen model is familiar with, and ensures
    it is of proper length by padding the beginning with spaces.

    Parameters:
        input_sentence (str): the input string given to the API call
        chars (str[]): the list of valid characters

    Returns:
        processed_seq (str): the string of proper length and valid chars
    """
    # lowercases all the character
    input_sentence = input_sentence.lower()

    # Removes invalid chars
    processed_seq = ""
    for c in input_sentence:
        if c in chars:
            processed_seq += c
    # truncate to
    processed_seq = " "*40 + processed_seq
    return processed_seq[-40:]

def sample(preds, temperature=1.0):
    """
    Given softmax vector preds, samples from it and returns a one-hot vector

    Parameters:
        preds (float[]): a softmax vector of arbitrary len
        temperature: likelihood to deviate from the highest value indices

    Returns:
        sampled (float[]): a one-hot vector sampled from preds
    """
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

def lstm_generate (input_sentence, lstm_model, charset, temp=0.5, length=10):
    """
    Generates text of given length from the seed text using the given lstm model

    Parameters:
        input_sentence (str): the input string given to the API call
        lstm_model (keras model): the actual model
        charset (char[]): an arr of chars the lstm trained on (in order of vectors)
        temp (float): the temperature or likelihood to deviate from training set
        length (int): the number of characters the model will generate

    Returns:
        final_text (str): the input_sentence with the generated chars appended
    """
    temperature = temp
    # cleans up the input_sentence
    seed = process_string(input_sentence, indices_char)
    final_text = input_sentence

    # generate one character at a time, adding to final_text each iteration
    for iters in range(length):

        # encodes the `maxlen=40` one-hot vectors to input to the lstm model
        x_pred = np.zeros((1, maxlen, len(indices_char)))
        for i, char in enumerate(seed):
            x_pred[0, i, charset.index(char)] = 1.

        # Run the model to predict on the input seq x_pred. Output is softmax
        softmax_pred = model.predict(x_pred, verbose=0)[0]
        next_index = sample(softmax_pred, temperature)
        next_char = charset[next_index]

        # Append the next char to the seed and repeat
        seed = seed[1:] + next_char
        final_text += next_char

    return final_text

def load_models ():
    """
    Returns a dict mapping model names to their instance and charset

    For ex, the key "AiW-5.h5" would yeild tuple (keras model, AiW charset)

    Returns:
        models (dict): maps string to model, charset tuples
    """

    models = {}

    # Loads Alice in Wonderland models
    aiw_chars_path = Path('./lstm-models/AiW/AiW-charset.json')
    with open(chars_path, 'r') as f:
        data = f.read()
    aiw_charset = json.loads(data)
    models['AiW-1'] = (load_model('./lstm-models/AiW/AiW-1.h5'), aiw_charset)
    models['AiW-5'] = (load_model('./lstm-models/AiW/AiW-5.h5'), aiw_charset)
    models['AiW-20'] = (load_model('./lstm-models/AiW/AiW-20.h5'), aiw_charset)
    models['AiW-40'] = (load_model('./lstm-models/AiW/AiW-40.h5'), aiw_charset)

    return models

"""
Example Usage:

# loads all models
models = load_models()

# select one model set
model, charset = d["AiW-5"]

# simulate being given a sentence
input_sentence = "Once upon a time, there was a kid detective named "

# send to the lstm generation function
lstm_generate(input_sentence, model, charset, temp=1, length=40)
"""
