import json
import os
import csv
from datetime import datetime
import pandas as pd
from collections import Counter
import numpy as np

from tqdm.auto import tqdm
import tensorflow as tf
from transformers import BertTokenizer


model = tf.keras.models.load_model('/Users/georgevatalis/Desktop/Portfolio/Toxicity/sentiment_model_20k')

tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

def prepare_data(input_text, tokenizer):
    token = tokenizer.encode_plus(
        input_text,
        max_length=256, 
        truncation=True, 
        padding='max_length', 
        add_special_tokens=True,
        return_tensors='tf'
    )
    return {
        'input_ids': tf.cast(token.input_ids, tf.float64),
        'attention_mask': tf.cast(token.attention_mask, tf.float64)
    }

def make_prediction(processed_data, model, classes):
    probs = model.predict(processed_data)[0]
    output = classes[probs.argmax()]
    prob_output = 'Reddit ' + str(probs[0]) + '\n' + 'Parler ' + str(probs[1]) 
    return output, prob_output

