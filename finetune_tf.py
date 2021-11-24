import random

import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer, TFAutoModel
import tqdm

model_name = 'mys/bert-base-turkish-cased-nli-mean'
batch_size = 32
epochs = 3

data = pd.read_csv('trfaq.csv')
questions = data["question"].tolist()
answers = data["answer"].tolist()

questions = ["<Q>" + q for q in questions]
answers = ["<A>" + a for a in answers]

tokenizer = AutoTokenizer.from_pretrained(model_name, additional_special_tokens=['<Q>', '<A>'])
model = TFAutoModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

optimizer = tf.keras.optimizers.Adam(lr=1e-5)

def get_embeddings(model, tokenizer, texts, training=False):
    tokens = tokenizer(texts, padding=True, truncation=True, return_tensors='tf', max_length=80)
    embs = model(**tokens, training=training)[0]

    attention_masks = tf.cast(tokens['attention_mask'], tf.float32)
    sample_length = tf.reduce_sum(attention_masks, axis=-1, keepdims=True)
    masked_embs = embs * tf.expand_dims(attention_masks, axis=-1)
    masked_embs = tf.reduce_sum(masked_embs, axis=1) / tf.cast(sample_length, tf.float32)

    return masked_embs



class MultipleNegativesRankingLoss(tf.keras.layers.Layer):
    def __init__(self, scaling=20.0, symmetric=False):
        super(MultipleNegativesRankingLoss, self).__init__()
        self.scaling = scaling
        self.symmetric = symmetric
        self.scce = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, a, b):
        assert len(tf.shape(a)) == 2, "a and b embeddings should be of rank 2"

        n, dim = tf.shape(a)

        a_norm = tf.math.l2_normalize(a, axis=1)
        b_norm = tf.math.l2_normalize(b, axis=1)
        logits = tf.matmul(a_norm, b_norm, transpose_b=True)
        labels = tf.range(n)
        loss = self.scce(labels, logits)

        if self.symmetric:
            loss = (loss + self.scce(labels, tf.transpose(logits, [1, 0]))) / 2

        return loss
        

mnr = MultipleNegativesRankingLoss(symmetric=True)


for epoch in tqdm.trange(epochs, desc='Epoch'):
    for offset in tqdm.trange(0, len(questions), batch_size, desc='Step'):
        qa = list(zip(questions, answers))
        random.shuffle(qa)
        questions, answers = zip(*qa)
        questions, answers = list(questions), list(answers)

        q_batch = questions[offset:offset+batch_size]
        a_batch = answers[offset:offset+batch_size]
        with tf.GradientTape() as tape:
            anchor_embs = get_embeddings(model, tokenizer, q_batch, True)
            positive_embs = get_embeddings(model, tokenizer, a_batch, True)
            loss = mnr(anchor_embs, positive_embs)
         
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        if offset % int(batch_size * 200) == 0:
            print("Loss at step %d: %.4f" % ((offset+batch_size) // batch_size, float(loss)))



model_name = model_name.split('/')[1] + "-faq-mnr"
tokenizer.save_pretrained(model_name)
model.save_pretrained(model_name)

