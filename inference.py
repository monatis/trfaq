import tensorflow as tf
import numpy as np
from transformers import AutoTokenizer, TFAutoModel


model_name = 'mys/bert-base-turkish-cased-nli-mean-faq-mnr'

questions = [
    "Merhaba",
    "Nasılsın?",
    "Bireysel araç kiralama yapıyor musunuz?",
    "Kurumsal araç kiralama yapıyor musunuz?"
]

answers = [
    "Merhaba, size nasıl yardımcı olabilirim?",
    "İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?",
    "Hayır, sadece Kurumsal Araç Kiralama operasyonları gerçekleştiriyoruz. Size başka nasıl yardımcı olabilirim?",
    "Evet, kurumsal araç kiralama hizmetleri sağlıyoruz. Size nasıl yardımcı olabilirim?"
]


questions = ["<Q>" + q for q in questions]
answers = ["<A>" + a for a in answers]


tokenizer = AutoTokenizer.from_pretrained(model_name)
model = TFAutoModel.from_pretrained(model_name)

def answer_faq(model, tokenizer, questions, answers, return_similarities=False):
    q_len = len(questions)
    tokens = tokenizer(questions + answers, padding=True, return_tensors='tf')
    embs = model(**tokens)[0]

    attention_masks = tf.cast(tokens['attention_mask'], tf.float32)
    sample_length = tf.reduce_sum(attention_masks, axis=-1, keepdims=True)
    masked_embs = embs * tf.expand_dims(attention_masks, axis=-1)
    masked_embs = tf.reduce_sum(masked_embs, axis=1) / tf.cast(sample_length, tf.float32)
    a = tf.math.l2_normalize(masked_embs[:q_len, :], axis=1)
    b = tf.math.l2_normalize(masked_embs[q_len:, :], axis=1)

    similarities = tf.matmul(a, b, transpose_b=True)
        
    scores = tf.nn.softmax(similarities)
    results = list(zip(answers, scores.numpy().squeeze().tolist()))
    sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
    sorted_results = [{"answer": answer.replace("<A>", ""), "score": f"{score:.4f}"} for answer, score in sorted_results]
    return sorted_results



for question in questions:
    results = answer_faq(model, tokenizer, [question], answers)
    print(question.replace("<Q>", ""))
    print(results)
    print("---------------------")
