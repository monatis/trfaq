# trfaq

Google supported this work by providing Google Cloud credit. Thank you Google for supporting the open source! 🎉

## What is this?
At this repo, I'm releasing the training script and a full working inference example for my model [mys/bert-base-turkish-cased-nli-mean-faq-mnr](https://huggingface.co/mys/bert-base-turkish-cased-nli-mean-faq-mnr) published on HuggingFace. Please note that the training code at [`finetune_tf.py`](/finetune_tf.py) is a simplified version of the original, which is intended for educational purposes and not optimized for anything. However, it contains an implementation of the Multiple Negatives Symmetric Ranking loss, and you can use it in your own work. Additionally, I cleaned and filtered the Turkish subset of the [clips/mqa](https://huggingface.co/datasets/clips/mqa) dataset, as it contains lots of mis-encoded texts. You can download this cleaned dataset [here](https://storage.googleapis.com/mys-released-models/nlp-data/trfaq.csv).

## Model
This is a finetuned version of [mys/bert-base-turkish-cased-nli-mean](https://huggingface.co/) for FAQ retrieval, which is itself a finetuned version of [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased) for NLI. It maps questions & answers to 768 dimensional vectors to be used for FAQ-style chatbots and answer retrieval in question-answering pipelines. It was trained on the Turkish subset of [clips/mqa](https://huggingface.co/datasets/clips/mqa) dataset after some cleaning/ filtering and with a Multiple Negatives Symmetric Ranking loss. Before finetuning, I added two special tokens to the tokenizer (i.e., <Q> for questions and <A> for answers) and resized the model embeddings, so you need to prepend the relevant tokens to the sequences before feeding them into the model. Please have a look at [my accompanying repo](https://github.com/monatis/trfaq) to see how it was finetuned and how it can be used in inference. The following code snippet is an excerpt from the inference at the repo.

## Usage
see [`inference.py`](/inference.py) for a full working example.

```python
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
```

And the output is:
```shell
Merhaba
[{'answer': 'Merhaba, size nasıl yardımcı olabilirim?', 'score': '0.2931'}, {'answer': 'İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?', 'score': '0.2751'}, {'answer': 'Hayır, sadece Kurumsal Araç Kiralama operasyonları gerçekleştiriyoruz. Size başka nasıl yardımcı olabilirim?', 'score': '0.2200'}, {'answer': 'Evet, kurumsal araç kiralama hizmetleri sağlıyoruz. Size nasıl yardımcı olabilirim?', 'score': '0.2118'}]
---------------------
Nasılsın?
[{'answer': 'İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?', 'score': '0.2808'}, {'answer': 'Merhaba, size nasıl yardımcı olabilirim?', 'score': '0.2623'}, {'answer': 'Hayır, sadece Kurumsal Araç Kiralama operasyonları gerçekleştiriyoruz. Size başka nasıl yardımcı olabilirim?', 'score': '0.2320'}, {'answer': 'Evet, kurumsal araç kiralama hizmetleri sağlıyoruz. Size nasıl yardımcı olabilirim?', 'score': '0.2249'}]
---------------------
Bireysel araç kiralama yapıyor musunuz?
[{'answer': 'Hayır, sadece Kurumsal Araç Kiralama operasyonları gerçekleştiriyoruz. Size başka nasıl yardımcı olabilirim?', 'score': '0.2861'}, {'answer': 'Evet, kurumsal araç kiralama hizmetleri sağlıyoruz. Size nasıl yardımcı olabilirim?', 'score': '0.2768'}, {'answer': 'İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?', 'score': '0.2215'}, {'answer': 'Merhaba, size nasıl yardımcı olabilirim?', 'score': '0.2156'}]
---------------------
Kurumsal araç kiralama yapıyor musunuz?
[{'answer': 'Evet, kurumsal araç kiralama hizmetleri sağlıyoruz. Size nasıl yardımcı olabilirim?', 'score': '0.3060'}, {'answer': 'Hayır, sadece Kurumsal Araç Kiralama operasyonları gerçekleştiriyoruz. Size başka nasıl yardımcı olabilirim?', 'score': '0.2929'}, {'answer': 'İyiyim, teşekkür ederim. Size nasıl yardımcı olabilirim?', 'score': '0.2066'}, {'answer': 'Merhaba, size nasıl yardımcı olabilirim?', 'score': '0.1945'}]
---------------------
```