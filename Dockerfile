FROM huggingface/transformers-tensorflow-cpu:4.9.1

RUN pip install --no-cache-dir flask==1.1.2 google-api-python-client google-auth-httplib2 google-auth-oauthlib && python3 -c "from transformers import AutoTokenizer, TFAutoModel; AutoTokenizer.from_pretrained('mys/bert-base-turkish-cased-nli-mean-faq-mnr'); TFAutoModel.from_pretrained('mys/bert-base-turkish-cased-nli-mean-faq-mnr')"

COPY sheets-bot/ /app/

CMD ["python3", "/app/app.py"]