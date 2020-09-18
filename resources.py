import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) #os.path.abspath('.')
ROUGE_DIR = os.path.join(BASE_DIR,'rouge','ROUGE-RELEASE-1.5.5/') #do not delete the '/' in the end

SUMMARY_LENGTH = 100
LANGUAGE = 'english'

# add by wchen, sentence_transformers type
bert_large_nli_mean_tokens_path = os.path.join(BASE_DIR, '..', 'pytorch_transformers', 'bert-large-nli-mean-tokens')
bert_large_nli_stsb_mean_tokens_path = os.path.join(BASE_DIR, '..', 'pytorch_transformers', 'bert-large-nli-stsb-mean-tokens')
roberta_large_nli_stsb_mean_tokens_path = os.path.join(BASE_DIR, '..', 'pytorch_transformers', 'roberta-large-nli-stsb-mean-tokens')
# bert type
bert_large_uncased_path = os.path.join(BASE_DIR, '..', 'pytorch_transformers', 'bert-large-uncased')
albert_large_v2_path = os.path.join(BASE_DIR, '..', 'pytorch_transformers', 'albert-large-v2')
roberta_large_path = os.path.join(BASE_DIR, '..', 'pytorch_transformers', 'roberta-large')
roberta_large_mnli_path = os.path.join(BASE_DIR, '..', 'pytorch_transformers', 'roberta-large-mnli')
roberta_large_openai_detector_path = os.path.join(BASE_DIR, '..', 'pytorch_transformers', 'roberta-large-openai-detector')

BERT_TYPE_PATH_DIC = {'bert': bert_large_uncased_path,
                      'albert': albert_large_v2_path,
                      'roberta_large_mnli': roberta_large_mnli_path,
                      'roberta_large_openai_detector': roberta_large_openai_detector_path,
                      'roberta_large': roberta_large_path}

SENT_TRANSFORMER_TYPE_PATH_DIC = {'bert_large_nli_mean_tokens': bert_large_nli_mean_tokens_path,
                                  'bert_large_nli_stsb_mean_tokens': bert_large_nli_stsb_mean_tokens_path,
                                  'roberta_large_nli_stsb_mean_tokens': roberta_large_nli_stsb_mean_tokens_path}