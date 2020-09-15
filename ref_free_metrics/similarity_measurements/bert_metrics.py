import sys
import os
sys.path.append('../..')

from sentence_transformers import SentenceTransformer
from transformers import *
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords
from tqdm import tqdm
import torch

from resources import BASE_DIR, LANGUAGE
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from ref_free_metrics.similarity_scorer import parse_documents
from utils import get_human_score
from summariser.data_processor.human_score_reader import TacData
from summariser.utils.evaluator import evaluateReward, addResult
from summariser.utils.data_helpers import sent2stokens_wostop, sent2tokens_wostop

# from resources import bert_large_nli_mean_tokens_path, bert_large_uncased_path
# from resources import bert_large_nli_stsb_mean_tokens_path, albert_large_v2_path
# from resources import roberta_large_path, roberta_large_mnli_path, roberta_large_openai_detector_path
from resources import BERT_TYPE_PATH_DIC, SENT_TRANSFORMER_TYPE_PATH_DIC

def get_token_vecs(model, tokenizer, sent_list, gpu, bert_type):
    token_vecs = None

    for i in range(0,len(sent_list),5):
        ss = ' '.join(sent_list[i:i+5])
        if 'roberta' in bert_type: ss = '<s>' + ss + '</s>'
        tokens = tokenizer.encode(ss)
        tokens = torch.tensor(tokens).unsqueeze(0)
        if gpu:
            tokens = tokens.to('cuda')
        vv = model(tokens)[0][0].data.cpu().numpy()
        if token_vecs is None: token_vecs = vv
        else: token_vecs = np.vstack((token_vecs,vv))
    return  token_vecs


def get_bert_vec_similarity(model, tokenizer, all_sents, ref_num, gpu, bert_type):
    vec_matrix = []
    non_idx = []
    for i,doc in enumerate(all_sents):
        if len(doc) == 0:
            non_idx.append(i)
            #if 'albert' in bert_type: vec_matrix.append([0.]*4096)
            #else: vec_matrix.append([0.]*1024)
            vec_matrix.append([0.]*1024)
            continue
        token_vecs = get_token_vecs(model, tokenizer, doc, gpu, bert_type)
        vec_matrix.append(np.mean(token_vecs,axis=0))
    sim_matrix = cosine_similarity(vec_matrix[ref_num:], vec_matrix[:ref_num])
    scores = np.mean(sim_matrix,axis=1)
    return [ss if j+ref_num not in non_idx else None for j,ss in enumerate(scores)]


def run_bert_vec_metrics(year, ref_metric, bert_type, gpu=True):
    print('year: {}, ref_metric: {}, bert_type: {}'.format(year,ref_metric,bert_type))

    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    tacData = TacData(BASE_DIR,year)
    human = tacData.getHumanScores('summary', 'pyramid') # responsiveness or pyramid
    bert_large_nli_mean_tokens_path = SENT_TRANSFORMER_TYPE_PATH_DIC['bert_large_nli_mean_tokens']
    sbert_model = SentenceTransformer(bert_large_nli_mean_tokens_path)# 'bert-large-nli-mean-tokens' 'bert-large-nli-stsb-mean-tokens')

    if bert_type == 'bert':
        bert_large_uncased_path = BERT_TYPE_PATH_DIC['bert']
        berttokenizer = BertTokenizer.from_pretrained(bert_large_uncased_path) # 'bert-large-uncased'
        bertmodel = BertModel.from_pretrained(bert_large_uncased_path) # 'bert-large-uncased'
    elif bert_type == 'albert':
        albert_large_v2_path = BERT_TYPE_PATH_DIC['albert']
        berttokenizer = AlbertTokenizer.from_pretrained(albert_large_v2_path) # 'albert-large-v2'
        bertmodel = AlbertModel.from_pretrained(albert_large_v2_path) # 'albert-large-v2'
    else:
        assert 'roberta' in bert_type
        if 'nli' in bert_type:
            roberta_large_mnli_path = BERT_TYPE_PATH_DIC['roberta_large_mnli']
            berttokenizer = RobertaTokenizer.from_pretrained(roberta_large_mnli_path) # 'roberta-large-mnli'
            bertmodel = RobertaModel.from_pretrained(roberta_large_mnli_path) # 'roberta-large-mnli'
        elif 'openai' in bert_type:
            roberta_large_openai_detector_path = BERT_TYPE_PATH_DIC['roberta_large_openai_detector']
            berttokenizer = RobertaTokenizer.from_pretrained(roberta_large_openai_detector_path) # 'roberta-large-openai-detector'
            bertmodel = RobertaModel.from_pretrained(roberta_large_openai_detector_path) # 'roberta-large-openai-detector'
        else:
            roberta_large_path = BERT_TYPE_PATH_DIC['roberta_large']
            berttokenizer = RobertaTokenizer.from_pretrained(roberta_large_path) # 'roberta-large'
            bertmodel = RobertaModel.from_pretrained(roberta_large_path) # 'roberta-large'
    if gpu: bertmodel.to('cuda')

    mystopwords = set(stopwords.words(LANGUAGE))
    stemmer = PorterStemmer()

    all_results = {}

    # use mover-score to compute scores
    for topic,docs,models in corpus_reader(year):
        if '.B' in topic: continue
        print('\n=====Topic {}====='.format(topic))
        all_sents = []
        if ref_metric == 'true_ref':
            sent_info_dic, sent_vecs, sents_weights = parse_documents(models,sbert_model,ref_metric)
        else:
            sent_info_dic, sent_vecs, sents_weights = parse_documents(docs,sbert_model,ref_metric)
        ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k]>=0.1}
        ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)
        if len(ref_dic) >= 15:
            for rs in ref_sources:
                ref_sents = [ref_dic[k]['text'] for k in ref_dic if ref_dic[k]['doc']==rs]
                all_sents.append(ref_sents)
        else:
            ref_sents = [ref_dic[k]['text'] for k in ref_dic]
            all_sents.append(ref_sents)
            ref_sources = [1]
        for ss in peer_summaries[topic]:
            all_sents.append(ss[1])
        # compute word-vec-cosine score
        pss = get_bert_vec_similarity(bertmodel,berttokenizer,all_sents,len(ref_sources),gpu,bert_type)
        # compute correlation
        # changed by wchen to adopt to both Linux and Windows machine
        # (topic, ss[0].split('/')[-1]), human)
        hss = [get_human_score(topic, os.path.basename(ss[0]), human) for ss in peer_summaries[topic]]
        pseudo_scores, human_scores = [], []
        for i in range(len(pss)):
            if hss[i] is not None and pss[i] is not None:
                pseudo_scores.append(pss[i])
                human_scores.append(hss[i])
        assert len(human_scores) == len(pseudo_scores)
        if len(human_scores) < 2: continue
        results = evaluateReward(pseudo_scores,human_scores)
        addResult(all_results,results)
        for kk in results:
            print('{}:\t{}'.format(kk,results[kk]))

    print('\n=====ALL RESULTS=====')
    print('year: {}, ref_metric: {}, bert_type: bert'.format(year,ref_metric))
    for kk in all_results:
        if kk.startswith('p_'): continue
        print('{}:\tmax {:.4f}, min {:.4f}, mean {:.4f}, median {:.4f}, significant {} out of {}'.format(kk, np.max(all_results[kk]), np.min(all_results[kk]), np.mean(all_results[kk]), np.median(all_results[kk]), len([p for p in all_results['p_{}'.format(kk)] if p<0.05]), len(all_results[kk])))


if __name__ == '__main__':
    # '08', '09', '2010', '2011'
    year = '2010'
    run_bert_vec_metrics(year=year, ref_metric='top12_1', bert_type='bert', gpu=False)

