import sys
import os
sys.path.append('../..')

from my_sentence_transformers import SentenceTransformer
from nltk.stem import PorterStemmer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from nltk.corpus import stopwords

from resources import BASE_DIR, LANGUAGE
from summariser.data_processor.corpus_reader import CorpusReader
from summariser.data_processor.sys_summ_reader import PeerSummaryReader
from ref_free_metrics.similarity_scorer import parse_documents
from utils import get_human_score
from summariser.data_processor.human_score_reader import TacData
from summariser.utils.evaluator import evaluateReward, addResult

from resources import BERT_TYPE_PATH_DIC, SENT_TRANSFORMER_TYPE_PATH_DIC
import config
import json

def moverscore_style_pseudo_ref_gen(year, ref_metric, eval_level='summary',
                                    sent_transformer_type='bert_large_nli_stsb_mean_tokens', device='cpu'):
    '''
    the format of moverscore style input dataset file

    {
	'D0939':{
		'references':[{'text':['sum_sent0', 'sum_sent1',...,'sum_sentN1'],'id':'D0939-A.M.100.H.F'}*4]
		'annotations':[{
			'responsiveness':3.0,
			'pyr_mod_score':0.364, //references have no this key
			'text':['sum_sent0', 'sum_sent1',...,'sum_sentN2'],
			'pyr_score':0.364,
			'topic_id':'D0939-A',
			'summ_id':1
		    }*(55 systems + 4 references)]
	    }
    }
    '''
    print('year: {}, ref_metric: {}'.format(year,ref_metric))
    corpus_reader = CorpusReader(BASE_DIR)
    peer_summaries = PeerSummaryReader(BASE_DIR)(year)
    tacData = TacData(BASE_DIR,year)
    human_pyramid = tacData.getHumanScores(eval_level, 'pyramid') # responsiveness or pyramid
    human_respns = tacData.getHumanScores(eval_level, 'responsiveness') # responsiveness or pyramid
    # assert sent_transformer_type == 'bert_large_nli_stsb_mean_tokens'
    sent_transformer_path = SENT_TRANSFORMER_TYPE_PATH_DIC[sent_transformer_type]
    bert_model = SentenceTransformer(sent_transformer_path, device=device)  # 'bert-large-nli-stsb-mean-tokens')
    all_results = {}

    moverscore_dataset = {}

    # use mover-score to compute scores
    for topic,docs,models in corpus_reader(year):
        if '.B' in topic: continue
        print('\n=====Topic {}====='.format(topic))
        if ref_metric == 'true_ref':
            sent_info_dic, sent_vecs, sents_weights = parse_documents(models,bert_model,ref_metric)
        else:
            sent_info_dic, sent_vecs, sents_weights = parse_documents(docs,bert_model,ref_metric)
        ref_dic = {k:sent_info_dic[k] for k in sent_info_dic if sents_weights[k]>=0.1}
        print('extracted sent ratio', len(ref_dic)*1./len(sent_info_dic))
        ref_sources = set(ref_dic[k]['doc'] for k in ref_dic)

        # build a moversocre-style data instance
        topic_name = topic.split('.')[0]
        # build reference list
        references = [{'text': [], 'id': rs} for rs in ref_sources]
        for k in sorted(ref_dic.keys()):
            for idx in range(len(references)):
                if ref_dic[k]['doc'] == references[idx]['id']:
                    # the sentence order is consistent with the order of k
                    # therefore, no reordering is required
                    references[idx]['text'].append(ref_dic[k]['text'])
                    break

        # build annotation list
        topic_id = topic.replace('.', '-')
        annotations = []
        current_system_summs = peer_summaries[topic]
        for sys_tuple in current_system_summs:
            one_annot = {'topic_id': topic_id}
            file_name = sys_tuple[0]
            summ_id = file_name.split('.')[-1]
            one_annot['summ_id'] = summ_id
            # 'responsiveness'
            sys_respns = human_respns['topic{}_sum{}'.format(topic_id, summ_id)]
            one_annot['responsiveness'] = sys_respns
            # 'pyr_score'
            sys_pyramid = human_pyramid['topic{}_sum{}'.format(topic_id, summ_id)]
            one_annot['pyr_score'] = sys_pyramid
            # 'text'
            sys_text = sys_tuple[1]
            one_annot['text'] = sys_text

            annotations.append(one_annot)
        annotations = sorted(annotations, key=lambda i: float(i['summ_id']))

        # add the data instance
        moverscore_dataset[topic_name] = {'references': references, 'annotations': annotations}
    # save the built moverscore style dataset file
    folder_name = os.path.join('data', 'moverscore_style_files')
    os.makedirs(folder_name, exist_ok=True)
    if ref_metric == 'true_ref':
        file_name = os.path.join(folder_name, 'tac.{}.trueRef.mds.gen.resp-pyr'.format(year))
    else:
        file_name = os.path.join(folder_name, 'tac.{}.psdRef.{}.mds.gen.resp-pyr'.format(year, ref_metric))
    json.dump(moverscore_dataset, open(file_name, 'w'))



if __name__ == '__main__':
    # get the general configuration
    parser = config.ArgumentParser("pseudo_ref_file_generator.py")
    config.pseudo_ref_file_generator_args(parser)
    opt = parser.parse_args()
    print("\nMetric: pseudo_ref_file_generator.py")
    print("Configurations:", opt)
    # '08', '09', '2010', '2011'
    year = opt.year
    ref_summ = opt.ref_summ
    ref_metric = opt.ref_metric
    eval_level = opt.evaluation_level
    sent_transformer_type = opt.sent_transformer_type
    device = opt.device
    moverscore_style_pseudo_ref_gen(year=year, ref_metric=ref_metric, eval_level=eval_level,
                                    sent_transformer_type=sent_transformer_type,  device=device)
