import configargparse as cfargparse

ArgumentParser = cfargparse.ArgumentParser


def general_args(parser):
    """
    add some general options
    """
    group = parser.add_argument_group("general_args")
    group.add('--year', '-year', type=str, default='2010',
              choices=['08', '09', '2010', '2011'],
              help="The year of the TAC data")
    group.add('--human_metric', '-human_metric', type=str, default='pyramid',
              choices=['scu','pyramid','lingustic','responsiveness'],
              help="The selected human metric which is used to computer correlations with the designed auto metrics")
    group.add('--ref_summ', '-ref_summ', action='store_true',
              help="use the gold references")
    group.add('--evaluation_level', '-evaluation_level', type=str, default='summary',
              choices=['summary', 'micro', 'system', 'macro'],
              help="The level to evaluate the summarization systems")
    group.add('--device', '-device', type=str, default='cpu', choices=["cpu", "cuda"],
              help="The selected device to run the code")


def pseudo_ref_sim_metrics_args(parser):
    '''
    add some options for bert_metrics, elmo_metrics, js_metrics, sbert_score_metrics
    '''
    group = parser.add_argument_group("pseudo_ref_sim_metrics_args")
    group.add('--ref_metric', '-ref_metric', type=str, default='top12_1',
              help="TopN_th: Top 'N' sentences are selected as the reference and 'th' is the threshold.")
    group.add('--bert_type', '-bert_type', type=str, default='bert',
              choices=['bert', 'albert', 'roberta_large_mnli', 'roberta_large_openai_detector', 'roberta_large'],
              help="The pretrained model used to encoding the tokens of summaries.")
    group.add('--sent_transformer_type', '-sent_transformer_type', type=str, default='bert_large_nli_mean_tokens',
              choices=['bert_large_nli_mean_tokens', 'bert_large_nli_stsb_mean_tokens', 'roberta_large_nli_stsb_mean_tokens'],
              help="The pretrained sentence transformer used to encoding the sentences of the documents.")


def pseudo_ref_wmd_metrics_args(parser):
    '''
    add some options for bert_metrics, elmo_metrics, js_metrics, sbert_score_metrics
    '''
    group = parser.add_argument_group("pseudo_ref_wmd_metrics_args")
    group.add('--wmd_score_type', '-wmd_score_type', type=str, default='f1', choices=['f1', 'precision', 'recall'],
              help="The type of word-mover-distance score")
    group.add('--wmd_weight_type', '-wmd_weight_type', type=str, default='none', choices=['idf', 'none'],
              help="The type of word-mover-distance weight")