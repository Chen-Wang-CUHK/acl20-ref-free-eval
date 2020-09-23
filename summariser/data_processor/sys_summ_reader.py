import os
import re
from collections import OrderedDict

from resources import BASE_DIR

class PeerSummaryReader:
    def __init__(self,base_path):
        self.base_path = base_path

    def __call__(self,year):
        # changed by wchen
        assert '08' == year or '09' == year or '2010' == year or '2011' == year
        if year in ['08', '09']:
            data_path = os.path.join(self.base_path,'data','human_evaluations','UpdateSumm{}_eval'.format(year), 'manual','peers')
        else:
            # year in [2010, 2011]
            data_path = os.path.join(self.base_path, 'data', 'human_evaluations', 'GuidedSumm{}_eval'.format(year), 'manual', 'peers')
        summ_dic = self.readPeerSummary(data_path)

        return summ_dic

    def readPeerSummary(self,mpath):
        peer_dic = OrderedDict()

        for peer in sorted(os.listdir(mpath)):
            topic = self.uniTopicName(peer)
            if topic not in peer_dic:
                peer_dic[topic] = []
            sents = self.readOnePeer(os.path.join(mpath,peer))
            peer_dic[topic].append((os.path.join(mpath,peer),sents))

        return peer_dic

    def readOnePeer(self,mpath):
        ff = open(mpath,'r',encoding='latin-1')
        sents = []
        # changed by wchen for reading from 'manual' folder
        annot_start = False
        peer_start = False
        for line in ff.readlines():
            orig_line = line
            line = line.strip()
            if annot_start and line == '</text>':
                break
            if peer_start:
                assert line.startswith('<line>') and line.endswith('</line>')
                line = line[len('<line>'):-len('</line>')]
                if line.strip() != '':
                    line = line.strip().replace('&lt;', '<').replace('&gt;', '>').replace('&apos;', "'")
                    line = line.replace('&amp;amp;', '&').replace('&amp;', '&').replace('&amp ;', '&')
                    line = line.replace('&quot;', '"').replace('&slash;', '/')

                    special_tokens = re.search("&[a-z]*?;", line)
                    assert special_tokens == None, "\nFile path: {}\nThis line contains special tokens {}:\n{}".format(
                        mpath, special_tokens, line)
                    sents.append(line)
            if line == '<annotation>':
                annot_start = True
            if annot_start and line == '<text>':
                peer_start = True

        ff.close()
        return sents

    def uniTopicName(self,name):
        doc_name = name.split('-')[0][:5]
        block_name = name.split('-')[1][0]
        return '{}.{}'.format(doc_name,block_name)



if __name__ == '__main__':
    peerReader = PeerSummaryReader(BASE_DIR)
    summ = peerReader('08')

    for topic in summ:
        print('topic {}, summ {}'.format(topic,summ[topic]))