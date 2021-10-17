'''
    process of pretraining dataset:   aspect extraction

    A example of Amazon review dataset:

        __label__2 Great CD: My lovely Pat has one of the GREAT voices of her generation. /
                    I have listened to this CD for YEARS and I still LOVE IT. /
                    When I'm in a good mood it makes me feel better. /
                    A bad mood just evaporates like sugar in the rain. This CD just oozes LIFE. /
                    Vocals are jusat STUUNNING and lyrics just kill. One of life's hidden gems. /
                    This is a desert isle CD in my book. Why she never made it big is just beyond me. /
                    Everytime I play this, no matter black, white, young, old, male, /
                    female EVERYBODY says one thing "Who was that singing ?"
'''

from nltk import sent_tokenize, word_tokenize, pos_tag
from nltk.corpus import stopwords
import pandas as pd

amazon_path = r" "  ## path of Amaozn review dataset
yelp_path = r" "  ## path of Yelp review dataset


def processing_amazon(path, type='train'):
    with open('./dataset/Amazon/{}.txt'.format(type), 'a', encoding='utf-8') as f:
        with open(path, 'r', encoding='utf-8') as a:
            for i in a.readlines():
                condidate = []
                label, _, text = i.partition(' ')
                label = str(int(label[-1]) - 1)
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    pos_tags = pos_tag(word_tokenize(sentence))
                    for word, pos in pos_tags:
                        if ('NN' in pos) and (word not in stopwords.words('english')):
                            condidate.append(word)
                try:
                    aspect = max(set(condidate), key=condidate.count)
                    f.write(aspect + ',' + label + ',' + text)
                except ValueError:
                    continue

def processing_yelp(path, type='train'):
    with open('./dataset/Yelp/{}.txt'.format(type), 'a', encoding='utf-8') as f:
        with open(path, 'r', encoding='utf-8') as a:
            dataset = pd.read_csv(path, header=None)
            for i in range(dataset.shape[0]):
                condidate = []
                text = dataset[1][i]
                if dataset[0][i] < 3:
                    label = str(1)
                elif dataset[0][i] > 3:
                    label = str(0)
                sentences = sent_tokenize(text)
                for sentence in sentences:
                    pos_tags = pos_tag(word_tokenize(sentence))
                    for word, pos in pos_tags:
                        if ('NN' in pos) and (word not in stopwords.words('english')):
                            condidate.append(word)
                try:
                    aspect = max(set(condidate), key=condidate.count)
                    f.write(aspect + ',' + label + ',' + text)
                except ValueError:
                    continue


processing_amazon(amazon_path, 'train')

processing_yelp(yelp_path, 'train')
