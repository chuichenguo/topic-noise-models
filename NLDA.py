# dataset = [['this', 'is', 'doc', '1'], ['this', 'is', 'doc', '2']]
from gensim import corpora
from settings.common import load_flat_dataset

dataset = load_flat_dataset('data/data_elonmusk_preprocess5.csv', delimiter=' ')
dictionary = corpora.Dictionary(dataset)
dictionary.filter_extremes()
corpus = [dictionary.doc2bow(doc) for doc in dataset]

from gensim.models import FastText
from gensim.models.fasttext import save_facebook_model
from settings.common import load_flat_dataset

dataset_name = 'data_elonmusk_preprocess5'
dataset = load_flat_dataset('data/{}.csv'.format(dataset_name))
ft = FastText(sentences=dataset, vector_size=100, min_count=50)
save_facebook_model(ft, 'local_{}_ft.bin'.format(dataset_name))

from tm_pipeline.tndmallet import TndMallet
from tm_pipeline.etndmallet import eTndMallet

tnd_path = 'mallet-tnd/bin/mallet'
etnd_path = 'mallet-etnd/bin/mallet'
mallet_path = 'mallet-2.0.8/bin/mallet'

from tm_pipeline.nlda import NLDA

model = NLDA(dataset=dataset, tnd_k=30, tnd_alpha=50, tnd_beta0=0.01, tnd_beta1=25, tnd_noise_words_max=200,
                 tnd_iterations=1000, lda_iterations=1000, lda_k=30, nlda_phi=10, nlda_topic_depth=100, top_words=20,
                 save_path='results/', mallet_tnd_path=tnd_path, mallet_lda_path=mallet_path, random_seed=1824, run=True)
                 
#print(model.lda_topics)
#print()
#for x in model.tnd_noise_distribution:
#	print(x)
for key,value in model.tnd_noise_distribution.items():
    print(key, ':', value)
print()
#print(model.topics)
for x in model.topics:
	print(x)
