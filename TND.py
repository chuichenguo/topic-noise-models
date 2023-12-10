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

model1 = TndMallet(tnd_path, corpus, num_topics=30, id2word=dictionary, workers=4,
                   alpha=50, beta=0.01, skew=25, noise_words_max=200, iterations=1000)

topics = model1.show_topics(num_topics=30, num_words=20, formatted=False)
for i in range(0,30):
	print(topics[i])

noise = model1.load_noise_dist()

noise_list = sorted([(x, noise[x]) for x in noise.keys()], key=lambda x: x[1], reverse=True)
for x in noise_list:
	print(x)
