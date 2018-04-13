import sqlite3
import re, nltk
import numpy as np
from lib import models
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from pywsd.utils import lemmatize_sentence

session = models.Session()

# create db
conn = sqlite3.connect('keywords.db')
c = conn.cursor()
c.execute('CREATE TABLE IF NOT EXISTS keywords (id integer primary key, keywords text)')


# For cleaning comment texts
patterns = re.compile(r"http\S+|<.*?>|\.\.+|&gt;+|#|/")
nonwords = ["quot","quote","class=","n't","body-line","ltr","hellip"]
stopwords = set(nltk.corpus.stopwords.words("english") + nonwords)


def image_contexts(usages, model, print_summaries=False):
    # Non-empty and unique text content for topic extraction
    usage_texts = [(u.image.id, u.context.content) for u in usages if u.context.content]
    preprocessed = remove_empty([(id, clean_comment(post)) for id, post in usage_texts])

    # keep track of which
    # post texts map to which images
    index = {}
    for i, (id, post) in enumerate(preprocessed):
        index[i] = id

    tokenized = [post[2] for id, post in preprocessed]
    # stripped = [' '.join(post[1]) for id, post in preprocessed]
    # unprocessed = [post[0] for id, post in preprocessed]

    # Calculate distances
    distances = calc_distances(tokenized, model, dist_type='word_movers')

    # Cluster comments
    dbscan = best_cluster(distances, model_name='glove', dist_type='word_movers')

    # Extract keywords
    keywords = extract_keywords(dbscan, tokenized)

    clusters = defaultdict(list)
    image_to_keywords = defaultdict(list)
    for i, label in enumerate(dbscan.labels_):
        id = index[i]
        clusters[label].append(id)
        image_to_keywords[id] += keywords[label]

    return clusters, image_to_keywords


def clean_comment(raw_text):
    # Strip URL strings and HTML code
    strip_text = " ".join([seg for seg in patterns.sub("\n", raw_text).split("\n") if seg not in [""," "]])
    # Tokenize and lemmatize according to part of speech
    lemma_text = lemmatize_sentence(strip_text)
    # Remove tokens containing numbers, punctuations, or stop words
    token_text = [tok for tok in lemma_text if re.search("[a-zA-Z]", tok) and not tok.startswith("'") and tok not in stopwords]
    # Stripped but not lemmatized text including stopwords for display
    strip_text = nltk.word_tokenize(strip_text)
    strip_text = [tok for tok in strip_text if re.search("[a-zA-Z]", tok) and not tok.startswith("'") and tok not in nonwords]
    return raw_text, strip_text, token_text


def remove_empty(preproc):
    preproc_edit = []
    for id, post in preproc:
        # post[0]=raw_text, post[1]=strip_text, post[2]=token_text
        # Exclude comment from analysis if text cleaning left no words
        if len(post[2]) > 0:
            preproc_edit.append((id, post))
    return preproc_edit


def calc_distances(posts, model, dist_type='cosine'):
    if dist_type == 'word_movers':
        # requires: pip install pyemd
        return np.array([[model.wmdistance(p1,p2) for p2 in posts] for p1 in posts])
    elif dist_type == 'cosine':
        import ipdb; ipdb.set_trace()
        return np.array([[cosine_dist(avg_post(p1,model.wv),avg_post(p2,model.wv)) for p2 in posts] for p1 in posts])


# Average embedding vectors for each word in a comment
def avg_post(post, word_vector):
    vec = np.zeros(300)
    for tok in post:
        if tok in word_vector:
            vec += word_vector[tok]

    # if no words in the post were in the w2v vocab,
    # initialize in this way:
    # <https://groups.google.com/forum/#!topic/word2vec-toolkit/J3Skqbe3VwQ>
    if not np.any(vec):
        return np.random.uniform(-0.25, 0.25, 300)
    else:
        return vec/len(post)


def cosine_dist(a, b):
    return 1.0 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def cluster_comments(dist_matrix, epsilon):
    cluster = DBSCAN(eps=epsilon, min_samples=2, metric='precomputed', n_jobs=-1).fit(dist_matrix)
    cluster_num = len(set(cluster.labels_)) - 1
    outlier_num = len([i for i in cluster.labels_ if i == -1])
    return cluster, cluster_num, outlier_num


def best_cluster(dist_matrix, model_name='glove', dist_type='cosine'):
    if dist_type == 'cosine':
        step = 0.01
    elif dist_type == 'word_movers':
        if model_name == 'word2vec':
            step = 0.05
        elif model_name == 'glove':
            step = 0.1

    if np.min(dist_matrix) <= 0:
        min_dist = step
    else:
        min_dist = np.min(dist_matrix)
    max_dist = np.max(dist_matrix)

    cluster_info, cluster_num, outlier_num = [], [], []
    for epsilon in np.arange(min_dist, max_dist+step, step):
        cluster = cluster_comments(dist_matrix, epsilon)
        cluster_info.append(cluster[0])
        cluster_num.append(cluster[1])
        outlier_num.append(cluster[2])

    # Best clustering parameters for a set of comments
    # has the maximum number of non-singleton clusters
    # with the minimum number of outlier comments
    max_cluster = np.where(cluster_num == np.max(cluster_num))[0]
    min_outlier = np.argmin(np.array(outlier_num)[max_cluster])
    best_index = max_cluster[min_outlier]

    return cluster_info[best_index]


def extract_keywords(clusters, posts):
    aggregated = []
    labels = list(set(clusters.labels_))
    for label in labels:
        if label > -1: # Not much difference from including outlier cluster
            indices = np.where(clusters.labels_ == label)[0]
            # Treat each cluster as one "document" for tf-idf calculations
            aggregated.append(' '.join([' '.join(posts[i]) for i in indices]))
    # Calculate tf-idf scores to find important words for each cluster
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(aggregated)
    # List of all words in vocabulary in same order as matrix columns
    features = vectorizer.get_feature_names()
    # Keep the top-scoring N words to describe each cluster or theme
    keywords = {labels[i]:top_tfidf_feats(row, features, top_n=5) for i,row in enumerate(list(tfidf))}
    keywords[-1] = ['miscellaneous']
    return keywords


def top_tfidf_feats(vector, features, top_n=5):
    vector = np.squeeze(vector.toarray()) # Convert from sparse to dense
    top_ids = np.argsort(vector)[::-1][:top_n] # Sort scores in descending order
    #return [(features[i], vector[i]) for i in top_ids if vector[i]>0] # Words associated with top scores
    return [features[i] for i in top_ids if vector[i]>0]

from tqdm import tqdm
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec

def load_embeddings(model_name='glove'):
    if model_name == 'word2vec':
        # MODEL DOWNLOAD: https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing
        return KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    elif model_name == 'glove':
        # MODEL DOWNLOAD: http://nlp.stanford.edu/data/glove.840B.300d.zip
        glove2word2vec('data/glove.840B.300d.txt','data/glove.840B.300d.w2vformat.txt')
        return KeyedVectors.load_word2vec_format('data/glove.840B.300d.w2vformat.txt')


if __name__ == '__main__':
    usages = session.query(models.ImageUsage).distinct(models.ImageUsage.context_id, models.ImageUsage.image_id)
    count = usages.count()
    usages = tqdm(usages, total=count)

    # Takes ~10 minutes for glove. ~3 minutes for word2vec.
    print('Loading vecs...')
    model = load_embeddings(model_name='glove')
    print('Done loading')

    clusters, keywords = image_contexts(usages, model)
    for id, kws in keywords.items():
        c.execute('INSERT INTO keywords VALUES (?, ?)', (id, ','.join(kws)))
        conn.commit()
