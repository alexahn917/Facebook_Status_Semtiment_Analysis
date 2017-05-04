from __future__ import print_function
import csv
import numpy as np
import pandas as pd
from sklearn import decomposition
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def main():
    with open("../data1.csv") as f:
        df = pd.read_csv(f, encoding='latin-1')
    df.fillna(0, inplace=True)

    # get data
    col_names = list(df.columns.values)
    X = df['status_message']
    X = [s.strip().replace('\"', '') for s in X]
    
    # params
    n_features = 1000
    n_topics = 10
    n_top_words = 20
    n_samples = len(X)

    # vectorize
    X_trans, topics = fit_lda(X, n_features, n_topics, n_top_words, n_samples)
    #fit_nmf(X, n_features, n_topics, n_top_words, n_samples)

    # names
    column_names = ["num_reactions", "num_comments", "num_shares", "num_likes", "num_loves", "num_wows", "num_hahas", "num_sads", "num_angrys"]
    topic_names = ["Family", "Urgent", "Offence", "School Crime", "Police/Crime/Satire", "Donald Trump", "Teenage Abortion", "Marriage", "Crime witness", "Attack/Terrorism", "Hilary Clinton"]
    
    results_df = pd.DataFrame(index=np.arange(len(topics)), columns=column_names)
    for column in column_names:
        results_df[column] = get_total_sentiments(X_trans, df, column)
        #print(get_total_sentiments(X_trans, df, column))
    
    # assign topics
    topic_labels = []
    for i in range(len(X_trans)):
        doc_topic = np.argmax(X_trans[i])
        topic_labels.append(np.argmax(X_trans[i]))

    #visualize_cor(results_df, column_names, topic_names)
    #visualize_mat(results_df, column_names, topic_names)
    #visualize_pca(X_trans, topic_labels)

def visualize_cor(results_df, column_names, topic_names):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(results_df.corr(), interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    ax.set_yticks(np.arange(results_df.shape[1]))
    ax.set_yticklabels(topic_names, rotation='horizontal', fontsize=10)
    ax.set_xticks(np.arange(len(column_names)))
    ax.set_xticklabels(topic_names, rotation=70, fontsize=8)
    plt.title("Correlation Matrix of Topics")
    plt.show()

def visualize_mat(results_df, column_names, topic_names):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(results_df, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar()
    ax.set_yticks(np.arange(results_df.shape[1]))
    ax.set_yticklabels(topic_names, rotation='horizontal', fontsize=11)
    ax.set_xticks(np.arange(len(column_names)))
    ax.set_xticklabels([(i.split('_')[1]) for i in column_names], rotation=70, fontsize=11)
    plt.title("Distribution of each user response")
    plt.show()

def visualize_pca(X_trans, topic_labels):
    # normalize X
    X = X_trans[:] - np.mean(X_trans[:])
    pca = decomposition.PCA(n_components=X[0,:].size)
    pca.fit(X)
    X_pca = pca.transform(X)
    E_vectors = pca.components_.T
    E_values = pca.explained_variance_
    print("Explained variance with 2 eigan vectors: %f%%" %np.sum(pca.explained_variance_ratio_[:2]))
    print("Explained variance with 3 eigan vectors: %f%%" %np.sum(pca.explained_variance_ratio_[:3]))

    plt.scatter(X_pca[:,0], X_pca[:,1], s=1, c=topic_labels, marker='o')
    plt.title('2 Principle Components Projection on Status Topic Distribution')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.show()

    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(X_pca[:,1], X_pca[:,2], X_pca[:,3], s=1, c=topic_labels, marker='o')
    plt.title('3 Principle Components Projection on Status Topic Distribution')
    ax2.set_xlabel('PC1')
    ax2.set_ylabel('PC2')
    ax2.set_zlabel('PC3')
    plt.show()


def fit_nmf(X, n_features, n_topics, n_top_words, n_samples):
    print("Fitting the NMF model with tf-idf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
    # NMF
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2,
                                   max_features=n_features,
                                   stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(X)
    # fit
    nmf = NMF(n_components=n_topics, random_state=1,
          alpha=.1, l1_ratio=.5).fit(tfidf)
    tfidf_feature_names = tfidf_vectorizer.get_feature_names()
    print("\nTopics in NMF model:")
    print_top_words(nmf, tfidf_feature_names, n_top_words)


def fit_lda(X, n_features, n_topics, n_top_words, n_samples):
    print("Fitting LDA models with tf features, "
      "n_samples=%d and n_features=%d..."
      % (n_samples, n_features))
    # LDA
    tf_vectorizer = CountVectorizer(max_df=0.95, min_df=2,
                                max_features=n_features,
                                ngram_range=(1,1),
                                stop_words='english')
    tf = tf_vectorizer.fit_transform(X)
    # fit
    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=5,
                                learning_method='online',
                                learning_offset=50.,
                                random_state=0)
    X = lda.fit_transform(tf)
    print("\nTopics in LDA model:")
    tf_feature_names = tf_vectorizer.get_feature_names()
    topics = print_top_words(lda, tf_feature_names, n_top_words)
    return X, topics

def print_top_words(model, feature_names, n_top_words):
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        new_topic = " ".join([feature_names[i]
                        for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(new_topic)
        topics.append(new_topic)
    print()
    return(topics)

def get_total_sentiments(X_trans, df, sentiment):
    num_topics = len(X_trans[0])
    total_sentiments = [0] * num_topics
    for i in range(len(X_trans)):
        total_sentiments += X_trans[i] * df[sentiment][i]

    total_num = np.sum(total_sentiments) + 0.0
    total_num_prec = [0.0] * num_topics
    #print("*Total %s*" %sentiment)
    for i in range(len(total_sentiments)):
        #print("Topic %d: %.2f" %(i, total_sentiments[i] / total_num))
        total_num_prec[i] = (np.log(total_sentiments[i]))# / total_num))
    return total_num_prec

if __name__ == '__main__':
    main()