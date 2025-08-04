#Code for Top cluster or topic finding with 5 bag of words (Top 5 Words extraction within each cluster)
import numpy
a=[]
tweet=1
print (corpus_stemmed[tweet])
topic =lda_train.get_document_topics(corpus_stemmed[tweet], minimum_probability=0.0)
#print (topic)
for i in range(23):
    a.append(topic[i][1])
array = numpy.array(a)
#max_index = array.argmax()
#print (max_index)
max_index = array.argsort()[-3:][::-1]
print (max_index)

topic_id=topic[max_index[0]][0]
lda_train.print_topics(33, num_words=5)[topic_id]
topic_text = lda_train.show_topic(topic_id)
print (topic_text [0][0] +'  '+ topic_text [1][0]+'  '+ topic_text [2][0]+'  '+ topic_text [3][0]+'  '+ topic_text [4][0])
arr1 = {'stemmed':[],'class':[]}
arr2 = {'Cluster ID':[],'Words':[]}
arr1 ['stemmed'] = df ['stemmed']
arr1 ['class'] = df ['Classification']

tweets = pd.DataFrame(arr1)
cluster = pd.DataFrame(arr2)
print(tweets)
print("----------------")
print(cluster)
# making corpus with assigned clusters and bag of words to each
i=0
for i in range(len(tweets)):
    topics=[]
    t=i
    topic=lda_train.get_document_topics(corpus_stemmed[t], minimum_probability=0.0)
    j=0
    for j in range(23):
        topics.append(topic[j][1])
    array = numpy.array(topics)
    max_index = array.argmax()
    topic_id=topic[max_index][0]

    lda_train.print_topics(33, num_words=5)[topic_id]
    topic_text = lda_train.show_topic(topic_id)
    #print (topic_text [0][0] +'  '+ topic_text [1][0]+'  '+ topic_text [2][0]+'  '+ topic_text [3][0]+'  '+ topic_text [4][0])
    new_row = {'Cluster ID': [topic_id], 'Words': [topic_text]}
    #[0][0] +' '+ topic_text [1][0]+' '+ topic_text [2][0]+' '+ topic_text [3][0]+' '+ topic_text [4][0]
    cluster = pd.concat([cluster, pd.DataFrame(new_row)], ignore_index=True)

import csv
i=0
with open('gdrive/My Drive/Colab Notebooks/Preprocessed Datasets/ClusteredDatasetNew.csv', 'w', encoding='Latin-1', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['stemmed', 'class','Cluster ID','Cluster Words'])
        for i in range(len(tweets)):
                    writer.writerow([tweets['stemmed'][i] , tweets['class'][i], cluster['Cluster ID'][i], cluster['Words'][i]])
    #arr.clear()

    train_vecs = []
rev = {'stemmed':[],'class':[]}
rev ['stemmed'] = df ['stemmed']
rev ['class'] = df ['Classification']

rev_train = pd.DataFrame(rev)
for i in range(len(rev_train)):
    top_topics = (lda_train.get_document_topics(corpus_stemmed[i], minimum_probability=0.0))

    topic_vec = [top_topics[i][1] for i in range(23)]
    #topic_vec.extend([rev_train.iloc[i]['class']]) # counts of reviews for restaurant
    #topic_vec.extend([len(rev_train.iloc[i].text)]) # length review
    train_vecs.append(topic_vec)
    import numpy as np
topic_array = np.zeros((23, 10), dtype=numpy.object_)
for i in range(23):
    listoftopics = lda_train.show_topic(i)
    for j in range(10):
        topic_array[i][j] = listoftopics[j][0]
print(topic_array[11])
