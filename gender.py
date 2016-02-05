from pos import count_pos
import sentiment_feature_extraction
import os, re
import json
import itertools
import xml.etree.ElementTree as et
import numpy as np
from sklearn import cross_validation, svm, linear_model, tree, ensemble, naive_bayes, neighbors
#pos_tags = ['CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS',
#            'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG',
#            'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

def xmltojson(file, gender):

    posts = []
    texts = ''
    tree = et.parse(file)
    documents = tree.iterfind('document')
    """for d in documents:
        post = {}
        post['text'] = d.text
        post['gender'] = gender
        posts.append(post)"""
    for d in documents:
        texts += d.text+'\n'
    post = {}
    post['text'] = texts
    post['gender'] = gender
    posts.append(post)

    return posts
    #return post

def tweetstojson():

    path = '/home/croman/PycharmProjects/gender/data/pan15-author-profiling-training-dataset-english-2015-04-23/'
    files = os.listdir(path)
    tweets = []
    files_gender = {}
    with open(path+'truth.txt', 'r') as truth:
        lines = truth.readlines()
        for line in lines:
            files_gender[line.split(':::')[0]+'.xml'] = line.split(':::')[1]
    print files_gender

    for f in files:
        if f.endswith('.xml'):
            tweets.append(xmltojson(path+f, files_gender[f]))
    tweets = list(itertools.chain(*tweets))

    with open('data/pan15-english/dataset.json', 'w+') as dataset:
        tweets = json.dumps(tweets)
        dataset.write(tweets)

def features(path, posts):
    gender = []
    pos_tags = []
    tag_count = []
    with open('EmoticonSentimentLexicon.txt') as emoticon_lexicon:
        lines = emoticon_lexicon.readlines()
        emot_dict = {}
        for l in lines:
            line = l.split('\t')
            emot_dict[line[0]] = int(line[1].split('\r')[0])

    for p in posts:
        text = p['text']
        emoticons = sentiment_feature_extraction.emoticons_from_dictionary(text, emot_dict)
        if text.split()<=0.0:
            length = len(text.split())
        else:
            length = 1.0
        caps = ('CAPS', float(sentiment_feature_extraction.caps_words(text))/length)
        elongated = ('ELONGATED', float(sentiment_feature_extraction.elonganted_words(text))/length)
        exclamation_interrogation_dict = sentiment_feature_extraction.exclamation_and_interrogation(text)
        excl = ('!', exclamation_interrogation_dict['!'])
        interr = ('?', exclamation_interrogation_dict['?'])
        omg_count = ('OMG', len(re.findall('omg', text, re.I)))
        heart_count = ('<3', len(re.findall('<3', text)))
        lol_count = ('lol', len(re.findall('lo+l', text, re.I)))
        lmfao_count = ('lmfao', len(re.findall('lmfa+o+', text, re.I)))
        emoticon_count = ('EMOTCOUNT', emoticons['number_emoticons'])
        emoticon_score = ('EMOTSCORE', emoticons['score_emoticons'])
        count = count_pos(text)
        count.extend((caps, elongated, excl, interr, emoticon_count, emoticon_score, heart_count, omg_count, lol_count, lmfao_count))
        tag_count.append(count)
        #print caps[1], elongated[1], emoticon_count[1]
        gender.append(p['gender'])
        print len(posts)

    pos_train, pos_test, gender_train, gender_test = cross_validation.train_test_split(
            tag_count, gender, test_size=0.1, random_state=0)


    for post in pos_train:
        for tag in post:
            if tag[0] not in pos_tags:
                pos_tags.append(tag[0])

    complete_tag_count = []
    for post in pos_train:
        p = dict(post)
        for pos in pos_tags:
            if pos not in p:
                post.append((pos, 0))
        post = sorted(post)
        complete_tag_count.append([i[1] for i in post])

    json_train = json.dumps(complete_tag_count)

    with open(path+'training-tags.txt', 'w') as training_file:
        training_file.write(json_train)

    test_tag_count = []
    for post in pos_test:
        p = dict(post)
        for pos in pos_tags:
            if pos not in p:
                post.append((pos, 0))
        post = sorted(post)
        test_tag_count.append([i[1] for i in post])

    json_test = json.dumps(test_tag_count)

    with open(path+'test-tags.txt', 'w') as test_file:
        test_file.write(json_test)

    with open(path+'gender-train.txt', 'w') as gender_train_file, open(path+'gender-test.txt', 'w') as gender_test_file:
        json_gender_train = json.dumps(gender_train)
        json_gender_test = json.dumps(gender_test)
        gender_train_file.write(json_gender_train)
        gender_test_file.write(json_gender_test)

def gender_identification(file, tagging, format):

    path = os.path.dirname(os.path.abspath(file))+'/data/'
    if not os.path.exists(path):
        os.makedirs(path)
    if format == 'xml':
        posts = xmltojson(file)
    elif format == 'json':
        with open(file, 'r') as json_file:
            posts = json.load(json_file)

    if tagging:
        features(path, posts)

    with open(path+'training-tags.txt', 'r') as training_file, open(path+'test-tags.txt', 'r') as test_file, open(path+'gender-train.txt', 'r') as gender_train_file, open(path+'gender-test.txt', 'r') as gender_test_file:
        complete_tag_count = []
        test_tag_count = []
        gender_train = []
        gender_test = []
        for line in training_file:
            complete_tag_count = json.loads(line)
        for line in test_file:
            test_tag_count = json.loads(line)
        for line in gender_train_file:
            gender_train = json.loads(line)
        for line in gender_test_file:
            gender_test = json.loads(line)

    complete_tag_count = np.array(complete_tag_count).astype(np.float)
    test_tag_count = np.array(test_tag_count).astype(np.float)
    gender_train = np.array(gender_train)
    gender_test = np.array(gender_test)
    tag_total = np.concatenate((complete_tag_count, test_tag_count))
    gender_total = np.concatenate((gender_train, gender_test))

    clf1 = linear_model.LogisticRegression()
    clf2 = ensemble.RandomForestClassifier(n_estimators=100)
    clf3 = tree.DecisionTreeClassifier(max_depth=5)
    clf4 = neighbors.KNeighborsClassifier(n_neighbors=10)
    clf5 = svm.SVC(kernel='linear', probability=True)
    eclf = ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('dt', clf3), ('kn', clf4),
                                                 ('svc', clf5)], voting='hard')
    cv = cross_validation.ShuffleSplit(tag_total.shape[0], n_iter=3, test_size=0.3, random_state=0)

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, eclf], ['Logistic Regression', 'Random Forest',
                                                     'Decision Tree', 'KNeighbors', 'SVC', 'Ensemble']):
        scores = cross_validation.cross_val_score(clf, tag_total, gender_total, cv=cv, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

    #scores = cross_validation.cross_val_score(clf, complete_tag_count, gender_train)
    #print scores.mean()
    #clf.fit(complete_tag_count, gender_train)
    #print clf.score(test_tag_count, gender_test)

#print gender_identification('data/blog/blog-gender-dataset.json', True, 'json')
#print gender_identification('data/blog/small-dataset.json', True, 'json')
print gender_identification('data/pan15-english/dataset.json', False, 'json')

