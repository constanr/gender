
# coding: utf-8

# # Load the dataset

from pos import count_pos
import sentiment_feature_extraction
import os, xml.etree.ElementTree as et, itertools
from bs4 import BeautifulSoup
import numpy as np
from sklearn import cross_validation, svm, linear_model, tree, ensemble, naive_bayes, neighbors, gaussian_process, grid_search
from sklearn.multiclass import OneVsRestClassifier
import csv, re
import pickle

def extract_targets(datapath):
    files_gender = {}
    files_age = {}
    with open(datapath+'truth.txt', 'rb') as truth:
        lines = truth.readlines()
        for line in lines:
            files_gender[line.split(':::')[0]+'.xml'] = line.split(':::')[1]
            files_age[line.split(':::')[0]+'.xml'] = line.split(':::')[2]
    return files_gender, files_age

def extract_tweets(language, year):
    if year == '2015':
        datapath = 'data/pan15-author-profiling-training-dataset-'+language+'-2015-04-23/'
        files_gender, files_age = extract_targets(datapath)
        files = os.listdir(datapath)
        tweets = []
        for f in files:
            if f.endswith('.xml'):
                posts = []
                texts = ''
                tree = et.parse(datapath+f)
                documents = tree.iterfind('document')
                for d in documents:
                    texts += d.text+'\n'
                post = {}
                post['text'] = texts
                post['gender'] = files_gender[f]
                post['age'] = files_age[f]
                posts.append(post)
                tweets.append(posts)
        tweets = list(itertools.chain(*tweets))
    elif year == '2016':
        datapath = 'data/pan16-author-profiling-training-dataset-'+language+'-2016-02-29/'
        files_gender, files_age = extract_targets(datapath)
        files = os.listdir(datapath)
        tweets = []
        i=1
        for f in files:
            if f.endswith('.xml'):                
                posts = []
                texts = ''
                root = et.parse(datapath+f)
                for d in root.find('documents').findall('document'):
                    if d.text!=None:
                        soup = BeautifulSoup(d.text, 'html.parser').get_text()
                        texts += soup+'\n'
                post = {}
                post['text'] = texts
                post['gender'] = files_gender[f]
                post['age'] = files_age[f]
                posts.append(post)
                tweets.append(posts)
            #print str(i)+'/'+str(len(files))
            i+=1
        tweets = list(itertools.chain(*tweets))    
    return tweets

# # Extract Features

"""from sklearn.feature_extraction.text import HashingVectorizer
from sklearn import cross_validation, linear_model
hv = HashingVectorizer(ngram_range=(1, 2), binary=True)
data = []
gender = []
age = []
for t in tweets:
    data.append(t['text'])
    gender.append(t['gender'])
    age.append(t['age'])

features = hv.transform(data)"""

def load_lexicons():
    with open('EmoticonSentimentLexicon.txt') as emoticon_lexicon:
        lines = emoticon_lexicon.readlines()
        global emot_dict
        emot_dict = {}
        for l in lines:
            line = l.split('\t')
            emot_dict[line[0]] = int(line[1].split('\r')[0])    

        global age_lexicon
        global gender_lexicon
        age_lexicon = {}
        gender_lexicon = {}
    with open('lexica/emnlp14age.csv') as age_lexicon_file:
        age_reader = csv.reader(age_lexicon_file, delimiter=',', quotechar='"')
        age_reader.next()
        global age_intercept
        age_intercept = age_reader.next()[1]
        for row in age_reader:
            age_lexicon[row[0]] = float(row[1])
    with open('lexica/emnlp14gender.csv') as gender_lexicon_file:
        gender_reader = csv.reader(gender_lexicon_file, delimiter=',', quotechar='"')
        gender_reader.next()
        global gender_intercept
        gender_intercept = gender_reader.next()[1]
        for row in gender_reader:
            gender_lexicon[row[0]] = float(row[1])

def extract_features(tweets, language):
    gender = []
    age = []
    pos_tags = []
    tag_count = []
    i = 0
    for t in tweets:
        text = t['text']
        emoticons = sentiment_feature_extraction.emoticons_from_dictionary(text, emot_dict)
        if text.split()<=0.0:
            length = len(text.split())
        else:
            length = 1.0
        caps = ('CAPS', float(sentiment_feature_extraction.caps_words(text)))
        elongated = ('ELONGATED', float(sentiment_feature_extraction.elonganted_words(text)))
        exclamation_interrogation_dict = sentiment_feature_extraction.exclamation_and_interrogation(text)
        excl = ('!', exclamation_interrogation_dict['!'])
        interr = ('?', exclamation_interrogation_dict['?'])
        omg_count = ('OMG', len(re.findall('omg+', text, re.I)))
        heart_count = ('<3', len(re.findall('<3+', text)))
        lol_count = ('lol', len(re.findall('lo+l', text, re.I)))
        lmfao_count = ('lmfao', len(re.findall('lmfa+o+', text, re.I)))
        emoticon_count = ('EMOTCOUNT', emoticons['number_emoticons'])
        emoticon_score = ('EMOTSCORE', emoticons['score_emoticons'])
        mention_count = ('@COUNT', len(re.findall('@username', text)))
        hashtag_count = ('#COUNT', len(re.findall('#', text)))
        rt_count = ('RT', len(re.findall('RT @username', text)))
        url_count = ('URL', len(re.findall('http[s]?://', text)))
        pic_count = ('PIC', len(re.findall('pic.twitter.com', text)))
        avg_text_length = ('TEXTLEN', length/len(text.split('\n')))
        words_length = 0
        for word in text.split():
            words_length += len(word)        
        avg_word_length = ('WORDLEN', words_length/length)
        count = count_pos(text, language)
        count_dict = dict(count)
        extrav_score = 0.0
        sum_tags = ['NN', 'JJ', 'IN', 'DT']
        sub_tags = ['PRP', 'VB', 'VBD', 'VBG', 'VBZ', 'VBP', 'VBN', 'RB', 'UH']
        for tag in sum_tags:
            if count_dict.has_key(tag):
                extrav_score += count_dict[tag]
        for tag in sub_tags:
            if count_dict.has_key(tag):
                extrav_score -= count_dict[tag]    
        extraversion = ('EXTRAV', (extrav_score+100)/2.0)

        if language == 'english':
            bf_words = 'wife|gf|girlfriend|dw'
            gf_words = 'husband|bf|boyfriend|hubby|dh'
        elif language == 'spanish':
            bf_words = 'mujer|novia|esposa'
            gf_words = 'marido|novio|esposo'
        elif language == 'dutch':
            bf_words = 'vrouw|vriendin'
            gf_words = 'man|bf|vriend'

        bf_count = ('GFCOUNT', len(re.findall(bf_words, text, re.I)))
        gf_count = ('BFCOUNT', len(re.findall(gf_words, text, re.I)))

        count.extend((caps, elongated, excl, interr, omg_count, heart_count, lol_count, lmfao_count, emoticon_count, 
                emoticon_score, mention_count, hashtag_count, rt_count, url_count, pic_count, avg_text_length, 
                avg_word_length, extraversion, bf_count, gf_count))

        if language == 'english':
            male_rationales = ('MALRAT', len(re.findall('bro|dude|homie', text, re.I)))
            female_rationales = ('FEMRAT', len(re.findall('cute', text, re.I)))

            gender_lex_count = 0.0
            age_lex_count = 0.0
            for word in text.split():
                if word in gender_lexicon.keys():
                    gender_lex_count += gender_lexicon[word]
                if word in age_lexicon.keys():
                    age_lex_count += age_lexicon[word]
            gender_lex = ('GLEX', float(gender_intercept)+gender_lex_count/length)
            age_lex = ('ALEX', float(age_intercept)+age_lex_count/length)

            count.extend((male_rationales, female_rationales, gender_lex, age_lex))

        gender.append(t['gender'])
        age.append(t['age'])
        #count.extend([('GENDER', t['gender']), ('AGE', t['age'])])
        tag_count.append(count)

        i += 1
        #print str(i)+'/'+str(len(tweets))
    return tag_count, gender, age

# Save features
def save_features(language, year, tag_count, gender, age):
    if year == '2015':
        datapath = 'data/pan15-author-profiling-training-dataset-'+language+'-2015-04-23/'
    elif year == '2016':
        datapath = 'data/pan16-author-profiling-training-dataset-'+language+'-2016-02-29/'

    if not os.path.isdir(datapath+'data'):
        os.makedirs(datapath+'data')
    with open(datapath+'data/tag_count.p', 'wb') as tagfile:
        pickle.dump(tag_count, tagfile)
    with open(datapath+'data/gender.p', 'wb') as genderfile:
        pickle.dump(gender, genderfile)
    with open(datapath+'data/age.p', 'wb') as agefile:
        pickle.dump(age, agefile)


# Load features
def load_features(language, year):
    if year == '2015':
        datapath = 'data/pan15-author-profiling-training-dataset-'+language+'-2015-04-23/'
    elif year == '2016':
        datapath = 'data/pan16-author-profiling-training-dataset-'+language+'-2016-02-29/'

    with open(datapath+'data/tag_count.p', 'rb') as tagfile:
        tag_count = pickle.load(tagfile)
    with open(datapath+'data/gender.p', 'rb') as genderfile:
        gender = pickle.load(genderfile)
    with open(datapath+'data/age.p', 'rb') as agefile:
        age = pickle.load(agefile)
        
    return tag_count, gender, age

def complete_tags(tag_count):
    pos_tags = []
    for post in tag_count: 
        for tag in post:
            if tag[0] not in pos_tags:
                pos_tags.append(tag[0])
    #print pos_tags

    complete_tag_count = []
    for post in tag_count:
        p = dict(post)
        for pos in pos_tags:
            if pos not in p:
                post.append((pos, 0))
        post = sorted(post)
        complete_tag_count.append([i[1] for i in post])
        
    return complete_tag_count

# # Train and evaluate classifiers

"""tag_total = np.array(complete_tag_count)
gender_total = np.array(gender)
age_total = np.array(age)

remove_index = pos_tags.index('EXTRAV')
remove_index = [3,4,8,9,11,12,14,15,18,19,20,22,23,24,25,26,29,31,32,33,34,35,36,37,38,39,41,42,44,47,48,49,
                50,52,53,54,55,56,57,58,59,60,61,62]
filtered_tags = np.zeros(shape=(len(tag_total),29))
for t in range(0, len(tag_total)-1):
    filtered_tags[t] = np.delete(tag_total[t], remove_index)
tag_total = filtered_tags"""

def train_and_evaluate(complete_tag_count, predicted_class):

    tag_total = np.array(complete_tag_count)
    predicted_final = np.array(predicted_class)

    #features_total = np.array(features)

    clf1 = linear_model.LogisticRegression()
    clf2 = ensemble.RandomForestClassifier(n_estimators=100)
    clf3 = ensemble.ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=1, random_state=0, criterion='entropy')
    clf4 = tree.DecisionTreeClassifier(max_depth=3)
    clf5 = svm.SVC(kernel='linear', probability=True, C=0.05)
    clf6 = naive_bayes.GaussianNB()
    clf7 = naive_bayes.BernoulliNB()
    clf8 = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0)
    clf9 = ensemble.AdaBoostClassifier(n_estimators=100)
    clf10 = OneVsRestClassifier(clf4)

    eclf = ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('ext', clf3), ('dt', clf4), ('kn', clf5),
                                                 ('svcl', clf6), ('gnb', clf7), ('gbc', clf8), ('ada', clf9), ('multi', clf10)
                                                 ], voting='soft')
    eclf2 = ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('ext', clf3), ('dt', clf4), ('kn', clf5),
                                                 ('svcl', clf6), ('gnb', clf7), ('gbc', clf8), ('ada', clf9), ('multi', clf10)
                                                 ], voting='hard')

    cv = cross_validation.KFold(tag_total.shape[0], 3)

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, eclf, eclf2], ['Logistic Regression', 
        'Random Forest', 'Extra Trees', 'Decision Tree', 'SVC Linear','Gaussian NB', 'Bernoulli NB', 'Gradient Boosting Classifier',
        'AdaBoost', 'One vs Rest', 'Soft Voting Ensemble', 'Hard Voting Ensemble']):
        scores = cross_validation.cross_val_score(clf, tag_total, predicted_final, cv=cv, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))

def predict_and_evaluate(complete_tag_count, feature_class, target_class):
    tag_total = np.array(complete_tag_count)
    feature_final = np.array(feature_class)
    target_final = np.array(target_class)
    complete_tags = []

    for i in range(0, len(tag_total)):
        user = tag_total[i]
        if feature_final[i]=='M':
            user = np.append(user, 0)
        elif feature_final[i]=='F':
            user = np.append(user, 1)
        complete_tags.append(user)

    complete_tags = np.array(complete_tags)

    cv = cross_validation.KFold(tag_total.shape[0], 3)

    clf = ensemble.ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=1, random_state=0, criterion='entropy')
    print cross_validation.cross_val_score(clf, tag_total, target_final, cv=cv, scoring='accuracy')
    print cross_validation.cross_val_score(clf, complete_tags, target_final, cv=cv, scoring='accuracy')

def features(language, year):
    tweets = extract_tweets(language, year)
    load_lexicons()
    tags, gender, age = extract_features(tweets, language)
    save_features(language, year, tags, gender, age)

def evaluate(language, year, target):
    tags, gender, age = load_features(language, year)
    if target == 'age':
        predicted_class = age
    elif target == 'gender':
        predicted_class = gender
    final_tags = complete_tags(tags)
    train_and_evaluate(final_tags, predicted_class)

"""language = 'spanish'
year = '2015'
tags, gender, age = load_features(language, year)
final_tags = complete_tags(tags)
target_class = age
feature_class = gender
predict_and_evaluate(final_tags, feature_class, target_class)"""

languages = ['dutch', 'spanish', 'english'] #spanish, english, english-nltk, dutch
years = ['2015', '2016'] #2015, 2016
targets = ['gender', 'age']

for y in ['2016']:
    for l in ['english']:
        #features(l, y)
        if l == 'dutch':
            print l, y, 'gender'
            evaluate(l, y, 'gender')
        else:
            for t in targets:
                print l, y, t
                evaluate(l, y, t)




"""cv = cross_validation.KFold(tag_total.shape[0], 3)

for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, eclf, grid], ['Logistic Regression', 'Random Forest',
                                                 'Decision Tree', 'SVC Linear','Gaussian NB', 'Bernoulli NB', 
                                                'Gradient Boosting Classifier', 'AdaBoost', 'One vs Rest', 'Voting Ensemble', 'Grid Search']):
    scores = cross_validation.cross_val_score(clf, features, predicted_class, cv=cv, scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))"""

"""clf8.fit(tag_total, predicted_class)
for c in range(1,len(clf8.feature_importances_)):
    print pos_tags[c], clf8.feature_importances_[c]
clf3.fit(tag_total, gender_total)
from sklearn.externals.six import StringIO
with open("gender.dot", 'w') as f:
    f = tree.export_graphviz(clf3, out_file=f, filled=True, class_names=['Female', 'Male'], feature_names=pos_tags)
"""