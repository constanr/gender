#!/usr/bin/env python
# -*- coding: utf-8 -*-

# # Load the dataset
from __future__ import division
from pos import count_pos
import sentiment_feature_extraction
import os, xml.etree.ElementTree as et, itertools
from bs4 import BeautifulSoup
import numpy as np
from sklearn import cross_validation, svm, linear_model, tree, ensemble, naive_bayes
from sklearn.multiclass import OneVsRestClassifier
import csv, re, clean_text, pickle
import extract_features_from_text as extractor
import sys
import docs2vecs


def extract_targets(datapath):
    files_gender = {}
    files_age = {}
    with open(datapath+'/truth.txt', 'rb') as truth:
        lines = truth.readlines()
        for line in lines:
            files_gender[line.split(':::')[0]+'.xml'] = line.split(':::')[1]
            files_age[line.split(':::')[0]+'.xml'] = line.split(':::')[2].split('\n')[0]
    return files_gender, files_age

def extract_tweets(datapath, task):
    if task == 'training':
        files_gender, files_age = extract_targets(datapath)
    files = os.listdir(datapath)
    tweets = []
    i=1
    for f in files:
        if f.endswith('.xml'):
            fid = f.split('.')[0]
            posts = []
            texts = ''
            root = et.parse(datapath+'/'+f)
            for d in root.find('documents').findall('document'):
                if d.text!=None:
                    soup = BeautifulSoup(d.text, 'html.parser').get_text()
                    texts += soup+'\n'
            post = {}
            post['text'] = texts
            if task == 'training':
                post['gender'] = files_gender[f]
                post['age'] = files_age[f]
            post['fid'] = fid
            posts.append(post)
            tweets.append(posts)
        #print str(i)+'/'+str(len(files))
        i+=1
    tweets = list(itertools.chain(*tweets))
    return tweets

# # Extract Features

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
    with open('dictionaries/english/dictionary-gender-v2.tsv','r') as f:
        gender_dict_reader = csv.reader(f, delimiter='\t', quotechar='"')
        global dictionary_gender
        dictionary_gender = dict()
        for row in gender_dict_reader:
            word_dict = dict()
            word_dict['pmi'] = float(row[1])
            word_dict['gender'] = row[2]
            dictionary_gender[row[0]] = word_dict
    global dictionary_LIWC_en
    dictionary_LIWC_en = extractor.load_LIWC('dictionaries/english/')

def extract_features(tweets, language, task, doc2vec):
    print 'Extracting features...'
    gender = []
    age = []
    pos_tags = []
    tag_count = []
    i = 0
    for t in tweets:
        text = t['text']
        if doc2vec == True:
            model = docs2vecs.load_model(language, docs2vecs.models_path, docs2vecs.models)
            count = docs2vecs.extract_vector(unicode(text), model)
        else:
            emoticons = sentiment_feature_extraction.emoticons_from_dictionary(text, emot_dict)
            if len(text.split())>0:
                length = float(len(text.split()))
            else:
                length = 1.0
            print length
            caps = ('CAPS', float(sentiment_feature_extraction.caps_words(text))/length)
            elongated = ('ELONGATED', float(sentiment_feature_extraction.elonganted_words(text))/length)
            exclamation_interrogation_dict = sentiment_feature_extraction.exclamation_and_interrogation(text)
            excl = ('!', float(exclamation_interrogation_dict['!'])/length)
            interr = ('?', float(exclamation_interrogation_dict['?'])/length)
            omg_count = ('OMG', float(len(re.findall('omg+', text, re.I)))/length)
            heart_count = ('<3', float(len(re.findall('<3+', text)))/length)
            lol_count = ('lol', float(len(re.findall('lo+l', text, re.I)))/length)
            lmfao_count = ('lmfao', float(len(re.findall('lmfa+o+', text, re.I)))/length)
            emoticon_count = ('EMOTCOUNT', float(emoticons['number_emoticons'])/length)
            if emoticons['number_emoticons']==0:
                emoticon_score = ('EMOTSCORE', 0.0)
            else:
                emoticon_score = ('EMOTSCORE', float(emoticons['score_emoticons'])/float(emoticons['number_emoticons']))
            mention_count = ('@COUNT', float(len(re.findall('@username', text)))/length)
            hashtag_count = ('#COUNT', float(len(re.findall('#', text)))/length)
            rt_count = ('RT', float(len(re.findall('RT @username', text)))/length)
            url_count = ('URL', float(len(re.findall('http[s]?://', text)))/length)
            pic_count = ('PIC', float(len(re.findall('pic.twitter.com', text)))/length)
            avg_text_length = ('TEXTLEN', length/len(text.split('\n')))
            words_length = 0
            for word in text.split():
                words_length += len(word)
            avg_word_length = ('WORDLEN', words_length/length)
            count = count_pos(text, language)
            count_dict = dict(count)
            """extrav_score = 0.0
            sum_tags = ['NN', 'JJ', 'IN', 'DT']
            sub_tags = ['PRP', 'VB', 'VBD', 'VBG', 'VBZ', 'VBP', 'VBN', 'RB', 'UH']
            for tag in sum_tags:
                if count_dict.has_key(tag):
                    extrav_score += count_dict[tag]*length
            for tag in sub_tags:
                if count_dict.has_key(tag):
                    extrav_score -= count_dict[tag]*length
            extraversion = ('EXTRAV', (extrav_score+100)/2.0)"""

            if language == 'english':
                bf_words = 'wife|gf|girlfriend|dw'
                gf_words = 'husband|bf|boyfriend|hubby|dh'
            elif language == 'spanish':
                bf_words = 'mujer|novia|esposa'
                gf_words = 'marido|novio|esposo'
            elif language == 'dutch':
                bf_words = 'vrouw|vriendin'
                gf_words = 'man|bf|vriend'

            bf_count = ('GFCOUNT', len(re.findall(bf_words, text, re.I))/length)
            gf_count = ('BFCOUNT', len(re.findall(gf_words, text, re.I))/length)

            count.extend((caps, elongated, excl, interr, omg_count, heart_count, lol_count, lmfao_count, emoticon_count,
                    emoticon_score, mention_count, hashtag_count, rt_count, url_count, pic_count, avg_text_length,
                    avg_word_length, bf_count, gf_count))

            if language == 'english':
                male_rationales = ('MALRAT', len(re.findall('bro|dude|homie', text, re.I))/length)
                female_rationales = ('FEMRAT', len(re.findall('cute', text, re.I))/length)

                gender_lex_count = 0.0
                age_lex_count = 0.0
                for word in text.split():
                    if word in gender_lexicon.keys():
                        gender_lex_count += gender_lexicon[word]
                    if word in age_lexicon.keys():
                        age_lex_count += age_lexicon[word]
                gender_lex = ('GLEX', (float(gender_intercept)+gender_lex_count)/length)
                age_lex = ('ALEX', (float(age_intercept)+age_lex_count)/length)

                selected_features_LIWC = extractor.extract_features_text(text, dictionary_LIWC_en)
                selected_categories_LIWC = extractor.features_to_categories(selected_features_LIWC, dictionary_LIWC_en)
                selected_categories_LIWC_normalized = {category:selected_categories_LIWC[category]/length for category in selected_categories_LIWC}
                tuples_LIWC = extractor.parse_dict_to_tuples(selected_categories_LIWC_normalized)

                count.extend((male_rationales, female_rationales, gender_lex, age_lex))
                count.extend(tuples_LIWC)

                male_count = 0
                smale_count = 0
                sfemale_count = 0
                female_count = 0
                pmi_count = 0
                matches = 0
                for word in clean_text.preprocessor_twitter(text).split():
                    if word in dictionary_gender:
                        pmi_count += dictionary_gender[word]['pmi']
                        matches += 1
                        if dictionary_gender[word]['gender'] == 'male':
                            male_count += 1
                        elif dictionary_gender[word]['gender'] == 'female':
                            female_count += 1
                        elif dictionary_gender[word]['gender'] == 'male+':
                            smale_count += 1
                        elif dictionary_gender[word]['gender'] == 'female+':
                            sfemale_count += 1
                if matches == 0:
                    matches = 1
                pmi = ('PMI_gender', pmi_count/matches)
                male = ('male_words', male_count/matches)
                female = ('female_words', female_count/matches)
                smale = ('male+_words', smale_count/matches)
                sfemale = ('female+_words', sfemale_count/matches)

                count.extend((pmi, male, female, smale, sfemale))
            count = sorted(count)
            count = [(tag[0],"%.4f" % tag[1]) for tag in count]
            print count

        tag_count.append(count)
        if task == 'training':
            gender.append(t['gender'])
            age.append(t['age'])
        i += 1
        print str(i)+'/'+str(len(tweets))

    return tag_count, gender, age

# Save features
def save_features(datapath, tag_count, gender, age):

    if not os.path.isdir(datapath+'/data'):
        os.makedirs(datapath+'/data')
    with open(datapath+'/data/tag_count.p', 'wb') as tagfile:
        pickle.dump(tag_count, tagfile)
    with open(datapath+'/data/gender.p', 'wb') as genderfile:
        pickle.dump(gender, genderfile)
    with open(datapath+'/data/age.p', 'wb') as agefile:
        pickle.dump(age, agefile)


# Load features
def load_features(datapath):
    with open(datapath+'/data/tag_count.p', 'rb') as tagfile:
        tag_count = pickle.load(tagfile)
    with open(datapath+'/data/gender.p', 'rb') as genderfile:
        gender = pickle.load(genderfile)
    with open(datapath+'/data/age.p', 'rb') as agefile:
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
        new_tags = []
        for pos in pos_tags:
            if pos in p:
                new_tags.append((pos, p[pos]))
            elif pos not in p:
                new_tags.append((pos, 0))
        new_tags = sorted(new_tags)
        complete_tag_count.append(new_tags)
    return complete_tag_count

def complete_test_tags(test_tag_count, training_tag_count):
    pos_tags = []
    for post in training_tag_count:
        for tag in post:
            if tag[0] not in pos_tags:
                pos_tags.append(tag[0])
    complete_tag_count = []
    for post in test_tag_count:
        p = dict(post)
        new_tags = []
        for pos in pos_tags:
            if pos in p:
                new_tags.append((pos, p[pos]))
            elif pos not in p:
                new_tags.append((pos, 0))
        new_tags = sorted(new_tags)
        complete_tag_count.append([i[1] for i in new_tags])

    return complete_tag_count

# # Train and evaluate classifiers

def train_and_evaluate(complete_tag_count, prediction, predicted_class, nonpredicted_class):

    tag_total = np.array(complete_tag_count)
    predicted_final = np.array(predicted_class)
    nonpredicted_final = np.array(nonpredicted_class)

    #features_total = np.array(features)

    clf1 = linear_model.LogisticRegression(n_jobs=9)
    clf2 = ensemble.RandomForestClassifier(n_estimators=100, n_jobs=9)
    clf3 = ensemble.ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=1, random_state=0, criterion='entropy',
                                        n_jobs=9)
    clf4 = tree.DecisionTreeClassifier(max_depth=3)
    clf5 = naive_bayes.GaussianNB()
    clf6 = naive_bayes.BernoulliNB()
    clf7 = ensemble.GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=1, random_state=0)
    clf8 = ensemble.AdaBoostClassifier(n_estimators=100)
    clf9 = OneVsRestClassifier(clf4, n_jobs=9)
    clf10 = svm.SVC(kernel='linear', probability=True, C=0.05)

    eclf = ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('ext', clf3), ('dt', clf4), ('gnb', clf5),
                                                 ('bnb', clf6), ('gbc', clf7), ('ada', clf8), ('1vr', clf9), ('svc', clf10)
                                                 ], voting='soft')
    eclf2 = ensemble.VotingClassifier(estimators=[('lr', clf1), ('rf', clf2), ('ext', clf3), ('dt', clf4), ('gnb', clf5),
                                                 ('bnb', clf6), ('gbc', clf7), ('ada', clf8), ('1vr', clf9), ('svc', clf10)
                                                 ], voting='hard')

    cv = cross_validation.StratifiedKFold(predicted_final, 10)

    for clf, label in zip([clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, eclf, eclf2], ['Logistic Regression',
        'Random Forest', 'Extra Trees', 'Decision Tree','Gaussian NB', 'Bernoulli NB', 'Gradient Boosting Classifier',
        'AdaBoost', 'One vs Rest', 'SVC Linear', 'Soft Voting Ensemble', 'Hard Voting Ensemble']):
        """scores = cross_validation.cross_val_score(clf, tag_total, predicted_final, cv=cv, scoring='accuracy')
        print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), label))"""
        if prediction == 'age':
            results = cross_validation.cross_val_predict(clf, tag_total, nonpredicted_final, cv=cv)
            final_tags = []
            for i in range(len(tag_total)):
                user = tag_total[i]
                user_gender = results[i]
                if user_gender == 'M' or user_gender == 'MALE':
                    g = 0
                elif user_gender == 'F' or user_gender == 'FEMALE':
                    g = 1
                user = np.append(user, g)
                final_tags.append(user)
        else:
            final_tags = tag_total
        scores = cross_validation.cross_val_score(clf, final_tags, predicted_final, cv=cv, scoring='accuracy')
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

def model_training(complete_tag_count, prediction, predicted_class, nonpredicted_class, language):
    tag_total = np.array(complete_tag_count)
    clf = ensemble.ExtraTreesClassifier(n_estimators=1000, max_depth=None, min_samples_split=1, random_state=0, criterion='entropy',
                                        n_jobs=9)
    """if prediction == 'age':
        cv = cross_validation.StratifiedKFold(predicted_final, 10)
        results = cross_validation.cross_val_predict(clf, tag_total, nonpredicted_final, cv=cv)
        final_tags = []
        for i in range(len(tag_total)):
            user = tag_total[i]
            user_gender = results[i]
            if user_gender == 'M' or user_gender == 'MALE':
                g = 0
            elif user_gender == 'F' or user_gender == 'FEMALE':
                g = 1
            user = np.append(user, g)
            final_tags.append(user)
    else:
        final_tags = tag_total"""
    final_tags = tag_total

    clf.fit(final_tags, predicted_class)

    with open('models/'+language+'-'+prediction+'-base.p', 'wb') as clffile:
        pickle.dump(clf, clffile)


def features(datapath, language):
    tweets = extract_tweets(datapath, 'training')
    load_lexicons()
    tags, gender, age = extract_features(tweets, language, 'training', False)
    save_features(datapath, tags, gender, age)

def add_tags(tags_list, new_tags):
    new_list = []
    for i in range(0, len(tags_list)):
        tags = tags_list[i]
        tags.append(new_tags[i])
        new_list.append(tags)
    return new_list

def evaluate(datapath, prediction):
    tags, gender, age = load_features(datapath)
    if prediction == 'age':
        predicted_class = age
        training_class = gender
    elif prediction == 'gender':
        predicted_class = gender
        training_class = age
    final_tags = complete_tags(tags)
    train_and_evaluate(final_tags, prediction, predicted_class, training_class)

def train(datapath, language, prediction):
    tags, gender, age = load_features(datapath)
    if prediction == 'age':
        predicted_class = age
        training_class = gender
    elif prediction == 'gender':
        predicted_class = gender
        training_class = age
    final_tuples = complete_tags(tags)
    final_tags = []
    for tuple in final_tuples:
        final_tags.append([i[1] for i in tuple])
    model_training(final_tags, prediction, predicted_class, training_class, language)


def predict(datapath, language, age_model, gender_model):
    load_lexicons()
    tweets = extract_tweets(datapath, 'predicting')
    if gender_model == 'doc2vec':
        reload(sys)
        sys.setdefaultencoding('UTF8')
        doc2vec_tags, gender, age = extract_features(tweets, language, 'predicting', True)
        doc2vec_final = np.array(doc2vec_tags)
    tags, gender, age = extract_features(tweets, language, 'predicting', False)
    training_tags = load_features('data/pan16-author-profiling-training-dataset-'+language+'-2016-04-25')[0]
    training_tags = complete_tags(training_tags)

    total_tags = complete_test_tags(tags, training_tags)

    final_tags = np.array(total_tags)

    with open('models/'+language+'-gender-'+gender_model+'.p', 'rb') as genderfile:
         gender_clf = pickle.load(genderfile)
    if language != 'dutch':
        with open('models/'+language+'-age-'+age_model+'.p', 'rb') as agefile:
            age_clf = pickle.load(agefile)
        age_results = age_clf.predict(final_tags)
    if gender_model == 'doc2vec':
        gender_results = gender_clf.predict(doc2vec_final)
    else:
        gender_results = gender_clf.predict(final_tags)

    if language == 'spanish':
        lang = 'es'
    elif language == 'english':
        lang = 'en'
    elif language == 'dutch':
        lang = 'nl'
    for i in range(0, len(gender_results)-1):
        id = tweets[i]['fid']
        author = et.Element('author')
        author.set('id', id)
        author.set('type', 'XX')
        author.set('lang', lang)
        if language == 'dutch':
            author.set('age_group', 'XX')
        else:
            author.set('age_group', age_results[i])
        author.set('gender', gender_results[i].lower())
        doc = et.ElementTree(author)
        #doc.write(output_dir+'/'+id+'.xml')

""""if __name__ == "__main__":
    path = sys.argv[1]
    language = str(sys.argv[2])
    age_model = str(sys.argv[3])
    gender_model = str(sys.argv[4])
    output_dir = sys.argv[5]
    predict(path, language, age_model, gender_model)"""

language = 'english'
path = 'data/pan16-author-profiling-training-dataset-'+language+'-2016-04-25'
features(path, language)
train(path, language, 'gender')
train(path, language, 'age')
#predict(path, language)