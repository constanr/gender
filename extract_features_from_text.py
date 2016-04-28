import re,pandas,csv

def load_EmoLex(path,lang):
    emoLex_mapping = {}
    emoLex = pandas.read_excel(path+"NRC-Emotion-Lexicon-v0.92-InManyLanguages-web.xlsx")
    
    emoLex = emoLex.to_dict('records')
    
    for element in emoLex:
        emotion_list = [key for key,value in element.iteritems() if value != 0 and type(value) is int]
        if len(emotion_list) == 0 or type(element[lang]) is int:
            continue
        
        emotion_list = [emotion+'_emoLex' for emotion in emotion_list]
        emoLex_mapping[element[lang]] = emotion_list
        
    return emoLex_mapping

def load_MoralDIC(path):
    with open(path+"moral-foundations-categories.txt",'r') as tsv:
        categories_LIWC = [line.strip().split('\t') for line in tsv]
    with open(path+"moral foundations dictionary.dic",'r') as tsv:
        words_MoralDIC = [line.strip().split('\t') for line in tsv]  
    categories_MoralDIC = {key: value for (key, value) in categories_LIWC}
    
    dictionary_MoralDIC = {}
    
    for word_categories in words_MoralDIC:
        word = re.sub('[*]','',word_categories[0])
        for category in word_categories[1:]:
            if category != '':
                moral_categories = category.split(' ')
        categories = [categories_MoralDIC[str(category).strip()]+'_moral' for category in moral_categories]
        dictionary_MoralDIC[word] = categories
    
    return dictionary_MoralDIC

def load_NRC_hashtag_emotion(path):
    hashtag_emotion_lexicon = {}
    with open(path+'NRC-Hashtag-Emotion-Lexicon-v0.2.txt','rb') as tsv:
        tsvin = csv.reader(tsv,delimiter='\t')
        for row in tsvin:
            if row[1] in hashtag_emotion_lexicon:
                hashtag_emotion_lexicon[row[1]][row[0]] = row[2]
            else:
                hashtag_emotion_lexicon[row[1]] = {}
                hashtag_emotion_lexicon[row[1]][row[0]] = row[2]
    return hashtag_emotion_lexicon

def load_MRC(path):
    dictionary_MRC = {}
    with open(path+"MRC_dictionary.txt") as mrc:
        lines = mrc.readlines()
        
        for line in lines:
            p = line.split()
            dictionary_MRC[p[0].lower()] = {}
            dictionary_MRC[p[0].lower()]['I'] = float(p[1])
            dictionary_MRC[p[0].lower()]['AOA'] = float(p[3])
            dictionary_MRC[p[0].lower()]['F'] = float(p[5])
            dictionary_MRC[p[0].lower()]['C'] = float(p[7])
            
    return dictionary_MRC


def load_LIWC(path):
    with open(path+"LIWC2007_Categories.txt",'r') as tsv:
        categories_LIWC = [line.strip().split('\t') for line in tsv]
    with open(path+"LIWC2007_English080730.dic",'r') as tsv:
        words_LIWC = [line.strip().split('\t') for line in tsv]
        
    categories_LIWC = {key: value for (key, value) in categories_LIWC}
    
    dictionary_LIWC = {}
    for word_categories in words_LIWC:
        word = re.sub('[*]','',word_categories[0])
        categories = [categories_LIWC[str(category).strip()]+'_LIWC' for category in word_categories[1:]]
        dictionary_LIWC[word] = categories
    
    return dictionary_LIWC

def extract_features_text(text,bag_of_features):
    selected_features = {}
    for feature in bag_of_features:
        ocurrences = text.count(feature)
        if ocurrences == 0:
            continue
        if feature not in selected_features:
            selected_features[feature] = ocurrences
        else:
            selected_features[feature] += ocurrences
    return selected_features

def identify_psychologist_feature(features,dictionary_MRC):
    imagenery = 0
    age_of_acquisition = 0
    familiarity = 0
    concreteness = 0
    matches = 0
    results = {}

    for feature in features:
        imagenery += dictionary_MRC[feature]['I']*features[feature]
        age_of_acquisition += dictionary_MRC[feature]['AOA']*features[feature]
        familiarity += dictionary_MRC[feature]['F']*features[feature]
        concreteness += dictionary_MRC[feature]['C']*features[feature]
        matches += features[feature]

    imagenery = float("{0:.2f}".format(round(imagenery / matches,2)))
    age_of_acquisition = float("{0:.2f}".format(round(age_of_acquisition / matches,2)))
    familiarity = float("{0:.2f}".format(round(familiarity / matches,2)))
    concreteness = float("{0:.2f}".format(round(concreteness / matches,2)))

    results['Imagenery'] = imagenery
    results['Age_of_Acquisition'] = age_of_acquisition
    results['Familiarity'] = familiarity
    results['Concreteness'] = concreteness

    return results

def emotion_hashtag(selected_features,hashtag_emotion_lexicon):
    emotion_user = {}
    matches = {}
    for word in selected_features:
        for emotion in hashtag_emotion_lexicon[word]:
            if emotion in emotion_user:
                emotion_user[emotion+'_NRC'] += selected_features[word]*float(hashtag_emotion_lexicon[word][emotion])
                matches[emotion+'_NRC'] += selected_features[word]
            else:
                emotion_user[emotion+'_NRC'] = selected_features[word]*float(hashtag_emotion_lexicon[word][emotion])
                matches[emotion+'_NRC'] = selected_features[word]

    score_emotion = {emotion : emotion_user[emotion]/matches[emotion] for emotion in emotion_user}
    return score_emotion

def features_to_categories(features,list_categories):
    selected_categories = {}
    for feature in features:
        categories = list_categories[feature]
        for category in categories:
            if category not in selected_categories:
                selected_categories[category] = features[feature]
            else:
                selected_categories[category] += features[feature]
    return selected_categories

def parse_dict_to_tuples(dictionary):
    tuples = [(key,dictionary[key]) for key in dictionary]
    return tuples