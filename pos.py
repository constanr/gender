from nltk import word_tokenize, FreqDist, pos_tag
import nltk.data
import pattern.en, pattern.es
import clean_text

# This function tags the POS (parts-of-speech) that appear in the input text and returns counts
# for each type and text.
# Input: JSON file {text="sample text",gender="M/F"}
# Output: list of (POS, count)
def count_pos(input, language):
    if language == 'english-nltk':
        words = word_tokenize(input)
        pos = pos_tag(words)

    elif language == 'english':
        s = pattern.en.parsetree(input, relations=True, lemmata=True)
        words = []
        pos = []
        for sentence in s:
            for w in sentence.words:
                words.append(w.string)
                pos.append((w.string, clean_text.clean_pos(w.type)))

    elif language == 'spanish':
        s = pattern.es.parsetree(input, relations=True, lemmata=True)
        words = []
        pos = []
        for sentence in s:
            for w in sentence.words:
                words.append(w.string)
                pos.append((w.string, clean_text.clean_pos(w.type)))

    elif language == 'dutch':
        words = word_tokenize(input, 'dutch')
        tagger = nltk.data.load('taggers/alpino_aubt.pickle')
        pos = tagger.tag(words)

    tags = FreqDist(tag for (word, tag) in pos)
    relative_frequency = []
    for item in tags.items():
        relative_frequency.append((item[0], float(item[1])/tags.N()))
    return relative_frequency

#print count_pos('I was watching TV', 'english')
#print count_pos('I was watching TV', 'pattern')
#print count_pos('Estaba viendo la TV', 'spanish')
#print count_pos('het groene boekje', 'dutch')

