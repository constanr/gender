from nltk import word_tokenize
import nltk
import json

# This function tags the POS (parts-of-speech) that appear in the input text and returns counts
# for each type and text.
# Input: JSON file {text="sample text",gender="M/F"}
# Output: list of (POS, count)
def count_pos(input):
    words = word_tokenize(input)
    pos = nltk.pos_tag(words)
    tags = nltk.FreqDist(tag for (word, tag) in pos)
    relative_frequency = []
    for item in tags.items():
        relative_frequency.append((item[0], float(item[1])/tags.N()))
    return relative_frequency
