#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  3 21:54:41 2017

@author: kaliis
"""

import re
import sys
import json
import string
import nltk
from imdbpie import Imdb
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tag import StanfordNERTagger
from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize

imdb = Imdb()
imdb = Imdb(anonymize=True) # to proxy requestsfrom nltk.tag import StanfordNERTagger

tagger = PerceptronTagger()

stanford_dir = '/home/kaliis/workdir/cloud/working/stanford-ner-2016-10-31/'
jarfile = stanford_dir + 'stanford-ner.jar'
modelfile = stanford_dir + 'classifiers/english.all.3class.distsim.crf.ser.gz'
stanfordNERTagger = StanfordNERTagger(model_filename=modelfile, path_to_jar=jarfile)

wordcolon = re.compile(r"\w*[:]") # to match Wikipedia: File: Portal: etc
def ordinary(title):
    '''
    Convert a given title into ASCII 

    Return - the title with '_' translated into a space,
    with %2C translated into ',' and so on; however, return
    None for title which translates poorly, due to foreign
    characters or if it begins with "Wikipedia:" (an internal
    Wiki page)
    '''
    title = title.strip().replace('_',' ')
    try:
        while '%' in title:
            percent = title.index('%')
            before = title[:percent]
            after =  title[percent+3:] 
            convert = chr(eval("0x"+title[percent+1:percent+3]))
            if convert not in string.printable: title = None 
            else: title = before + convert + after
    except:
        return None
    if wordcolon.match(title): return None
    return title

def cleanWikiLink(proposed):
    '''
    Local function that cleans up a link, from 
    various forms, eg. "Category:junk", "cat|Cat",
    "\xf3a5","Topic#subtopic|", etc. 
    
    Either returns None (too hard of a topic), 
    or the first topic before the | separator
    '''
    if '|' in proposed:
        proposed = proposed.split('|')[0] 
    if '#' in proposed:
        proposed = proposed.split('#')[0] 
    if ':' in proposed:
        return None
    if any(c not in string.printable for c in proposed):
        return None
    return proposed

REGEX_WIKILINK = '(?<=\[\[).*?(?=\]\])'
def get_wikilinks(line):
    '''returns a list of statements(word or group of words)
    which have internal links(in the wikipedia dateset)'''
    links = re.findall(REGEX_WIKILINK, line)
    return links

REGEX_WORDS_IN_BOLD = "'(?<='{3})[\w \s]*(?='{3})"
def get_bold_text(line):
    '''returns a list of statements(word or group of words)
    written in Bold(in the wikipedia dateset)'''
    links = re.findall(REGEX_WORDS_IN_BOLD, line)
    return links

REGEX_HEADING_LEVEL2 = '^\=\=[^=].*[^=]\=\=$'
REGEX_HEADING_LEVEL3 = '^\=\=\=[^=].*[^=]\=\=\=$'
REGEX_HEADING_LEVEL4 = '^\=\=\=\=[^=].*[^=]\=\=\=\=$'
REGEX_HEADING_LEVEL5 = '^\=\=\=\=\=[^=].*[^=]\=\=\=\=\=$'
REGEX_HEADING_ALL_LEVELS = '^\=\=.*\=\=$'
def get_headings(line, REGEX_HEADING_LEVEL):
    links = re.findall(REGEX_HEADING_LEVEL, line)
    return links

REGEX_TEXT_IN_BRACES = '(?<=\{\{).*?(?=\}\})'
REGEX_TEXT_WITH_STARTING_BRACES = '(?<=\{\{).*?$'
REGEX_TEXT_WITH_ENDING_BRACES = '^.*?(?=\}\})'
REGEX_TEXT_ONLY_START = '^(\[\[)(?!File:)|^(\w*)'
REGEX_REF = '(?<=\<ref).*?(?=\</ref>)'
REGEX_REF_START = '(?<=\<ref).*?$'
REGEX_REF_END = '^.*?(?=\</ref>)'
def plain_text(line):
    '''Remove text in braces - replace with empty string'''
    line = re.sub(REGEX_TEXT_IN_BRACES, '', line)
    '''Remove braces - replace with empty string'''
    line = re.sub("\{\{\}\}", '', line)
    '''Remove text after starting braces - replace with empty string'''
    line = re.sub(REGEX_TEXT_WITH_STARTING_BRACES, '', line)
    '''Remove starting braces - replace with empty string'''
    line = re.sub("\{\{", '', line)
    '''Remove text before closing braces - replace with empty string'''
    line = re.sub(REGEX_TEXT_WITH_ENDING_BRACES, '', line)
    '''Remove ending braces - replace with empty string'''
    line = re.sub("\}\}", '', line)
    '''Remove text in between ref tags - replace with empty string'''
    line = re.sub(REGEX_REF, "", line)
    '''Remove ref tags - replace with empty string'''
    line = re.sub("<ref</ref>", "", line)
    '''Remove text after starting ref tags - replace with empty string'''
    line = re.sub(REGEX_REF_START, "", line)
    '''Remove starting ref tags - replace with empty string'''
    line = re.sub("<ref", "", line)
    '''Remove text before closing ref tags - replace with empty string'''
    line = re.sub(REGEX_REF_END, "", line)
    '''Remove closing ref tags - replace with empty string'''
    line = re.sub("</ref>", "", line)
    '''Remove text other than words and following special characters - replace with empty string'''
    pure_text = re.sub('[^\w\s.!,?]', '', line)
    return pure_text

def get_ner_tags(text):
    '''Return a list of word,tag determined by using 
    Stanford NER(Named Entity Recognizer) Tagger'''
    return stanfordNERTagger.tag(text.split())

#Open the File
f = open("part0001")
lines = f.readlines()

start = "$$$===cs5630s17===$$$===Title===$$$"
end = "$$$===cs5630s17===$$$===cs5630s17===$$$"

#Create a WikiArticle List
wikiArticleList = list()
isNewArticle = True
for line in lines:
    if isNewArticle:
        if start in line.strip():
            isNewArticle = False
            '''Initialize an article'''
            wikiArticle = dict()
            '''Get and Set the title of the WikiArticle'''
            '''Initialize the wikilinks array'''
            '''Initialize the headings array'''
            parts = line.strip().split(" ")
            wikiArticle['title'] = ordinary(parts[-1])
            wikiArticle['wikilinks'] = list()
            wikiArticle['headings'] = list()
            wikiArticle['headings_level2'] = list()
            wikiArticle['headings_level3'] = list()
            wikiArticle['headings_level4'] = list()
            wikiArticle['headings_level5'] = list()
            wikiArticle['pure_text'] = ""
            wikiArticle['ner_tags'] = list()
            wikiArticle['raw_text'] = list()
            wikiArticle['text_in_bold'] = list()
            continue
    else:
        '''Get all the wikilinks present in the line'''
        wikilinks = get_wikilinks(line)
        if len(wikilinks) != 0:
            for wikilink in wikilinks:
                topic = cleanWikiLink(wikilink)
                if topic:
    	                wikiArticle['wikilinks'].append(topic)
        '''Get all the headings present in the line'''
        headings = get_headings(line, REGEX_HEADING_ALL_LEVELS)
        headings_level2 = get_headings(line, REGEX_HEADING_LEVEL2)
        headings_level3 = get_headings(line, REGEX_HEADING_LEVEL3)
        headings_level4 = get_headings(line, REGEX_HEADING_LEVEL4)
        headings_level5 = get_headings(line, REGEX_HEADING_LEVEL5)
        if len(headings) != 0:
            for heading in headings:
                '''Remove all leading and trailing equal signs and append it to the headings list'''
                wikiArticle['headings'].append(heading.strip("="))
        if len(headings_level2) != 0:
            for heading in headings_level2:
                '''Remove all leading and trailing equal signs and append it to the headings_level2 list'''
                wikiArticle['headings_level2'].append(heading.strip("="))
        if len(headings_level3) != 0:
            for heading in headings_level3:
                '''Remove all leading and trailing equal signs and append it to the headings_level3 list'''
                wikiArticle['headings_level3'].append(heading.strip("="))
        if len(headings_level4) != 0:
            for heading in headings_level4:
                '''Remove all leading and trailing equal signs and append it to the headings_level4 list'''
                wikiArticle['headings_level4'].append(heading.strip("="))
        if len(headings_level5) != 0:
            for heading in headings_level5:
                '''Remove all leading and trailing equal signs and append it to the headings_level5 list'''
                wikiArticle['headings_level5'].append(heading.strip("="))
        '''Get the text that is in Bold'''
        bold_words = get_bold_text(line)
        if len(bold_words) != 0:
            for bold_sentence in bold_words:
                '''Remove all leading and trailing "'" signs and append it to the bold_text list'''
                #print("bold_sentence = ", bold_sentence)
                bold_sentence = bold_sentence.strip("'")
                wikiArticle['text_in_bold'].append(bold_sentence)
        if re.match(REGEX_TEXT_ONLY_START, line).group(0) != '':
            wikiArticle['pure_text'] += plain_text(line) + " "
        if end not in line.strip():
            wikiArticle['raw_text'].append(line)
            continue
        if end in line.strip():
            isNewArticle = True
            if wikiArticle['pure_text'] != "":
                wikiArticle['ner_tags'] = get_ner_tags(wikiArticle['pure_text'])
            wikiArticleList.append(wikiArticle)
            break

wikiArticleCount = len(wikiArticleList)
#print "No. of wikiArticles = ", len(wikiArticleList)
print "No. of wikiArticles = ", wikiArticleCount
print "text_in_bold---------------------------------------------------------------------"
print wikiArticleList[0]['text_in_bold']
print "pure_text********************************************************************"
print wikiArticleList[0]['pure_text']
print "Stanford NER Tags********************************************************************"
#print wikiArticleList[0]['ner_tags']

#print "********************************************************************"
#Remove for wikiArticle in wikiArticleList:
#Remove     print "wikiArticle['title'] = ", wikiArticle['title']

#Remove wikilinksCount = 0
#Remove for wikiArticle in wikiArticleList:
#Remove     wikilinksCount += len(wikiArticle['wikilinks'])
#Remove print "Total number of wikilinks = ", wikilinksCount
#Remove print "Average number of wikilinks per wikiArticle = ", wikilinksCount//wikiArticleCount

#Remove headingsCount = 0
#Remove headings_level2Count = 0
#Remove headings_level3Count = 0
#Remove headings_level4Count = 0
#Remove headings_level5Count = 0
#Remove for wikiArticle in wikiArticleList:
#Remove     headingsCount += len(wikiArticle['headings'])
#Remove     headings_level2Count += len(wikiArticle['headings_level2'])
#Remove     headings_level3Count += len(wikiArticle['headings_level3'])
#Remove     headings_level4Count += len(wikiArticle['headings_level4'])
#Remove     headings_level5Count += len(wikiArticle['headings_level5'])    
#Remove print "Total number of headings = ", headingsCount
#Remove print "Total number of level 2 headings = ", headings_level2Count
#Remove print "Total number of level 3 headings = ", headings_level3Count
#Remove print "Total number of level 4 headings = ", headings_level4Count
#Remove print "Total number of level 5 headings = ", headings_level5Count
#Remove print(wikiArticleList[0]['headings'])


''' Stop words usually refer to the most common words in a language, 
    there is no single universal list of stop words used.
    by all natural language processing tools.
    Reduces Dimensionality.
    removes stop words '''
#def remove_stops(data_str):
#    # expects a string
#    stops = set(stopwords.words("english"))
#    list_pos = 0
#    cleaned_str = ''
#    text = data_str.split()
#    for word in text:
#        if word not in stops:
#            # rebuild cleaned_str
#            if list_pos == 0:
#                cleaned_str = word
#            else:
#                cleaned_str = cleaned_str + ' ' + word
#            list_pos += 1
#    return cleaned_str
def remove_stops(data_str):
    cleaned_str = ''
    if data_str is not None and data_str != '':
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(data_str)
        filtered_words = [word for word in word_tokens if word not in stop_words]
        #stops = stopwords.words('english')
        #filtered_words = [word for word in word_list if word not in stops]
        cleaned_str = ' '.join(filtered_words)
    return cleaned_str

''' Lemmatise different forms of a word(families of derivationally related words with similar meanings) '''
def lemmatize(data_str):
    # expects a string
    list_pos = 0
    cleaned_str = ''
    lmtzr = WordNetLemmatizer()
    text = data_str.split()
    tagged_words = tagger.tag(text)
    for word in tagged_words:
        if 'v' in word[1].lower():
            lemma = lmtzr.lemmatize(word[0], pos='v')
        else:
            lemma = lmtzr.lemmatize(word[0], pos='n')
        if list_pos == 0:
            cleaned_str = lemma
        else:
            cleaned_str = cleaned_str + ' ' + lemma
        list_pos += 1
    return cleaned_str

''' Part-of-speech(POS) tagging - Tag words using POS Tagging,
    keep just the words that are tagged Nouns, Adjectives and Verbs '''
def tag_and_remove(data_str):
    cleaned_str = ' '
    # noun tags
    nn_tags = ['NN', 'NNP', 'NNP', 'NNPS', 'NNS']
    # adjectives
    jj_tags = ['JJ', 'JJR', 'JJS']
    # verbs
    vb_tags = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    nltk_tags = nn_tags + jj_tags + vb_tags

    # break string into 'words'
    text = data_str.split()

    # tag the text and keep only those with the right tags
    tagged_text = tagger.tag(text)
    for tagged_word in tagged_text:
        if tagged_word[1] in nltk_tags:
            cleaned_str += tagged_word[0] + ' '

    return cleaned_str

tempstr = "Hello, how are you doing? what are you upto? Sam organizes everything. My friend is not organized. Cars are usually found everywhere in the US"
#data_str = wikiArticleList[0]['pure_text']
a = wikiArticleList[0]['pure_text']
#a = tempstr
print("------------aaaa--------------")
print("wikiArticleList[0][pure_text] = ", a)
b = remove_stops(a)
print("------------bbbb--------------")
print("remove_stops(wikiArticleList[0]['pure_text']) = ", b)
c = lemmatize(b)
print("------------cccc--------------")
#print("lemmatize(remove_stops(wikiArticleList[0]['pure_text'])) = ", c)
d = tag_and_remove(c)
print("------------dddd--------------")
#print("tag_and_remove(lemmatize(remove_stops(wikiArticleList[0]['pure_text']))) = ", d)


sentiment_dictionary = {}
for line in open("AFINN-111.txt"):
    word, score = line.split('\t')
    sentiment_dictionary[word] = int(score)
''' Do sentiment analysis on a group of sentences
    and return the sentiment scores (pos, neg)'''
def sentiment_analysis(data_str):
    result = []
    for sentence in sent_tokenize(data_str):
        pos = 0
        neg = 0
        for word in word_tokenize(sentence):
            score = sentiment_dictionary.get(word, 0)
            if score > 0:
                pos += score
            if score < 0:
                neg += score
        result.append([pos, neg])
    return result

result = sentiment_analysis("Srini is the most peaceful person. He is very lazy.")
print("-------------Sentiment Analysis-----------------")
for s in result: print(s)

#print("-------------ImdbPie-----------------")
#top250 = imdb.top_250()
#print(top250[0]['title'])
#file = open('top250.txt', 'w+')
#for movie in top250:
#    print("Movie = ", movie['title'])
#    file.write("\n" + movie['title'].encode('utf8'))
    
