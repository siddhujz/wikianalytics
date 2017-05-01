#!/usr/bin/python

'''
Set up the Spark Context and erase any existing HDFS output
'''
from pyspark import SparkConf, SparkContext
from pyspark.sql import SparkSession, SQLContext, Row
from pyspark.sql.functions import split, explode, desc
from pprint import pprint
import re, subprocess, sys, random, string, os
import nltk
from nltk.tag import StanfordNERTagger, PerceptronTagger
from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.tag.perceptron import PerceptronTagger
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from operator import add

#stanford_dir = '/mnt2/stanford-ner-2016-10-31/'
#jarfile = stanford_dir + 'stanford-ner.jar'
#modelfile = stanford_dir + 'classifiers/english.all.3class.distsim.crf.ser.gz'
#stanfordNERTagger = StanfordNERTagger(model_filename=modelfile, path_to_jar=jarfile)
#stanfordNERTagger.java_options='-mx4096m'
os.environ['NLTK_DATA'] = "/mnt1/nltk_data"
os.environ['CLASSPATH'] = "/mnt1/stanford-ner-2016-10-31/stanford-ner.jar" 
os.environ['STANFORD_MODELS'] = "/mnt1/stanford-ner-2016-10-31/classifiers"
stanfordNERTagger = StanfordNERTagger('english.all.3class.distsim.crf.ser.gz')
os.environ['AP_MODEL_LOC'] = "file:/mnt1/nltk_data/taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle"
#AP_MODEL_LOC = 'file:/mnt1/nltk_data/taggers/averaged_perceptron_tagger/averaged_perceptron_tagger.pickle'

# Module-level global variables for the `tokenize` function below
PUNCTUATION = set(string.punctuation)
STOPWORDS = set(stopwords.words('english'))
#STEMMER = PorterStemmer()

# Function to break text into "tokens", lowercase them, remove punctuation and stopwords
def tokenize(data_str):
    cleaned_str = ''
    if data_str is not None and data_str != '':
        tokens = word_tokenize(data_str)
        lowercased = [t.lower() for t in tokens]
        no_punctuation = []
        for word in lowercased:
            punct_removed = ''.join([letter for letter in word if not letter in PUNCTUATION])
            no_punctuation.append(punct_removed)
        no_stopwords = [w for w in no_punctuation if not w in STOPWORDS]
        #stemmed = [STEMMER.stem(w) for w in no_stopwords]
        #return [w for w in stemmed if w]
        #return [w for w in no_stopwords if w]
        cleaned_str = ' '.join(no_stopwords)
    return cleaned_str

#tagger = PerceptronTagger()
tagger = stanfordNERTagger

conf = SparkConf()
conf.setMaster("yarn-client")
conf.setAppName("Siddhartha Kodali: Wikipedia Text Analysis")
conf.set("spark.executor.memory","2g")
sc = SparkContext(conf = conf)
subprocess.call("hdfs dfs -rm -r output",shell=True)

spark = SparkSession.builder.appName("Wikipedia Text Analysis").config("spark.sql.crossJoin.enabled","true").getOrCreate()

sqlContext = SQLContext(sc)

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

REGEX_HEADING_LEVEL2 = '^\=\=[^=].*[^=]\=\=$'
REGEX_HEADING_LEVEL3 = '^\=\=\=[^=].*[^=]\=\=\=$'
REGEX_HEADING_LEVEL4 = '^\=\=\=\=[^=].*[^=]\=\=\=\=$'
REGEX_HEADING_LEVEL5 = '^\=\=\=\=\=[^=].*[^=]\=\=\=\=\=$'
REGEX_HEADING_ALL_LEVELS = '^\=\=.*\=\=$'
def get_headings(line, REGEX_HEADING_LEVEL):
    links = re.findall(REGEX_HEADING_LEVEL, line)
    return links

#REGEX_WORDS_IN_BOLD = "^'''.*'''$"
REGEX_WORDS_IN_BOLD = "(?<='{3})[\w \s]*(?='{3})"
def get_bold_text(line):
    '''returns a list of statements(word or group of words)
    written in Bold(in the wikipedia dateset)'''
    links = re.findall(REGEX_WORDS_IN_BOLD, line)
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

def parseAndFilterArticlesFromPartion(lines):
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
                    wikiArticle['text_in_bold'].append(bold_sentence)
            if re.match(REGEX_TEXT_ONLY_START, line).group(0) != '':
                wikiArticle['pure_text'] += plain_text(line) + " "
            if end not in line.strip():
                wikiArticle['raw_text'].append(line)
                continue
            if end in line.strip():
                isNewArticle = True
                wikiArticleList.append(wikiArticle)
    return iter(wikiArticleList)


'''Read the input data'''
#file = sc.textFile("hdfs://localhost:9000/user/ec2-user/samplewikidata/part0001.gz")
#file = sc.textFile("s3://cs5630s17-instructor/wiki-text/part054*")
file = sc.textFile("s3://cs5630s17-instructor/wiki-text/part0*")
#file = sc.textFile("s3://cs5630s17-instructor/wiki-text/part0001.gz")
#file = sc.textFile("hdfs://localhost:9000/user/skodali/part0001")
rddCount = file.count()
#print("rdd count = ", rddCount)

'''----------------------------------------------------------------------------'''
'''Wikipedia articles - Title'''
print("----------------------------------------------------------------------------")
print("Wikipedia articles - Title")
rdd = file.map(lambda x: x.encode("ascii", "ignore"))
wikiArticleListRDD = rdd.mapPartitions(parseAndFilterArticlesFromPartion)
wikiArticleListRDD = wikiArticleListRDD.filter(lambda x: x is not None)
'''Get count of all wiki articles'''
wikiArticleRDDCount = wikiArticleListRDD.count()
print("Total number of Wikipedia Articles = ", wikiArticleRDDCount)
wikiArticleListRDDFirst5 = wikiArticleListRDD.take(5)
print("wikiArticleListRDDFirst5 = ", wikiArticleListRDDFirst5)

'''Create a dataframe using the wikiArticleListRDD'''
#dataframe = wikiArticleListRDD.toDF()
#dataframe.show()
#dataframe = spark.createDataFrame(wikiArticleListRDD)
'''TODO: Need to check this'''
'''/opt/spark/spark-2.1.0-bin-hadoop2.7/python/pyspark/sql/session.py:336: 
   UserWarning: Using RDD of dict to inferSchema is deprecated. Use pyspark.sql.Row instead
   warnings.warn("Using RDD of dict to inferSchema is deprecated.'''


'''----------------------------------------------------------------------------'''
'''Wikipedia articles - wikilinks'''
'''return number of wikilinks for each rdd'''
print("----------------------------------------------------------------------------")
print("Wikipedia articles - wikilinks")
def lenLinks(wordlist):
    return len(wordlist)

wikilinksRDD = wikiArticleListRDD.map(lambda x:(x['title'], x['wikilinks']))
wikilinksCountRDD = wikilinksRDD.map(lambda (x, y):(x, lenLinks(y)))

wikilinksTopN = wikilinksCountRDD.map(lambda (k,v): (v,k)).sortByKey(False)
wikilinksTopNFiltered = wikilinksTopN.filter(lambda (count, title): title is not None and title != "")
print("Wikipedia articles - wikilinks - Top 20 articles with most wikilinks")
for (count, title) in wikilinksTopNFiltered.take(20):
    print("%s: %i" % (title, count))

print("***********************************")
wikilinksRDD = wikiArticleListRDD.map(lambda x:x['wikilinks'])
wikilinksCountRDD = wikilinksRDD.map(lenLinks)
print("Total number of Wikipedia Links = ", wikilinksCountRDD.sum())

'''Get count of wikilinks for first RDD'''
wikilinksCountRDDFirst = wikilinksCountRDD.take(1)
#print("wikilinksCountRDDFirst = ", wikilinksCountRDDFirst)

'''Get mean of wikilinks for all WikiArticles'''
wikilinksCountRDDMean = wikilinksCountRDD.mean()
print("mean = ", wikilinksCountRDDMean)

'''Get min of wikilinks for all WikiArticles'''
wikilinksCountRDDMin = wikilinksCountRDD.min()
print("min = ", wikilinksCountRDDMin)

'''Get max of wikilinks for all WikiArticles'''
wikilinksCountRDDMax = wikilinksCountRDD.max()
print("max = ", wikilinksCountRDDMax)

'''Get stdev of wikilinks for all WikiArticles'''
wikilinksCountRDDStdev = wikilinksCountRDD.stdev()
print("stdev = ", wikilinksCountRDDStdev)


'''----------------------------------------------------------------------------'''
'''Wikipedia articles - headings'''
'''return number of headings for each rdd'''
print("----------------------------------------------------------------------------")
print("Wikipedia articles - headings")
'''Wikipedia articles - headings'''
def lenHeadings(headingslist):
    return len(headingslist)

headingsRDD = wikiArticleListRDD.map(lambda x:(x['title'], x['headings']))
headingsCountRDD = headingsRDD.map(lambda (x, y):(x, lenHeadings(y)))

headingsTopN = headingsCountRDD.map(lambda (k,v): (v,k)).sortByKey(False)
headingsTopNFiltered = headingsTopN.filter(lambda (count, title): title is not None and title != "")
print("Wikipedia articles - headings - Top 20 articles with most headings")
for (count, title) in headingsTopNFiltered.take(20):
    print("%s: %i" % (title, count))

print("***********************************")
headingsRDD = wikiArticleListRDD.map(lambda x:x['headings'])
headingsCountRDD = headingsRDD.map(lenHeadings)
print("Total number of Headings/Sub Headings = ", headingsCountRDD.sum())

'''Get count of headings for first RDD'''
headingsCountRDDFirst = headingsCountRDD.take(1)
#print("headingsCountRDDFirst = ", headingsCountRDDFirst)

'''Get mean of headings for all WikiArticles'''
headingsCountRDDMean = headingsCountRDD.mean()
print("mean = ", headingsCountRDDMean)

'''Get min of headings for all WikiArticles'''
headingsCountRDDMin = headingsCountRDD.min()
print("min = ", headingsCountRDDMin)

'''Get max of headings for all WikiArticles'''
headingsCountRDDMax = headingsCountRDD.max()
print("max = ", headingsCountRDDMax)

'''Get stdev of headings for all WikiArticles'''
headingsCountRDDStdev = headingsCountRDD.stdev()
print("stdev = ", headingsCountRDDStdev)


'''----------------------------------------------------------------------------'''
'''Wikipedia articles - bold text citations'''
'''return number of bold text citations for each rdd'''
print("----------------------------------------------------------------------------")
print("Wikipedia articles - bold text")
'''Wikipedia articles - bold text citations'''
def lenBoldTextCitations(boldTextList):
    return len(boldTextList)

boldTextRDD = wikiArticleListRDD.map(lambda x:(x['title'], x['text_in_bold']))
boldTextCountRDD = boldTextRDD.map(lambda (x, y):(x, lenBoldTextCitations(y)))

boldTextTopN = boldTextCountRDD.map(lambda (k,v): (v,k)).sortByKey(False)
boldTextTopNFiltered = boldTextTopN.filter(lambda (count, title): title is not None and title != "")
print("Wikipedia articles - bold text - Top 20 articles with most bold words")
for (count, title) in boldTextTopNFiltered.take(20):
    print("%s: %i" % (title, count))

print("***********************************")
boldTextRDD = wikiArticleListRDD.map(lambda x:x['text_in_bold'])
boldTextCountRDD = boldTextRDD.map(lenBoldTextCitations)
print("Total number of Bold Text Citations = ", boldTextCountRDD.sum())

'''Get count of boldText citations for first RDD'''
boldTextCountRDDFirst = boldTextCountRDD.take(1)
#print("boldTextCountRDDFirst = ", boldTextCountRDDFirst)

'''Get mean of boldText citations for all WikiArticles'''
boldTextCountRDDMean = boldTextCountRDD.mean()
print("mean = ", boldTextCountRDDMean)

'''Get min of boldText citations for all WikiArticles'''
boldTextCountRDDMin = boldTextCountRDD.min()
print("min = ", boldTextCountRDDMin)

'''Get max of boldText citations for all WikiArticles'''
boldTextCountRDDMax = boldTextCountRDD.max()
print("max = ", boldTextCountRDDMax)

'''Get stdev of boldText citations for all WikiArticles'''
boldTextCountRDDStdev = boldTextCountRDD.stdev()
print("stdev = ", boldTextCountRDDStdev)


'''----------------------------------------------------------------------------'''
'''Wikipedia articles - NLTK Wordnet Lemmatize Stopwords'''
#print("----------------------------------------------------------------------------")
#print("Wikipedia articles - NLTK Wordnet Lemmatize Stopwords")
''' Stop words usually refer to the most common words in a language, 
    there is no single universal list of stop words used.
    by all natural language processing tools.
    Reduces Dimensionality.
    removes stop words '''
def remove_stops(wikiArticle):
    # expects a string
    data_str = wikiArticle['pure_text']
    stops = set(stopwords.words("english"))
    list_pos = 0
    cleaned_str = ''
    text = data_str.split()
    for word in text:
        if word not in stops:
            # rebuild cleaned_str
            if list_pos == 0:
                cleaned_str = word
            else:
                cleaned_str = cleaned_str + ' ' + word
            list_pos += 1
    wikiArticle['pure_text'] = cleaned_str
    return wikiArticle

''' Lemmatise different forms of a word(families of derivationally related words with similar meanings) '''
def lemmatize(wikiArticle):
    data_str = wikiArticle['pure_text']
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
    wikiArticle['pure_text'] = cleaned_str        
    return wikiArticle

''' Part-of-speech(POS) tagging - Tag words using POS Tagging,
    keep just the words that are tagged Nouns, Adjectives and Verbs '''
def tag_and_remove(wikiArticle):
    data_str = wikiArticle['pure_text']
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
    wikiArticle['pure_text'] = cleaned_str        
    return wikiArticle

#wikiArticleListLemmatizedRDD = wikiArticleListRDD.map(lemmatize)
#print("wikiArticleListLemmatizedRDD.take(2) = ", wikiArticleListLemmatizedRDD.take(2))
#print("--------------")
#print("wikiArticleListLemmatizedRDD.take(2)[1]['pure_text'] = ", wikiArticleListLemmatizedRDD.take(2)[1]['pure_text'])


'''----------------------------------------------------------------------------'''
'''Wikipedia articles - Stanford NER Tagging'''
'''return words along with Stanford NER tags'''
print("----------------------------------------------------------------------------")
print("Wikipedia articles - NER Tags")
'''Wikipedia articles - Stanford NER tags'''
def get_ner_tags(wikiArticle):
    '''Returns words with NER tags using
    Stanford NER(Named Entity Recognizer) Tagger'''
    return stanfordNERTagger.tag(wikiArticle['pure_text'].split())

tagsRDD = wikiArticleListRDD.map(get_ner_tags)
#print("tagsRDDFirst5 = ", tagsRDD.take(5))


''' returns title to as a dict object '''
def parseTop250IMDBMovieTitles(title):
    title = ordinary(title.strip())
    return title

''' returns just the title and pure_text in a dictionary '''
def getTitleAndPuretext(wikiArticle):
    title = wikiArticle['title']
    pure_text = wikiArticle['pure_text']
    return (title, pure_text)

simpleWikiListRDD = wikiArticleListRDD.map(getTitleAndPuretext)
simpleWikiListRDD = simpleWikiListRDD.map(lambda x: Row(title=x[0], pure_text=x[1]))
simpleWikiDF = sqlContext.createDataFrame(simpleWikiListRDD)
print("----------------------------------------------------------------------------")
simpleWikiDF.show()

print("simpleWikiDF.printSchema() = ")
simpleWikiDF.printSchema()


''' Top rated 250 movie names loaded into an RDD '''
top250RDD = sc.textFile("top250.txt")


'''----------------------------------------------------------------------------'''
'''Top 250 IMDB Movies - Title'''
print("----------------------------------------------------------------------------")
print("Top 250 IMDB Movies - Title")
top250IMDBMovieListRDD = top250RDD.map(parseTop250IMDBMovieTitles).zipWithIndex()
top250IMDBMovieListRDD = top250IMDBMovieListRDD.map(lambda x: Row(zippedindex=x[1], title=x[0]))
'''Get count of Top 250 IMDB Movies List RDD'''
top250IMDBMovieDF = sqlContext.createDataFrame(top250IMDBMovieListRDD)
print("top250IMDBMovieDF.printSchema() = ")
top250IMDBMovieDF.printSchema()
#print("top250IMDBMovieDF.count() == ", top250IMDBMovieDF.count())
top250IMDBMovieDF.show()


'''----------------------------------------------------------------------------'''
'''Top 250 IMDB Movies - Title - After Join'''
print("----------------------------------------------------------------------------")
print("Top 250 IMDB Movies - Title - After Join")
#joinDF = top250IMDBMovieDF.join(simpleWikiDF, top250IMDBMovieDF.title == simpleWikiDF.title, 'left_outer').select(top250IMDBMovieDF.title, simpleWikiDF.pure_text).collect()
joinDF = top250IMDBMovieDF.join(simpleWikiDF, top250IMDBMovieDF.title == simpleWikiDF.title).select(top250IMDBMovieDF.title, simpleWikiDF.pure_text)
joinDF.show()

print("joinDF.printSchema() = ")
joinDF.printSchema()
print("Top 250 IMDB Movies - Found in Wikipedia")
print("joinDF.count() = ", joinDF.count())

joinDFRDD = joinDF.rdd.map(tuple)
#print("simpleWikiListRDD.lookup('Olaf Renn') = ", simpleWikiListRDD.lookup('Olaf Renn'))
#Olaf Renn,Elsie Andrews,George Grey Andrews


'''----------------------------------------------------------------------------'''
'''Top 250 IMDB Movies - Sentiment Analysis'''
print("----------------------------------------------------------------------------")
print("Top 250 IMDB Movies - Sentiment Analysis")
sentiment_dictionary = {}
for line in open("AFINN-111.txt"):
    word, score = line.split('\t')
    sentiment_dictionary[word] = int(score)
''' Do sentiment analysis on a group of sentences
    and return the sentiment scores (pos, neg)'''
def sentiment_analysis(pure_text):
    result = []
    if pure_text != None:
        for sentence in sent_tokenize(pure_text):
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

''' wikiTuple is a tuple with (title, pure_text) '''
def sentiment(wikiTuple):
    return sentiment_analysis(wikiTuple[1])

moviesSentimentRDD = joinDFRDD.map(lambda (x,y): (x, sentiment_analysis(y)))

''' Return number of positive, negative and neutral sentances
 in the form of a tuple for pure_text of a wikipedia article '''
def sentenceSentimentCount(sentimentArr):
    pos = 0
    neg = 0
    neu = 0
    for [x, y] in sentimentArr:
        if (x + y) > 0:
            pos += 1
        elif (x + y) < 0:
            neg += 1
        else:
            neu += 1
    return (pos, neg, neu)

moviesSentimentRDDFiltered = moviesSentimentRDD.map(lambda (x,y): (x, sentenceSentimentCount(y)))
for (title, (pos, neg, neu)) in moviesSentimentRDDFiltered.collect():
    print("%s: (%i %i %i)" % (title, pos, neg, neu))



