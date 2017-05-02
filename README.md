# wikianalytics
Text Analysis of Wikipedia dump using Apache Spark, Amazon s3

Technologies, Tools and Libraries used:
- Amazon EMR cluster
- Spark
- Python
- Amazon s3
- nltk
- Stanford NER Tagger 

This project is mainly focused on parsing of the wikipedia dump(around 58GB of raw text which split across 750 files of equal size) and analyzing the data. 
Some of the findings include
- Number of Wikipedia articles
- Number of headings/sub headings
- Count of text phrases in 'Bold'
- Number of internal wikilinks (links to other wikipedia pages)
- Top 20 articles, mean, min, max, stdev of the above findings
- NER tagging(Used Stanford NERTagger) of the wikipedia text
- Word count - Get the top 20 most used words in wikipedia excluding the stopwords(commonly used words - taken from NLTK library)
- Get the top 250 movies in IMDB and get the wikipedia articles for those movies - Sentiment analysis of the text in these articles to find if they can be related to the rankings given to them in IMDB
