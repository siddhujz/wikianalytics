import sys, os, re, gzip, string
from pprint import pprint
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf = conf)

wordcolon = re.compile(r"\w*[:]") # to match Wikipedia: File: Portal: etc
Crosswords = set()
with open("crossword.txt",'r') as F:
  for line in F: Crosswords.add(line.strip())

def ordinary(Title):
  '''
  Convert a given Title into ASCII 

  Return - the Title with '_' translated into a space,
  with %2C translated into ',' and so on; however, return
  None for Title which translates poorly, due to foreign
  characters or if it begins with "Wikipedia:" (an internal
  Wiki page)
  '''
  Title = Title.strip().replace('_',' ')
  try:
    while '%' in Title:
      percent = Title.index('%')
      before = Title[:percent]
      after =  Title[percent+3:] 
      convert = chr(eval("0x"+Title[percent+1:percent+3]))
      if convert not in string.printable: Title = None 
      else: Title = before + convert + after
  except:
    return None
  if wordcolon.match(Title): return None
  return Title

def procpart(part_iter):   
  '''
  Process one partition of the data (in this test version,
  part is the name of a compressed file, using gzip).

  Parameter part_iter is an iterator of the partition's items
         
  Return a list of (key,value) pairs where the key is
  the converted title of a wiki article, and the value
  is the raw text of that article
  '''
  ArticleList = list()
  Title = None 
  # simple state machine to isolate an article
  # 0 => looking for Title; 1 => accumulating, looking for end of article
  State = 0
  for line in part_iter:
     line = line.encode("Ascii","ignore")
     if State == 0:
       if  not line.startswith("$$$===cs5630s17===$$$===Title===$$$"):
         continue # ignore irrelevant lines
       Title = line.strip().split()[-1]  # grab the title
       pTitle = ordinary(Title) 
       if pTitle == None: 
         continue  # ignore difficultly titled articles
       Current = list()
       State = 1
       continue
     else:  # State == 1
       if not line.startswith("$$$===cs5630s17===$$$===cs5630s17===$$$"):
         Current.append(line)
         continue
       ArticleList.append( (pTitle,Current) )
       State = 0
       continue
  return iter(ArticleList)

def interlinks(Article):
  '''
  Given a (title,text-list) pair, where text-list
  is a list of the raw text lines in the article,
  return a new list (title,topic-list) where the 
  topic-list consists of all the [[wiki topic]] 
  links to other articles (the square brackets removed) 

  NOTE: currently this code only works for links 
  confined to a single line
  '''
  def cleanlink(proposed):
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
  Links = list() 
  linkfind = re.compile(r"\[\[")
  entry,text = Article
  for line in text:
      p = linkfind.finditer(line)
      for hit in p:
        _, start = hit.span()
        end = line[start:].find("]]")
        if end < 0: break
        end += start
        topic = cleanlink(line[start:end].strip())
        if topic: Links.append(topic)
  return (entry,Links) 

def countlist(keylist):
  '''
  Simple map from a (key,list) to (key,len(list)),
  so for example countlist could be used from the
  result of interlinks to make a count of the number
  of internal wikipedia links going out from each
  article
  ''' 
  key,linklist = keylist
  return (key,len(linklist))

def cullredirects(ArticleList):
  '''
  Given a list of (title,text) pairs,
  return a new list which gets rid of all 
  all the #REDIRECT wiki entries 
  '''
  NewArticleList = list()
  for entry,text in ArticleList:
    if text and text[0].startswith('#REDIRECT'): continue 
    NewArticleList.append( (entry,text) )
  return NewArticleList

def onlybikes(ArticleList):
  '''
  Given a list of (title,text) pairs,
  return a new list which gets rid of all 
  all article that DO NOT have the word unicycle
  somewhere in the text
  '''
  NewArticleList = list()
  for entry,text in ArticleList:
    for line in text:
      if 'unicycle' in line:
         NewArticleList.append( (entry,text) )
         #pprint((entry,text))
         break
  return NewArticleList

def onlytext(ArticleList):
  '''
  Given a list of (title,text) pairs, 
  return a new list (title,puretext) which is an
  attempt to extract just the natural language
  of each article. Each instance of "puretext"
  is a string with all punctuation removed 
  except for quotes in common words (don't, can't)
  and periods at the end of sentences. In the 
  "purtext" string, all letters are converted to 
  lowercase, and anything not in the Crosswords
  dictionary is discarded. 
  '''
  alphax = re.compile(r"[a-z'.]")
  TextOnlyList = dict()
  for entry,text in ArticleList:
    tokens = (' '.join(text)).lower().split()
    newtext = list()
    for token in tokens:
      if any(c not in string.printable for c in token): continue # skip nasty words
      word,prev = '',None
      for match in alphax.finditer(token):
        s,_ = match.span() 
        if prev == None or prev+1 == s:
           word += token[s]
           prev = s
        else:
           word += ' '
           prev = None 
      # word is a candidate, however we only accept some candidates              
      word = word.strip()
      candidate = word
      if word.endswith("."): candidate = word[:-1]
      if candidate not in Crosswords: continue 
      newtext.append(word)
    TextOnlyList[entry] = ' '.join(newtext)
  return TextOnlyList.items()

def getrefs(ArticleList):
  '''
  Given a list of (title,text) pairs, find all <ref ...> info </ref>
  markups, and extract the info, creating a new list of (title,info-list)
  data which has all the ref information for each article
  '''
  tagfind = re.compile(r"(<ref>)|(</ref>)|(<ref)")
  refmap = dict() 
  for entry,text in ArticleList:
    refmap[entry] = list()
    joined = '\n'.join(text)
    L = tagfind.finditer(joined)
    for hit in L:
      start,end = hit.span()
      if joined[start:end].startswith("<ref"):
        infostart = end
      elif joined[start:end].startswith("</ref"):
        refmap[entry].append(joined[infostart:start])
  return refmap.items()

def geturls(ArticleList):
  '''
  Given a list of (title,text) pairs, find all [http:website.domain/etc]
  strings, extract the URLs, and return a list of pairs (title,urllist) 
  '''
  linkfind = re.compile(r"(\[http://)|(\[https://)")
  urllist = dict() 
  for entry,text in ArticleList:
    urllist[entry] = list()
    for line in text:
      p = linkfind.search(line)
      if not p: continue
      start,end = p.span()
      q = line[end:].index(']')
      if q < 0: continue # the ] was missing
      link = line[start+1:end+q]
      link = link.split()[0]   # in case link had a blank / comment part
      urllist[entry].append(link)
  return urllist.items()

def geturis(RefList):
  '''
  Given the information from a (title,info-list) pairs list, 
  isolate just the url=http://website-etc.com/.. and return 
  a list of (title,uri-list) pairs
  '''
  UrlList = dict()
  urlfind = re.compile(r"(url=\s+)|(url=)")
  terminatefind = re.compile(r"(\s)|($)|(\|)")
  for entry,infolist in RefList:
    UrlList[entry] = list()
    for item in infolist:
      p = urlfind.search(item)
      if not p: continue
      start, end = p.span()
      q = terminatefind.search(item[end:])
      qstart, _ = q.span()
      qstart += end  # because we started later in string  
      UrlList[entry].append(item[end:qstart])
  return UrlList.items() 

def getdomains(UriList):
  '''
  Given a list of (title,uri-list) pairs, produce a new
  list of pairs (title,domain-list) which has just the 
  internet domains from the URIs in the uri-lists
  '''
  DomainList = dict()
  domainstart = re.compile(r"\w+://") 
  domainend = re.compile(r"(/)|(\?)|(\s)|(\$)")
  for entry,infolist in UriList:
    DomainList[entry] = list()
    for item in infolist:
      if not item: continue
      p = domainstart.match(item)
      if not p: continue
      _,end = p.span()
      q = domainend.search(item[end:])
      if not q: continue
      start,_ = q.span()
      start += end  # because of offset
      newitem = item[end:start]  # start after http:// go up to first / 
      if newitem not in DomainList[entry]: 
        DomainList[entry].append(newitem)
  return DomainList.items() 

def nomarkup(ArticleList):
  '''
  Given a list of (title,text) pairs, return a new list of 
  (title,text) pairs with all markup tags (<ref> .. </ref>, <br>, etc)
  removed. This also removes anything between matching tags.  
  '''
  tagfind = re.compile(r"(<\w+>)|(<\w+/>)|(</\w+>)|(<!)|(>)")
  opentagfind = re.compile(r"<\w+")
  refmap = dict() 
  for entry,text in ArticleList:
    joined = '\n'.join(text)
    newtext = list()
    openTag = None # will be the name of an open tag
    cursor = 0     # current position in joined
    # now go through all the pieces of tagless, 
    # skipping over open <ref> tags until the matching closure is found
    while True:
      p = tagfind.search(joined[cursor:])
      if not p and not openTag:
        newtext.append(joined[cursor:])
      if not p: break
      pstart,pend = p.span()
      pstart,pend = pstart+cursor,pend+cursor
      print "*** Hit", joined[pstart:pend]
      if openTag and joined[pstart:pend].startswith(closeTag):        
         openTag = None # no longer searching
         assert pend > cursor
         cursor = pend
         continue  # effectively this skips over
      if openTag: 
         assert pend > cursor
         print "*** Skipping because OpenTag =", openTag, "skipped:", joined[cursor:pend]
         cursor = pend
         continue # still searching, so skip over 
      # determine whether this is a new open tag or not
      if joined[pstart:pend].endswith("/>") or joined[pstart:pend].startswith("</"):
         assert pend > cursor
         cursor = pend
         continue # skip self-closing tags and stray closing tags
      if joined[pstart:pend].startswith(">"):
         assert pend > cursor
         cursor = pend
         continue # skip accidental match on comment closure 
      q = opentagfind.match(joined[pstart:pend])
      if not q:
         newtext.append(joined[cursor:pend])
         assert pend > cursor
         cursor = pend
         continue # skip over strange case 
      openTag = joined[pstart:pend]
      if openTag not in ("<!","<ref","<ref>"):
         openTag = None # oops, never mind
         newtext.append(joined[cursor:pend])
         assert pend > cursor
         cursor = pend
         continue # skip over harmless tags 
      if openTag == "<!": closeTag = ">"
      elif openTag == "<ref>": closeTag = "</ref>"
      elif openTag == "<ref":  closeTag = "</ref>"
      assert pend > cursor
      cursor = pend
    refmap[entry] = (''.join(newtext)).split('\n')
  return refmap.items()

def bikecount(part):
  n = 0
  S = onlybikes(cullredirects(procpart(part)))
  #pprint (getdomains(geturis(getrefs(S))))
  #pprint (getdomains(geturls(S)))
  # pprint(nomarkup(S))
  # pprint(interlinks(S))
  pprint(onlytext(S))
  n += len(S)
  return n

# test
A = sc.textFile("s3://cs5630s17-instructor/wiki-text/part002*")
B = A.mapPartitions(procpart)
C = B.map(interlinks)
D = C.map(countlist)
D.saveAsTextFile("linkcounts")
