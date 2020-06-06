import xml.etree.ElementTree as ET
from string import punctuation
import sys
import nltk
import filecmp
from nltk.tag import StanfordPOSTagger
import nltk
from nltk.corpus import stopwords


def parseSentences(root):
    sentences = []
    for sentence in root.iter("sentence"):
            text = sentence.find("text")
            
            aspects = []
            for aspectTerm in sentence.iter("aspectTerm"):
                aspects.append(aspectTerm.attrib)
            sentences.append({"text":text.text, "aspects": aspects})

    return sentences


removeStopWord=False
filename = sys.argv[1]
if(len(sys.argv)==3):
    removeStopWord = sys.argv[2]=='-nostopwords'

print("Processing %s..." % filename)

def createIOB2():

    tree = ET.parse(filename)
    root = tree.getroot()
    sentences =  parseSentences(root)


    print(len(sentences))
    tokenizedSentences = []
    for sentence in sentences:
        text =  sentence['text']#.replace(',','').replace('.',' ')
        
        aspects=[]
        for aspect in sentence['aspects']:
            aspects.append(aspect['term'].split())
        
        #text = nltk.word_tokenize(text)
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        text = tokenizer.tokenize(text)
        if removeStopWord:
            text=[w for w in text if not w in  stopwords.words('english')]
        #standfordPosTagger = StanfordPOSTagger('standfordPosTagger\english-bidirectional-distsim.tagger','standfordPosTagger\stanford-postagger.jar')
        #tokenizedSentence=standfordPosTagger.tag(text)
        tokenizedSentence = nltk.pos_tag(text)
        #print('Done POS tagging ', sentence['text'])
        #print('Now adding iob2 ', sentence['text'])
        for i in range(len(tokenizedSentence)):
            for aspect in aspects:
                if len(tokenizedSentence[i]) <3 and tokenizedSentence[i][0] in aspect:
                    if aspect.index(tokenizedSentence[i][0]) == 0:
                        tokenizedSentence[i]=tokenizedSentence[i] +('B-A',)
                    else:
                        tokenizedSentence[i]=tokenizedSentence[i] +('I-A',)
            if len(tokenizedSentence[i]) < 3: # it had no aspect
                tokenizedSentence[i]=tokenizedSentence[i] +('O',)

            assert(len(tokenizedSentence[i])==3)
            if len(tokenizedSentence[i]) > 3:
                print(aspects)
                print(tokenizedSentence)
                exit()

        tokenizedSentences.append(tokenizedSentence)
    
    assert(len(tokenizedSentences)==len(sentences))
    return tokenizedSentences

iob2Sentences = createIOB2()

if removeStopWord:
    filename=filename+'-nostopwords'

with open(filename+'.iob','w',encoding='utf-8') as out:
    for i in range(0,len(iob2Sentences)):
        for word in iob2Sentences[i]:
            out.write('\t'.join(word)+'\n')
        out.write('\n')

print('Convertion is complete')
#assert(filecmp.cmp("original_%s.iob" % filename,"%s.iob" % filename))