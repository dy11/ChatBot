#!/usr/bin/env Python
# coding=utf-8
import codecs
import csv
import sys
import gensim
import nltk
import numpy as np
import pickle
from scipy.linalg import norm
from queue import PriorityQueue
import re


class Sen_Index(object):
    def __init__(self, index, score):
        self.index = index
        self.score = score
        #print(self.score)
        return

    def __lt__(self,other):
        return self.score > other.score
def filter_line(text):
    text = text.lower()

    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"he's", "he is", text)
    text = re.sub(r"she's", "she is", text)
    text = re.sub(r"it's", "it is", text)
    text = re.sub(r"that's", "that is", text)
    text = re.sub(r"what's", "that is", text)
    text = re.sub(r"where's", "where is", text)
    text = re.sub(r"how's", "how is", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"\'ve", " have", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"won't", "will not", text)
    text = re.sub(r"can't", "cannot", text)
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"n'", "ng", text)
    text = re.sub(r"'bout", "about", text)
    text = re.sub(r"'til", "until", text)
    text = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,\']", "", text)

    return text

def handle_movie_data():
    lines = []
    conv_lines = []
    movie_line = codecs.open("movie_lines.txt", "r", encoding='utf-8',
                             errors='ignore')
    conv = open("movie_conversations.txt", "r")


    for line in movie_line:
        temp = line.split("+++$+++")
        text = filter_line(temp[4][1:])
        lines.append([temp[0].replace(" ",""), text])


    movie_line.close()

    with open('movie_lines.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for line in lines:
            writer.writerow(line)

    for line in conv:
        temp = line.split("+++$+++")
        temp = temp[3]
        temp = temp.replace("'","")
        temp = temp.replace("[","")
        temp = temp.replace("]","")
        conv_lines.append(temp)
        #print(temp)
    with open('conv_lines.csv', 'w', newline='') as f:
        for line in conv_lines:
            f.write(line)




def createIndex():
    #meta_lineIndex_path = os.path.join(config.data_root, 'lineIndex.csv')
    convline_dict = {}
    conv_dict = {}
    with open("movie_lines.csv", newline='')as f:
        lineIndex_reader = csv.reader(f)
        for row in lineIndex_reader:
            convline_dict[row[0]] = row[1]


    with open("conv_lines.csv", newline='')as f:
        convLine = csv.reader(f)

        for line in convLine:
            #lines = line.split(",")
            length = len(line) - 1
            last = ""
            for inx, l in enumerate(line):
                l= l[1:]
                if inx == 0:
                    last = l
                elif inx == length:
                    conv_dict[last] = l
                else:
                    conv_dict[last] = l
                    last = l
    #print(conv_dict)

    with open("movie_lines.file", "wb") as f:
        pickle.dump(convline_dict, f)

    with open("conv_lines.file", "wb") as f:
        pickle.dump(conv_dict, f)




def generate_sentece_vec():
    model_file = 'GoogleNews-vectors-negative300.bin' # load google's pre-trained word2vec model
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
    convline_dict = {}
    with open("movie_lines.file", "rb") as f:
        convline_dict = pickle.load(f)
    sentence_vec = {}


    for index in convline_dict:
        score = np.zeros(300)
        words = nltk.word_tokenize(convline_dict[index])

        for word in words:
            if word in model:
                score += model[word]
        if(len(words) != 0):
            score /= len(words)

        sentence_vec[index] = score
    with open("sentece_vec.file", "wb") as f:
        pickle.dump(sentence_vec, f)
    #print(sentence_vec["L150"])

def retrieve_sentence(sentence, model): #, model, gold
    que_size = 5

    score = np.zeros(300)
    words = nltk.word_tokenize(sentence)
    # calculate query's sentence vector
    for word in words:
        if word in model:
            score += model[word]
    if(len(words) != 0):
        score /= len(words)

    # calculate cos score
    sentence_vec = {}
    with open("sentece_vec.file", "rb") as f:
        sentence_vec = pickle.load(f)
    #high_score = -1
    que = PriorityQueue()



    # get highest cos score
    for index in sentence_vec:
        cos_sc = np.dot(score, sentence_vec[index]) / (np.linalg.norm(score) * np.linalg.norm(sentence_vec[index]))
        que.put(Sen_Index(index, cos_sc))

        """
        if que.qsize() >= que_size:
            que.put(Sen_Index(index, cos_sc))
            que.get()
        else:
            que.put(Sen_Index(index, cos_sc))
        """

        """
        if cos_sc > high_score:
            high_score = cos_sc
            res_sentence = index
        """




    convline_dict = {}
    conv_dict = {}
    with open("movie_lines.file", "rb") as f:
        convline_dict = pickle.load(f)
    with open("conv_lines.file", "rb") as f:
        conv_dict = pickle.load(f)

    #print("Q: ", sentence)


    i = 0
    res = []
    #res.append(gold)

    while not que.empty() and i < que_size:
        i += 1
        ob = que.get()
        #print(ob.index)

        if ob.index not in conv_dict:
            continue
        else:
            if len(res) < 4:
                res.append(convline_dict[conv_dict[ob.index]].replace("\n", ""))

            print("simi sentence is ", convline_dict[ob.index])
            print("cos socre is ", ob.score)
            print("Answer: ", convline_dict[conv_dict[ob.index]])

        #print(ob.score)
    return res

def test():
    goldList = []
    gl = np.load('answerList.dat')
    for index, line in enumerate(gl):
        line = str(line)
        line = filter_line(line[1:])
        goldList.append(line)


    model_file = 'GoogleNews-vectors-negative300.bin' # load google's pre-trained word2vec model
    model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
    answerList = []
    questionList = np.load('questionList.dat')
    for index, line in enumerate(questionList):
        #print(line)
        line = str(line)
        line = filter_line(line[1:])
        gold = goldList[index]
        ans = retrieve_sentence(line, model, gold)

        answerList.append(ans)

        if(index % 100 == 0):
            print(index)


    with open("retrieve_multi_answer.dat", "wb") as f:
        pickle.dump(answerList, f)



#test()
#handle_movie_data()
#generate_sentece_vec()
model_file = 'GoogleNews-vectors-negative300.bin' # load google's pre-trained word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
while(True):
    sentence = input("Q: ")
    retrieve_sentence(sentence,model)
#createIndex()




