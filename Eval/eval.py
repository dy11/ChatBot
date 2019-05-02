import argparse
import collections
import numpy as np
import os
import re
import string
import sys
import pickle
import pandas as pd
import tensorflow as tf
import time
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

OPTS = None


def parse_args():
    parser = argparse.ArgumentParser('Evaluation script for Chatbot project')
    parser.add_argument('data_file', help='Input data result file.')
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)

    return parser.parse_args()

# Load the data
def parse_data():
    lines = open('movie_lines.txt', encoding='utf-8', errors='ignore').read().split('\n')
    conv_lines = open('movie_conversations.txt', encoding='utf-8', errors='ignore').read().split('\n')

    return lines, conv_lines

def retrieve_bleu(goldList, predList):
    chencherry = nltk.translate.bleu_score.SmoothingFunction()
    sum = 0
    count = len(goldList)
    for i in range(len(goldList)):
        ref = [goldList[i].split(" ")]
        hyp = predList[i].split(" ")
        sum += nltk.translate.bleu_score.sentence_bleu(ref, hyp,smoothing_function=chencherry.method3)
    return sum/count

def retrieve_bleu2(goldList, predList):
    chencherry = nltk.translate.bleu_score.SmoothingFunction()

    sum = 0
    count = len(goldList)
    for i in range(len(goldList)):
        ref = [goldList[i].split(" ")]
        hyp1 = predList[i][0].split(" ")
        hyp2 = predList[i][1].split(" ")
        sum += max(nltk.translate.bleu_score.sentence_bleu(ref, hyp1),nltk.translate.bleu_score.sentence_bleu(ref, hyp2))
    return sum/count

def gen_bleu(goldList, predList):
    chencherry = nltk.translate.bleu_score.SmoothingFunction()

    sum = 0
    count = len(goldList)

    for i in range(len(goldList)):
        ref = [goldList[i]]
        hyp = predList[i]
        sum += nltk.translate.bleu_score.sentence_bleu(ref, hyp,smoothing_function=chencherry.method1)

    return sum / count

def cos_sim(vector_a, vector_b):

    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

def foo(gold_toks,pred_toks):
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())

    stop_words = set(stopwords.words('english'))
    filtered_toks = []
    for w in pred_toks:
        if w not in stop_words:
            filtered_toks.append(w)
    common2 = collections.Counter(gold_toks) & collections.Counter(filtered_toks)
    num_same2 = sum(common2.values())

    if num_same == 0:
        return 0, 0, 0, 0

    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)

    if num_same2 > 0:
        return f1, 1, recall, precision
    else:
        return f1, 0, recall, precision

def gen_F1(goldList, predList):

    sum = 0
    sum2 = 0
    recallSum = 0
    precisionSum = 0
    count = 0

    for i in range(len(goldList)):
        f1, bin, recall, precision = foo(goldList[i], predList[i])
        recallSum += recall
        precisionSum += precision
        sum += f1
        sum2 += bin
        count += 1

    return sum/count, sum2/count, recallSum/count, precisionSum/count

def retrieve_F1(goldList, predList):
    top1 = 0
    top2 = 0

    return top1, top2

def retrieve_eval():
    goldList = []
    top1PredList = []
    top2PredList = []
    #answerList = np.load('2000lines/answerList.dat')
    rawData = np.load('retrieve_multi_answer_demo.dat')
    print(rawData)

    for i in rawData:
        temp = []
        if i[1] and i[2]:
            goldList.append(i[0])
            if i[1] == i[0]:
                top1PredList.append(i[2])
            else:
                top1PredList.append(i[1])

            for j in range(1,3):
                temp.append(i[j])
            top2PredList.append(temp)

    print(retrieve_bleu(goldList, top1PredList))
    print(gen_F1(goldList, top1PredList))
    #print(retrieve_bleu2(goldList, top2PredList))

    #print(retrieve_F1(goldList, top1PredList))
    #print(retrieve_F1(goldList, top2PredList))

    return

def gen_eval():
    goldList = []
    predList = []

    rawData = np.load('2000lines/test.dat')
    pred1List = np.load('2000lines/pred2List.dat')
    predict2 = np.load('2000lines/predict2.dat')
    predict3 = np.load('2000lines/predict3.dat')
    ansList = np.load('2000lines/answerList2.dat')
    question = np.load('2000lines/questionList.dat')

    #print(str(predict3[0]).replace('b','').replace('\'','').split(" "))

    for n,i in enumerate(rawData):
        if i[1] and i[2]:
            goldList.append(i[1])
            #predList.append(i[2])
            #goldList.append((str(ansList.item(i))).split(" "))
            predList.append(str(predict3[n]).replace('b','').replace('\'','').split(" "))
            #predList.append(pred1List[n])

            # For m
            """
            if n == 20:
                for t in range(0,20):
                    print(question[t])
                    print(goldList[t])
                    print(predList[t])
                    print("----------------------------------------")
            """
    print(gen_bleu(goldList, predList))
    print(gen_F1(goldList, predList))

    return

def main():
    if OPTS.data_file:
        pickle_in = open(OPTS.data_file, "rb")
        rawData = pickle.load(pickle_in)

    goldList = []
    predList = []
    lines, conv_lines = parse_data()
    for i in rawData:
        goldList.append(i[0][1])
        predList.append(i[1][1])

    gen_eval()
    #retrieve_eval()


if __name__ == '__main__':
    OPTS = parse_args()
    """
    if OPTS.out_image_dir:
      import matplotlib
      matplotlib.use('Agg')
      import matplotlib.pyplot as plt
    """
    main()
