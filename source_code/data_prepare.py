# -*- coding: utf-8 -*-
import operator
from os import makedirs
from os.path import exists
import argparse
from configs import *
import pickle
import numpy as np
import re
from random import shuffle
import string
import struct

stop_words = {"-lrb-", "-rrb-", "-"}
unk_words = {"unk", "<unk>"}

def load_vocab(src_path, tgt_path):
    vocab_dict = {}
    vocab_list = []
    with open(src_path, 'r') as src:
        for line in src_path:
            words = line.lower().split()
            for word in words:
                if word not in vocab_dict:
                    vocab_dict[word] = 1
                    vocab_list.append(word)
                else: 
                    vocab_dict[word] += 1

    with open(tgt_path, 'r') as src:
        for line in src_path:
            words = line.lower().split()
            for word in words:
                if word not in vocab_dict:
                    vocab_dict[word] = 1
                    vocab_list.append(word)
                else: 
                    vocab_dict[word] += 1
    return vocab_dict, vocab_list
    pass

def to_dict(xys, dic):
    # dict should not consider test set!!!!!
    for xy in xys:
        sents, summs = xy
        y = summs[0]
        for w in y:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
                
        x = sents[0]
        for w in x:
            if w in dic:
                dic[w] += 1
            else:
                dic[w] = 1
    return dic

def prepare_dict(src_path, tgt_path, train_xy_list, configs):
    print ("fitering and building dict...")
    use_abisee = True
    all_dic1 = {}
    all_dic2 = {}
    dic_list = []
    all_dic1, dic_list = load_vocab(src_path, tgt_path)
    all_dic2 = to_dict(train_xy_list, all_dic2)
    for w, tf in all_dic2.items():
        if w not in all_dic1:
            all_dic1[w] = tf

    candiate_list = dic_list[0:configs.PG_DICT_SIZE] # 50000
    candiate_set = set(candiate_list)

    dic = {}
    w2i = {}
    i2w = {}
    w2w = {}

    for w in [configs.W_PAD, configs.W_UNK, configs.W_EOS]:
        w2i[w] = len(dic)
        i2w[w2i[w]] = w
        dic[w] = 10000
        w2w[w] = w

    for w, tf in all_dic1.items():
        if w in candiate_set:
            w2i[w] = len(dic)
            i2w[w2i[w]] = w
            dic[w] = tf
            w2w[w] = w
        else:
            w2w[w] = configs.W_UNK 

    hfw = []
    sorted_x = sorted(dic.items(), key=operator.itemgetter(1), reverse=True)
    for w in sorted_x:
        hfw.append(w[0])

    assert len(hfw) == len(dic)
    assert len(w2i) == len(dic)
    return all_dic1, dic, hfw, w2i, i2w, w2w
 

def get_xy_tuple(cont, head, configs):
    x = read_cont(cont, configs)
    y = read_head(head, configs)

    if x != None and y != None:
        return (x, y)
    else:
        return None

def read_cont(src_line, cfg):
    lines = []
    line = src_line #del_num(f_cont)
    words = line.split()
    lines += words
    lines += [cfg.W_EOS]
    return (lines, src_line) 

def read_head(tgt_line, cfg):
    lines = []
    line = tgt_line 
    words = line.split()
    lines += words
    lines += [cfg.W_EOS]
    return (lines, tgt_line) 

def load_lines(src_path, tgt_path,  configs):
    lines = []
    with open(src_path, 'r') as src_file, open(tgt_path, 'r') as tgt_file:
        for src_line, tgt_line in zip(src_file, tgt_file):

            src_line = src_line.strip().lower()
            tgt_line = tgt_line.strip().lower()
            xy_tuple = get_xy_tuple(src_line, tgt_line, configs)
            lines.append(xy_tuple)
    return lines


def prepare_dir():
   return configs

def prepare_so_dataset():
    # Prepare Dirs
    configs = DeepmindConfigs()
    TRAINING_PATH = configs.cc.TRAINING_DATA_PATH
    VALIDATE_PATH = configs.cc.VALIDATE_DATA_PATH
    TESTING_PATH = configs.cc.TESTING_DATA_PATH
    RESULT_PATH = configs.cc.RESULT_PATH
    MODEL_PATH = configs.cc.MODEL_PATH
    BEAM_SUMM_PATH = configs.cc.BEAM_SUMM_PATH
    BEAM_GT_PATH = configs.cc.BEAM_GT_PATH
    GROUND_TRUTH_PATH = configs.cc.GROUND_TRUTH_PATH
    SUMM_PATH = configs.cc.SUMM_PATH
    TMP_PATH = configs.cc.TMP_PATH

    print ("train: " + TRAINING_PATH)
    print ("test: " + TESTING_PATH)
    print ("validate: " + VALIDATE_PATH) 
    print ("result: " + RESULT_PATH)
    print ("model: " + MODEL_PATH)
    print ("tmp: " + TMP_PATH)

    if not exists(TRAINING_PATH):
        makedirs(TRAINING_PATH)
    if not exists(VALIDATE_PATH):
        makedirs(VALIDATE_PATH)
    if not exists(TESTING_PATH):
        makedirs(TESTING_PATH)
    if not exists(RESULT_PATH):
        makedirs(RESULT_PATH)
    if not exists(MODEL_PATH):
        makedirs(MODEL_PATH)
    if not exists(BEAM_SUMM_PATH):
        makedirs(BEAM_SUMM_PATH)
    if not exists(BEAM_GT_PATH):
        makedirs(BEAM_GT_PATH)
    if not exists(GROUND_TRUTH_PATH):
        makedirs(GROUND_TRUTH_PATH)
    if not exists(SUMM_PATH):
        makedirs(SUMM_PATH)
    if not exists(TMP_PATH):
        makedirs(TMP_PATH)
    
    # Prepare Dataset 
    src_path = "./so-data/src-train.txt"
    tgt_path = "./so-data/tgt-train.txt"
    print ("trainset...")
    train_xy_list = load_lines(src_path, tgt_path,  configs)
    print ("dump train...")
    pickle.dump(train_xy_list, open(TRAINING_PATH + "train.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    
    all_dic1, dic, hfw, w2i, i2w, w2w = prepare_dict(src_path, tgt_path, train_xy_list, configs)
    print ("dump dict...")
    pickle.dump([all_dic1, dic, hfw, w2i, i2w, w2w], open(TRAINING_PATH + "dic.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)

    # src_path = "./so-data/src-val.txt"
    # tgt_path = "./so-data/tgt-val.txt"
    # print ("valset...")
    # val_xy_list = load_lines(src_path, tgt_path,  configs)
    # print ("dump val...")
    # pickle.dump(val_xy_list, open(VALIDATE_PATH + "./val.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    
    # src_path = "./so-data/src-test.txt"
    # tgt_path = "./so-data/tgt-test.txt"
    # print ("testset...")
    # test_xy_list = load_lines(src_path, tgt_path,  configs)
    # print ("dump val...")
    # pickle.dump(test_xy_list, open(TESTING_PATH + "./val.pkl", "wb"), protocol = pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    prepare_so_dataset()
    print("Finished")

