# -*- coding:utf-8 -*-
import unicodedata
import nltk
import pandas as pd
from rouge import Rouge
from nltk.translate import bleu
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.bleu_score import SmoothingFunction

class Evaluation(object):
    
    def __init__(self, hypo_path, ref_path, eval_path):
        self.hypo_path = hypo_path
        self.ref_path = ref_path
        self.eval_path = eval_path 
        self.hypothesis_lst = self.get_hypothesis()
        self.reference_lst = self.get_reference()
        assert len(self.hypothesis_lst) == len(self.reference_lst)
        self.evaluation_report = self.get_evaluation_report()
        pass
    
    def get_hypothesis(self): 
        hypothesis_lst = []
        with open(self.hypo_path, "r") as hypo_file:
            for line in hypo_file:
                hypothesis_lst.append(line.strip()) 
        return hypothesis_lst
        pass

    def get_reference(self):
        reference_lst = []
        with open(self.ref_path, "r") as ref_file:
            for line in ref_file:
                reference_lst.append(line.strip())
        return reference_lst
        pass
    
    def get_evaluation_report(self):
        """
        """
        evaluation_report = [] 
        for hypothesis, reference in zip(self.hypothesis_lst, self.reference_lst):
            try:
                bleu_score = self.get_bleu_score(hypothesis, reference)    
                rouge_score = self.get_rouge_score(hypothesis, reference)
                evaluation_report.append( (hypothesis, reference, bleu_score, rouge_score) )
            except:
                continue
        return evaluation_report
        pass
    
    def save_report(self):
        with open(self.eval_path, "w") as report:
            for e in self.evaluation_report:
                # print(e)
                hypothesis, reference, bleu_score, rouge_score = e[0], e[1], e[2], e[3]
                bleu1_score, bleu2_score, bleu3_score, bleu4_score = bleu_score[0], bleu_score[1], bleu_score[2], bleu_score[3]
                rouge1_p, rouge1_r, rouge1_f = rouge_score[0]["p"], rouge_score[0]["r"], rouge_score[0]["f"]
                rouge2_p, rouge2_r, rouge2_f = rouge_score[1]["p"], rouge_score[1]["r"], rouge_score[1]["f"]
                rougel_p, rougel_r, rougel_f = rouge_score[2]["p"], rouge_score[2]["r"], rouge_score[2]["f"]
                # print(bleu1_score, bleu2_score, bleu3_score, bleu4_score)
                # print(rouge1_f, rouge2_f, rougel_f)
                # report.write(str(hypothesis) + '\t' + str(reference) + '\t' + str(bleu4_score))
                report.write(str(bleu1_score) + "," + str(bleu2_score) + "," + str(bleu3_score) + "," + str(bleu4_score) + ",")
                report.write(str(rouge1_f) + "," + str(rouge2_f) + "," + str(rougel_f) + "\n")
                # break
        pass
    
    def write_review(self):
        with open('./evaluation_review', "w") as report:
            for e in self.evaluation_report:
                hypothesis, reference, bleu_score, rouge_score = e[0], e[1], e[2], e[3]
                bleu1_score, bleu2_score, bleu3_score, bleu4_score = bleu_score[0], bleu_score[1], bleu_score[2], bleu_score[3]
                report.write(str(hypothesis) + '\t' + str(reference) + '\t' + str(bleu4_score) + '\n')
        pass



    def get_bleu_score(self, hypothesis, reference):
        """
            hypothesis: string
            reference: string
        """ 
        cc = SmoothingFunction()
        hypothesis = hypothesis.strip().split(' ')
        reference = reference.strip().split(' ')
        bleu1_score = sentence_bleu([reference], hypothesis, smoothing_function=cc.method4, weights=(1., 0) )
        bleu2_score = sentence_bleu([reference], hypothesis, smoothing_function=cc.method4, weights=(1./2, 1./2))
        bleu3_score = sentence_bleu([reference], hypothesis, smoothing_function=cc.method4, weights=(1./3, 1./3, 1./3))
        bleu4_score = sentence_bleu([reference], hypothesis, smoothing_function=cc.method4)

        return bleu1_score, bleu2_score, bleu3_score, bleu4_score
        pass

    def get_rouge_score(self, hypothesis, reference):
        """
            hypothesis: string
            reference: string
        """
        rouge = Rouge()
        rouge_scores = rouge.get_scores(hypothesis, reference)
        rouge_1 = rouge_scores[0]["rouge-1"]
        rouge_2 = rouge_scores[0]["rouge-2"]
        rouge_l = rouge_scores[0]["rouge-l"]
        return rouge_1, rouge_2, rouge_l
        pass


def evaluation_stats(eval_fpath):
    colnames = ['bleu1', 'bleu2', 'bleu3', 'bleu4', 'rouge-1', 'rouge-2', 'rouge-l']
    eva_df = pd.read_csv(eval_fpath, header=None, names=colnames)
    print(eva_df[:10])
    bleu1_mean = eva_df['bleu1'].mean()
    bleu1_std = eva_df['bleu1'].std()
    bleu1_var = eva_df['bleu1'].var()

    bleu2_mean = eva_df["bleu2"].mean()
    bleu2_std = eva_df["bleu2"].std()
    bleu2_var = eva_df["bleu2"].var()

    bleu3_mean = eva_df["bleu3"].mean()
    bleu3_std = eva_df["bleu3"].std()
    bleu3_var = eva_df["bleu3"].var()

    bleu4_mean = eva_df["bleu4"].mean()
    bleu4_std = eva_df["bleu4"].std()
    bleu4_var = eva_df["bleu4"].var()

    rouge_1_mean = eva_df["rouge-1"].mean()
    rouge_1_std = eva_df["rouge-1"].std()
    rouge_1_var = eva_df["rouge-1"].var()

    rouge_2_mean = eva_df["rouge-2"].mean()
    rouge_2_std = eva_df["rouge-2"].std()
    rouge_2_var = eva_df["rouge-2"].var()

    rouge_l_mean = eva_df["rouge-l"].mean()
    rouge_l_std = eva_df["rouge-l"].std()
    rouge_l_var = eva_df["rouge-l"].var()


    print("Bleu1")
    print(bleu1_mean, bleu1_std, bleu1_var)
    print("Bleu2")
    print(bleu2_mean, bleu2_std, bleu2_var)
    print("Bleu3")
    print(bleu3_mean, bleu3_std, bleu3_var)
    print("Bleu4")
    print(bleu4_mean, bleu4_std, bleu4_var)
    print("rouge-1")
    print(rouge_1_mean, rouge_1_std, rouge_1_var)
    print("rouge-2")
    print(rouge_2_mean, rouge_2_std, rouge_2_var)
    print("rouge-l")
    print(rouge_l_mean, rouge_l_std, rouge_l_var)
    

def main():
    
    hypo_path = "./hypothesis.txt"
    ref_path =  './reference.txt'
    eval_path = "./evaluation_report"
    eva = Evaluation(hypo_path, ref_path, eval_path)
    eva.save_report()
    evaluation_stats(eval_path)
    eva.write_review()
    pass

if __name__ == '__main__':
    main()

