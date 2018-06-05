import numpy as np
from copy import deepcopy
import operator
import os

from model import *

# ============================================================= #
#                      Predict with viterbi                     #
# ============================================================= #

def state2word(seq_state):
    seq_word = []
    current_word = ""
    for i in range(len(seq_state) - 1):
        for word in hmm_dict:
            (a, b) = hmm_dict[word]
            if seq_state[i] <= b and seq_state[i] >= a + 3:
                if current_word != word and seq_state[i] > seq_state[i-1]:
                    current_word = word
                    if current_word == "zero2":
                        seq_word.append("zero")
                    else:
                        seq_word.append(current_word)
                elif seq_state[i] < seq_state[i-1]:
                    current_word = ""
    return seq_word

test_files_path = file_list()

if __name__ == "__main__":
    with open("recognized_test.txt", "w") as recog:
        recog.write("#!MLF!#" + '\n')
        for path in test_files_path:
            recog.write("\"" + path.replace("txt", 'rec') + "\"\n")
            test_file = open(path)
            x = {}
            for i in range(1, int(test_file.readline().split()[0]) + 1):
                x[i] = list(map(float, test_file.readline().split()))
            print(path, 'Viterbi Doing')
            try:
                state_pred = viterbi(hmm, x)
                result = state2word(state_pred)
                for i in range(len(result)):
                    recog.write(result[i] + "\n")
            except:
                recog.write(".\n")
                print('Error', path)
                continue
            recog.write(".\n")
            print(path, 'DONE')
