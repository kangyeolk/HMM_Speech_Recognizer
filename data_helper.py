import os
words = {}
unigram = {}
bigram = {}

with open('./usage/dictionary.txt', 'r') as dic:
    for line in dic:
        line = line.split()
        if line[0] == "<s>":
            continue
        else:
            words[line[0]] = line
            words[line[0]][0] = 'sil'

with open('./usage/bigram.txt', 'r') as bigr:
    for line in bigr:
        line = line.split()
        bigram[(line[0], line[1])] = float(line[2])

with open('./usage/unigram.txt', 'r') as unigr:
    for line in unigr:
        line = line.split()
        unigram[line[0]] = float(line[1])

def file_list():
    test_files = []
    for root, _, files in os.walk("tst", topdown=False):
        for file in files:
            file_path = os.path.join(root, file).replace("\\", "/")
            test_files.append(file_path)
    return test_files
