for word_1 in hmm_dict:
    (a, b) = hmm_dict[word_1]
    hmm.a[0, a] = unigram[word_1]
    for word_2 in hmm_dict:
        (c, d) = hmm_dict[word_2]
        escape = words_hmm[word_1].nstates - 1
        hmm.a[b - 1, c] = 0
        hmm.a[b, c] = 0
        if word_1 != "zero2" and word_2 != "zero2":
            hmm.a[b - 1, c + 3] = bigram[(word_1, word_2)] * words_hmm[word_1].a[escape - 2, escape] * PARAM
            hmm.a[b, c + 3] = bigram[(word_1, word_2)] * words_hmm[word_1].a[escape - 1, escape] * PARAM
        else:
            if word_1 == "zero2" and word_1 != "zero2":
                hmm.a[b - 1, c + 3] = bigram[("zero", word_2)] * words_hmm[word_1].a[escape - 2, escape] * PARAM
                hmm.a[b, c + 3] = bigram[("zero", word_2)] * words_hmm[word_1].a[escape - 1, escape] * PARAM
            elif word_2 == "zero2" and word_1 != "zero2":
                hmm.a[b - 1, c + 3] = bigram[(word_1, "zero")] * words_hmm[word_1].a[escape - 2, escape] * PARAM
                hmm.a[b, c + 3] = bigram[(word_1, "zero")] * words_hmm[word_1].a[escape - 1, escape] * PARAM
            elif word_1 == "zero2" and word_2 == "zero2":
                hmm.a[b - 1, c + 3] = bigram[("zero", "zero")] * words_hmm[word_1].a[escape - 2, escape] * PARAM
                hmm.a[b, c + 3] = bigram[("zero", "zero")] * words_hmm[word_1].a[escape - 1, escape] * PARAM


for word in hmm_dict:
    (a, b) = hmm_dict[word_1]
    hmm.a[b - 1, hmm.nstates - 1] = words_hmm[word_1].a[words_hmm[word_1].nstates - 2, words_hmm[word_1].nstates - 1]
    hmm.a[b, hmm.nstates - 1] = words_hmm[word_1].a[words_hmm[word_1].nstates - 1, words_hmm[word_1].nstates - 1]


def viterbi33(hmm, x):
    V = {}
    traj = {}
    iter = list(range(hmm.nstates))
    for s in range(hmm.nstates):
        V[(1, s)] = logproduct(log(hmm.tran[0, s]), hmm.emss(s, x[1]))
        traj[(1, s)] = 0
    for t in range(2, len(x) + 1):
        for j in range(hmm.nstates):
            V[(t, j)] = MINUS_INF
            traj[(t, j)] = 0
            tmp = list(map(lambda i: logproduct(V[(t - 1, i)], log(hmm.tran[i][j])), iter))
            V[(t, j)] = np.max(tmp)
            traj[t, j] = tmp.index(V[(t, j)])
            if j == hmm.nstates - 1:
                V[t, j] = MINUS_INF
            else:
                V[t, j] = logproduct(V[t, j], hmm.emss(j, x[t]))
    return traj
