import nltk
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
from nltk.tag import pos_tag
from nltk import word_tokenize
from nltk.corpus import stopwords
import string
from collections import defaultdict
from collections import Counter
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
import ast
from statannot import add_stat_annotation
import copy
from statistics import mean, stdev
from math import sqrt
#import eval_lib as elib
import linguistic_stats_functions as elib
import sys

stop = stopwords.words('english') + list(string.punctuation)

def get_dataframe_from_files(files):
    res = pd.DataFrame(columns=['context', 'id', 'target_utt','prev_utt', 'prev_utt_full', 'first_mention', 'context', 'version', 'text', 'model', 'v', 'chain_index_text'])
    count = 0
    for file in files:
        ## count models
        jump_len = 0
        with open(file) as file_in:
            for line in file_in:
                jump_len+=1
                if line == "\n" or jump_len > 15:
                    break

        print(jump_len)
        with open(file) as file_in:
            lines = []
            for line in file_in:
                #print(line)
                lines.append(line)
                #print(line == '\n')

        model_name = file.split('.txt')[0]
        print(model_name)

        for i in range(0,len(lines),jump_len):
            count +=1
            print(count % 6252, count, "{:.2%}".format(count/float(6252 * 3)), "{:.2%}".format((count % 6252)/float(6252)))
            context = ast.literal_eval(lines[i].split('I:')[1].strip())
            turn = lines[i+1].split('T:')[1].strip()
            target_utt = lines[i+2].split('U:')[1].replace('<eos>','').replace('<sos>','').strip().split(' ')
            previous_utt = lines[i+3].split('P:')[1].replace('<eos>','').replace('<sos>','').strip().split(' ')
            previous_utt_full = lines[i+4].split('A:')[1].replace('<eos>','').replace('<sos>','').strip().split(' ')
            chain_all_utts = ast.literal_eval(lines[i+5].split('R:')[1].strip())
            target_utt = get_target_no_unk(target_utt, chain_all_utts)
            pos_in_chain = get_target_position(target_utt, chain_all_utts )
            first = 'later'
            if previous_utt[0] == '<nohs>':
                first = 'first'
                previous_utt[0].replace('<nohs>', '')

            res.loc[len(res)] = [context, turn, target_utt, previous_utt, previous_utt_full,
                                 first, context, model_name + 'origional', target_utt, model_name, 'origional', pos_in_chain]
            for p in range(6, jump_len-1):
                if ':' in lines[i+p]:
                    name, pred = lines[i+p].split(': ', 1)
                    pred = pred.replace('<eos>','').replace('<sos>','').strip().split(' ')
                    res.loc[len(res)] = [context, turn, target_utt, previous_utt, previous_utt_full, first, context, model_name + name, pred, model_name, name, pos_in_chain]

    versions = res.version.unique()
    o_versions = [i for i in versions if 'origional' in i]
    o_replace = o_versions[0]
    res['temp'] = res.apply(lambda row: rename_one_orig(row, o_versions, o_replace), axis = 1)
    print(res.temp.unique())
    res.version = res.temp
    res = res[(res.version != 'drop')]
    return res

def get_target_no_unk(target, chain):
    if not '<unk>' in target:
        return target
    for match in chain:
        unk_indexes_target = [i for i, j in enumerate(target) if j == '<unk>']
        m = match.split(' ')
        for i in unk_indexes_target:
            if i < len(m):
                m[i] = '<unk>'
        t_s = ' '.join(target)
        t_m = ' '.join(m)
        if t_s == t_m:
            return match.split(' ')
    print('ERROR: cannot find target in chain')
    return []

def rename_one_orig(row, o_versions, o_replace):
    if row.version == o_replace:
        return 'origional'
    elif row.version in o_versions:
        return 'drop'
    else:
        return row.version

def get_target_position(target, chain):
    t = ' '.join(target)
    for i in range(0,len(chain)):
        if t == chain[i]:
            return i
    print('ERROR: problem in chain')
    return -1

def content_tag_list(text, tags):
    tokens_out = []
    for i in range(0, len(text)):
        if text[i] not in stop:
            tokens_out.append(tags[i])
    return tokens_out

def content_token_list(text, tokens):
    tokens_out = []
    for i in range(0, len(text)):
        if text[i] not in stop:
            tokens_out.append(tokens[i])
    return tokens_out

def content_prop_text(text):
    if len(text) < 1:
        return 0
    return len([i for i in text if i not in stop])/float(len(text))

def content_token_prop(tokens):
    print(tokens)
    pos_tags = [item[1] for item in pos_tag(tokens, tagset='universal')]
    a = 0
    v = 0
    n = 0
    denom = 0.0
    for i in range(len(pos_tags)):
        if tokens[i] not in stop:
            denom += 1
            p = pos_tags[i]
            if p == 'ADJ':
                a +=1
            elif p == 'VERB':
                v += 1
            elif p == 'NOUN':
                n += 1
    a_res = 0 if denom < 1 else a/denom
    v_res = 0 if denom < 1 else v/denom
    n_res = 0 if denom < 1 else n/denom
    return a_res, v_res, n_res


## resuse
def prop_nn_bigrams_reused(text, tags, text2):
    bigramreuse = 0
    nn_reuse = 0
    bigrams_prev = [text2[i] + ' ' + text2[i+1] for i in range(len(text2) - 1)]
    for i in range(len(text) - 1):
        bigram = text[i] + ' ' + text[i+1]
        if bigram in bigrams_prev:
            bigramreuse +=1
            if tags[i] == 'NOUN' and tags[i+1] == 'NOUN':
                if text[i] not in stop and text[i+1] not in stop:
                    nn_reuse +=1

    return nn_reuse/ (bigramreuse) if bigramreuse > 0 else np.NaN


def get_shared_words(current, prev):
    res = []
    for wd in current:
        if wd in prev:
            res.append(wd)
    return res

def get_shared_wd_tags(current, prev, current_tags):
    res = []
    for i in range(len(current)):
        if current[i] in prev:
            res.append(current_tags[i])
    return res

def get_prop_tag_content(tag, taglist, wdlist):
    tags = 0
    denom = 0
    for i in range(len(taglist)):
        if wdlist[i] not in stop: ## only content
            denom += 1
            if taglist[i] == tag:
                tags += 1
    return np.NaN if denom < 1 else tags/denom

def prop_content_reused(text, prev):
    current = 0.0
    reused = 0
    for i in range(len(text)):
        if text[i] not in stop:
            current +=1
            if text[i] in prev:
                reused += 1
    return reused/current if current > 0 else 0

def get_content_bigram_strings(text):
    bigrams = []
    if len(text)<1:
        return []
    for i in range(0, len(text) - 1):
        if text[i] not in stop and text[i + 1] not in stop:
            bigram = text[i] + ' ' + text[i+1]
            bigrams.append(bigram)
    return bigrams

def count_words_reused(prev, target):
    overlap = 0
    for tok in target:
        if tok in prev:
            overlap += 1
    return overlap

# givenness
def prop_givenness(text):
        count = 0
        markers = ['again', 'before', 'one', 'same', 'also', 'the']
        for wd in text:
            if wd in markers:
                count += 1
        return count / float(len(text))

def prop_seen_markers(text):
    seen = ['again', 'before', 'one', 'same', 'also']
    count = 0
    print(text)
    for w in text:
        print(w)
        if w in seen:
            count += 1
    return count/float(max(1, len(text)))

def prop_the(string):
    count = string.count('the')
    return count/float(max(1, len(string)))

def prop_a(string):
    indef = ['some', 'a', 'an']
    count = 0
    for w in string:
        if w in indef:
            count += 1
    return count/float(max(1, len(string)))

def cohens_d(c0, c1):
    try:
        cohens_d = (mean(c0) - mean(c1)) / (sqrt((stdev(c0) ** 2 + stdev(c1) ** 2) / 2))
    except:
        return 0
    return cohens_d

def sigstar(pval):
    if pval < 0.001:
        return '***'
    elif pval < 0.005:
        return '**'
    elif pval < 0.05:
        return '*'
    else:
        return ''

def compare(res, mention1, mention2, version, col1, col2):
    first = res[((res.mention == mention1) & (res.version == version))& (res[col1].notnull())][col1].values
    later = res[((res.mention == mention2) & (res.version == version))& (res[col2].notnull())][col2].values
    print(min(first), min(later), max(first), max(later), mean(first), mean(later))
    model = res[((res.mention == mention1) & (res.version == version))].model.unique()[0]
    if version == 'origional':
        model = 'human'
        version_label = 'human'
    else:
        print(version, model)
        version_label= version.split('_')[-1]
        model = model.split('_')[-1]
    cod = elib.cohens_d(first, later)
    tstat, pval = stats.ttest_ind(first, later)
    sig = elib.sigstar(pval)
    if col1 == col2:
        aspect = col1
    else:
        aspect = col1 + '-' + col2
    row = [aspect, model, version_label, first.mean(), later.mean(), cod, sig, pval, tstat]
    print(row)
    return row

def get_stats_CPT(df, version1, version2, column_name, first_or_later):
    human = df[(df.mention == first_or_later) & (df.version== version1) & (df[column_name].notnull())][column_name]
    model = df[(df.mention == first_or_later) & (df.version== version2) & (df[column_name].notnull())][column_name]
    label = df[(df.mention == first_or_later) & (df.version== version2)].model.unique()[0]
    cod = cohens_d(human, model)
    tstat, pval = stats.ttest_ind(human, model)
    return mean(human), mean(model), cod, tstat, pval, label

def sortlabel(model, version):
    if version == 'origional':
        model_label = 'human'
        version_label = 'human'
    else:
        version_label= version.split('_')[-1]
        model_label = model.split('_')[-1]
    return model_label, version_label

def get_rows(res, version, mention, human, col_name):
    human_mean, model_mean, cod, tstat, pval, model_label = get_stats_CPT(res, human, version, col_name, mention)
    sig = elib.sigstar(pval)
    col_lable = col_name
    model_label, version_label = sortlabel(model_label, version)
    row = [col_lable, model_label, version_label, human_mean, model_mean, cod, sig, pval, tstat, mention]
    return row
