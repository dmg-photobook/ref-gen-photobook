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

## these variables are the same as the ones in the last two tables of appendix G
compression = ['text_len_content', 'content_prop',
                          'cont_noun_prop', 'cont_adj_prop', 'cont_verb_prop']
givenness = ['prop_the', 'prop_a',
                        'prop_seen_markers',
                        'prop_givenness_markers']
reuse = ['prop_content_from_prev', 'prop_reused_content_bigrams',
                 'prop_nouns_in_reused_content_tokens', 'prop_adjs_in_reused_content_tokens', 'prop_verbs_in_reused_content_tokens',
                 'prop_nn_content_bigrams_reused']

### functions
def get_prop_tag_content(tag, taglist, wdlist):
    tags = 0
    denom = 0
    for i in range(len(taglist)):
        if wdlist[i] not in stop: ## only content
            denom += 1
            if taglist[i] == tag:
                tags += 1
    return np.NaN if denom < 1 else tags/denom



if len(sys.argv) == 1:
    print('specify files to process as a list of command line arguments')
    sys.exit()

## Create dataframe from output files
print('loading files...')
files = sys.argv[1:]
#files = ['all_copy_TEST.txt', 'all_base_TEST.txt', 'all_histatt_TEST.txt']
print(files)
res = elib.get_dataframe_from_files(files)
#res.to_csv('model_output_summary.csv')

## Calculate stats
print('tagging...')
res['tags'] = res.apply(lambda row: [item[1] for item in pos_tag(row.text, tagset='universal')], axis = 1)
res['prev_tokens'] = res.apply(lambda row: [item[1] for item in pos_tag(row.prev_utt_full, tagset='universal')], axis = 1)

res['content_text'] = res.apply(lambda row: [i for i in row.text if i not in stop], axis = 1)
res['content_tokens'] = res.apply(lambda row: elib.content_tag_list(row.text, row.tags), axis = 1)
res['content_prev'] = res.apply(lambda row: [i for i in row.prev_utt_full if i not in stop], axis = 1)
res['content_tokens_prev'] = res.apply(lambda row: elib.content_token_list(row.prev_utt_full, row.prev_tokens), axis = 1)

res['reused_wds'] = res.apply(lambda row:  elib.get_shared_words(row.text, row.prev_utt_full), axis = 1)
res['reused_tags'] = res.apply(lambda row:  elib.get_shared_wd_tags(row.text, row.prev_utt_full, row.tags), axis = 1)


print('compression stats...')
res['text_len_content'] = res.apply(lambda row: len(row.content_text), axis = 1)
res['content_prop'] = res.apply(lambda row: elib.content_prop_text(row.text), axis = 1)

res['temp'] = res.apply(lambda row: elib.content_token_prop(row.text), axis = 1)
res['cont_adj_prop'] = res.apply(lambda row: row.temp[0], axis = 1)
res['cont_verb_prop'] = res.apply(lambda row: row.temp[1], axis = 1)
res['cont_noun_prop'] = res.apply(lambda row: row.temp[2], axis = 1)


print('givenness stats...')
res['prop_givenness_markers'] = res.apply(lambda row: elib.prop_givenness(row.text), axis = 1)
res['prop_the'] = res.apply(lambda row: elib.prop_the(row.text), axis = 1)
res['prop_a'] = res.apply(lambda row: elib.prop_a(row.text), axis = 1)
res['prop_seen_markers'] = res.apply(lambda row: elib.prop_seen_markers(row.text), axis = 1)


print('reuse stats...')

res['prop_content_from_prev'] = res.apply(lambda row:   elib.prop_content_reused(row.text, row.prev_utt_full), axis = 1)

res['content_bigrams'] = res.apply(lambda row: elib.get_content_bigram_strings(row.text), axis = 1)
res['content_bigrams_prev'] = res.apply(lambda row: elib.get_content_bigram_strings(row.prev_utt_full), axis = 1)
res['count_content_bigrams_reused'] = res.apply(lambda row: elib.count_words_reused(row.content_bigrams_prev, row.content_bigrams), axis = 1)
res['prop_reused_content_bigrams'] = res.apply(lambda row: row.count_content_bigrams_reused/float(max(1, len(row.content_bigrams))), axis = 1)


res['prop_nouns_in_reused_content_tokens'] = res.apply(lambda row:  elib.get_prop_tag_content('NOUN', row.reused_tags, row.reused_wds), axis = 1)
res['prop_adjs_in_reused_content_tokens'] = res.apply(lambda row:  elib.get_prop_tag_content('ADJ', row.reused_tags, row.reused_wds), axis = 1)
res['prop_verbs_in_reused_content_tokens'] = res.apply(lambda row:  elib.get_prop_tag_content('VERB', row.reused_tags, row.reused_wds), axis = 1)

res['prop_nn_content_bigrams_reused'] = res.apply(lambda row:   elib.prop_nn_bigrams_reused(row.text, row.tags, row.prev_utt_full), axis = 1)

#res.to_csv('model_output_summary_stats.csv')
res['mention'] = res.first_mention

## firstly calculate the compression and givenness stats: first vs the later are compared
rows = []
cols = ['comparison', 'model', 'version', 'meanFirst', 'meanLater', 'cohensD', 'sig', 'pval', 'tstat']
aspects = compression + givenness
for aspect in aspects:
    for version in res.version.unique():
        row = elib.compare(res, 'first', 'later', version, aspect, aspect)
        rows.append(row)
first_later = pd.DataFrame(rows, columns=cols)
first_later.to_csv('firstVlater_stats_out.csv')

## now reuse: only the later utterances are inspected: thus we test differences between human and model
cols = ['comparison', 'model', 'version', 'meanHuman', 'meanGenerated', 'cohensD', 'sig', 'pval', 'tstat', 'mention']
rows = []
versions = res.version.unique()
versions = [i for i in versions if i not in ['origional']]
human = 'origional'
for col_name in reuse:
    mention = 'later'
    for version in versions:
        row = elib.get_rows(res, version, mention, human, col_name)
        rows.append(row)
reuse = pd.DataFrame(rows, columns=cols)
reuse.to_csv('reuse_stats_out.csv')
