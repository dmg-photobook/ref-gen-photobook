import json
import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from processor import Log


def load_logs(dir_path):
    print('>> Loading logs from "{}"'.format(dir_path))

    file_count = 0
    for _, _, files in os.walk(dir_path):
        for file in files:
            file_count += int(file.endswith('.json'))
    print('{} files found.'.format(file_count))

    logs = {}
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith('.json'):
                with open(os.path.join(root, file), 'r') as logfile:
                    log = Log(json.load(logfile))
                    if log.complete:
                        logs[log.game_id] = log

    print('DONE. Loaded {} completed game logs.'.format(len(logs)))
    return logs


def stopwords_filter(text, stopwords):
    filtered = []
    for tok in text.split(' '):
        if tok.lower() not in stopwords:
            filtered.append(tok)

    return ' '.join(filtered)


def text_to_bow(text, model, tokenizer, stopwords=None):
    if text == '':
        return None

    input_ids = torch.tensor([tokenizer.encode(text, add_special_tokens=True)])
    with torch.no_grad():
        last_hidden_states = model(input_ids)[0]
        text_repr = last_hidden_states[0, 1:-1, :].data.numpy()
        bow = hidden_to_bow(text_repr)

    tokens = tokenizer.tokenize(text)
    assert len(tokens) == len(bow)

    if stopwords:
        bow_tmp = []
        tokens_tmp = []

        for tok, vec in zip(tokens, bow):
            if tok not in stopwords:
                bow_tmp.append(vec)
                tokens_tmp.append(tok)

        bow = bow_tmp
        tokens = tokens_tmp

    return tokens, input_ids, bow


def hidden_to_bow(hidden_states, normalize=True):
    bow = []
    for t in np.arange(hidden_states.shape[0]):
        h_t = np.copy(hidden_states[t, :])
        if normalize:
            h_t /= np.linalg.norm(h_t)
        bow.append(h_t)
    return bow


def get_captions(captions_path, chains):
    id2str = {}
    img_strings = [path.split('/')[-1] for path in chains]

    with open(captions_path, 'r') as f:
        annotations = json.loads(f.read())

    for image_data in annotations['images']:
        coco_id = image_data['coco_url'].split('/')[-1]
        if coco_id in img_strings:
            id2str[int(image_data['id'])] = coco_id

    n_captions = 0
    captions = defaultdict(list)
    for ann in annotations['annotations']:
        try:
            captions[id2str[int(ann['image_id'])]].append(ann['caption'].strip())
            n_captions += 1
        except KeyError:
            continue

    print('{} captions collected for {} images.'.format(n_captions, len(captions)))
    return captions


def preprocess_captions(image_paths, captions, model, tokenizer):
    caption_representations = defaultdict(list)

    for img_path in tqdm(image_paths):
        img_id_str = img_path.split('/')[-1]

        # -----------------------------------------------------------------------------------
        # Get BERT contextualised representations for all tokens of the >=5 image captions
        # -----------------------------------------------------------------------------------
        for caption in captions[img_id_str]:
            input_ids = torch.tensor([tokenizer.encode(caption, add_special_tokens=True)])

            # 2. get last layer's hidden state for each token in the current caption
            with torch.no_grad():
                last_hidden_states = model(input_ids)[0]  # Models outputs are now tuples
                reprs = last_hidden_states[0, 1:-1, :].data.numpy()  # [0, 0, :] for CLS

            # 3. build a bag-of-contextualised-words representation of the current caption
            bow_reprs = hidden_to_bow(reprs, normalize=True)

            # 5. store bag-of-words representations
            caption_tokens = tokenizer.tokenize(caption)
            assert len(caption_tokens) == len(bow_reprs)
            caption_representations[img_id_str].append(tuple(zip(caption_tokens, bow_reprs)))

    return caption_representations


def group_by_game(chains):
    chains_by_game = {img: {} for img in chains}
    for img, utterances in chains.items():
        assert img in chains_by_game
        for utt in utterances:
            try:
                chains_by_game[img][int(utt['Game_ID'])].append(utt)
            except KeyError:
                chains_by_game[img][int(utt['Game_ID'])] = [utt]

    return chains_by_game
