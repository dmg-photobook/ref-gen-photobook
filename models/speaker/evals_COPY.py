import torch
import json

import torch.nn as nn
import torch.nn.functional as F

import os

from bert_score import score

#beam search
#topk
#topp

# built via modifying https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/eval.py

def eval_beam_histatt_copy(split_data_loader, model, args, best_score, print_gen, device,
                      beam_size, max_len, vocab, mask_attn, nlgeval_obj, isValidation, timestamp, vocab_full, mask_nohs,
                           isTest):
    """
        Evaluation

        :param beam_size: beam size at which to generate captions for evaluation
        :return: Official MSCOCO evaluator scores - bleu4, cider, rouge, meteor
        """

    # Lists to store references (true captions), and hypothesis (prediction) for each image
    # If for n images, we have n hypotheses, and references a, b, c... for each image, we need -
    # references = [[ref1a, ref1b, ref1c], [ref2a, ref2b], ...], hypotheses = [hyp1, hyp2, ...]

    references = []
    hypotheses = []

    count = 0

    empty_count = 0

    breaking = args.breaking

    sos_token = torch.tensor(vocab['<sos>']).to(device)
    eos_token = torch.tensor(vocab['<eos>']).to(device)

    if isValidation:
        split = 'val'
    elif isTest:
        split = 'test'
    else:
        split = 'train'

    coverage_flag = True if 'cov' in args.model_type else False

    file_name = args.model_type + '_' + args.metric + '_' + split + '_' + timestamp  # overwrites previous versions!

    for i, data in enumerate(split_data_loader):
        # print(i)

        completed_sentences = []
        completed_scores = []

        beam_k = beam_size

        if breaking and count == 5:
            break

        count += 1

        # dataset details
        # only the parts I will use for this type of model


        utterances_text_ids_actual = data['actual_utterance']  # to be decoded, we don't use this here in beam search!
        target_utterance = utterances_text_ids_actual[:, 1:]
        #orig_text_reference = data['orig_utterance']
        reference_chain = data['reference_chain'][0]  # batch size 1  # full set of references for a single instance
        # obtained from the whole chain

        prev_utterance = data['prev_utterance']
        prev_utt_lengths = data['prev_length']

        prev_utterance_ids_actual = data['actual_prev_utterance']
        prev_history_actual = data['actual_prev_histories']

        # prev_reference = [vocab_full[int(w)] for w in prev_utterance_ids_actual[0] if w not in
        #              [vocab_full.word2index['<sos>'], vocab_full.word2index['<eos>'], vocab_full.word2index['<pad>']]]
        #
        # prev_reference_string = [' '.join(prev_reference)]
        # print('prev', prev_reference_string)

        unk_words = data['unk_words']
        mapped_utterance_words = data['mapped_utterance_words']
        mapped_prev_words = data['mapped_prev_words']

        # encoder input: prev_utterance_ids (previous utterance with UNK tokens), or NOHS

        # decoder input: utterances_text_ids (sentence to be generated with UNK tokens, teacher-forcing)

        # target generation: mapped_utterance_words for LOSS (decoder target with words from decoder input kept the same and the UNKs are
        # replaced with temporary IDs added to the vocab (if that word exists in prev_utterance_ids_actual)
        # if not still UNK)

        # reference sentence: utterances_text_ids_actual (to calculate metric scores, we will give the original sentence)

        visual_context = data['concat_context']
        target_img_feats = data['target_img_feats']

        max_length_tensor = prev_utterance.shape[1]

        masks = mask_attn(prev_utt_lengths, max_length_tensor, device)

        masks_nohs = mask_nohs(prev_utt_lengths, max_length_tensor, device)

        # unks in hist
        len_unk_words_batch = unk_words.shape[1]  # batch, max_unk

        visual_context_hid = model.relu(model.lin_viscontext(visual_context))
        target_img_hid = model.relu(model.linear_separate(target_img_feats))

        concat_visual_input = model.relu(model.linear_hid(torch.cat((visual_context_hid, target_img_hid), dim=1)))

        embeds_words = model.embedding(prev_utterance)  # b, l, d

        # pack sequence

        sorted_prev_utt_lens, sorted_idx = torch.sort(prev_utt_lengths, descending=True)
        embeds_words = embeds_words[sorted_idx]

        concat_visual_input = concat_visual_input[sorted_idx]

        # RuntimeError: Cannot pack empty tensors.
        packed_input = nn.utils.rnn.pack_padded_sequence(embeds_words, sorted_prev_utt_lens, batch_first=True)

        # start lstm with average visual context:
        # conditioned on the visual context

        # he, ce = self.init_hidden(batch_size, device)
        concat_visual_input = torch.stack((concat_visual_input, concat_visual_input), dim=0)

        packed_outputs, hidden = model.lstm_encoder(packed_input, hx=(concat_visual_input, concat_visual_input))

        # re-pad sequence
        outputs, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, batch_first=True)
        # already concat forward backward

        # un-sort
        _, reversed_idx = torch.sort(sorted_idx)
        outputs = outputs[reversed_idx]

        # ONLY THE HIDDEN AND OUTPUT ARE REVERSED
        # next_utterance is aligned (pre_utterance info is not)
        batch_out_hidden = hidden[0][:, reversed_idx]  # .squeeze(0)

        # start decoder with these

        # teacher forcing?

        if len_unk_words_batch == 0:
            len_unk_words_batch += 1
            nohs_flag = True
            # to account for NOHS in scatter (final gen distribution.scatter(...attention dist))
            # extended vocab
            # and then discard the last index
            # otherwise scatter would complain about the index
            # we mask the attention scores for NOHS for the copy mechanism
            # although we still apply attention to it
        else:
            nohs_flag = False

        decoder_hid = model.linear_dec(torch.cat((batch_out_hidden[0], batch_out_hidden[1]), dim=1))

        history_att = model.lin2att_hist(outputs)

        decoder_hid = decoder_hid.expand(beam_k, -1)

        # multiple copies of the decoder
        h1, c1 = decoder_hid, decoder_hid

        # ***** beam search *****

        gen_len = 0

        decoder_input = sos_token.expand(beam_k, 1)  # beam_k sos copies

        coverage_vector = torch.zeros(beam_k, prev_utterance.shape[1], 1).to(device) # coverage is all 0 at the beginning

        gen_sentences_k = decoder_input  # all start off with sos now

        top_scores = torch.zeros(beam_k, 1).to(device)  # top-k generation scores

        while True:

            # EOS?

            if gen_len > max_len:
                break  # very long sentence generated

            # generate

            # sos segment eos
            # base model with visual input

            # to look up embeddings, replace temp unk with UNK
            # 2791 is vocablen, highest actual ID can be 2789 (2790 is nohs)
            temp_unk_indices = [w for w in range(len(decoder_input)) if decoder_input[w] > (len(vocab)-2)]
            decoder_input[temp_unk_indices] = torch.tensor(1).to(device)

            decoder_embeds = model.embedding(decoder_input).squeeze(1)

            decoder_input_gen = model.tanh(model.lin2pgen_decinp(decoder_embeds))

            h1, c1 = model.lstm_decoder(decoder_embeds, hx=(h1, c1))

            h1_att = model.lin2att_hid(h1)

            if coverage_flag:
                attention_out = model.attention(model.tanh(history_att + h1_att.unsqueeze(1)
                                                         + model.linear_coverage(coverage_vector)))

            else:
                attention_out = model.attention(model.tanh(history_att + h1_att.unsqueeze(1)))

            attention_out = attention_out.masked_fill_(masks, float('-inf'))

            att_weights = model.softmax(attention_out)

            att_context_vector = (history_att * att_weights).sum(dim=1)

            att_context_vector_pgen = model.tanh(model.lin2pgen_enccontext(att_context_vector))

            h1_gen = model.tanh(model.lin2pgen_decstate(h1))

            p_gen = torch.sigmoid(att_context_vector_pgen + h1_gen + decoder_input_gen)  # soft switch
            #print(p_gen)

            p_vocab = model.logsoftmax(model.lin_mid2voc(model.lin_hid2mid(torch.cat((h1, att_context_vector), dim=1))))

            # extend the vocab to add the temp unk IDs (for each example, the same temp ID can refer to a dif word)
            # has 1 superfluous index, if prev_utt = NOHS
            p_vocab_extended = torch.cat((p_vocab, torch.zeros(beam_k, len_unk_words_batch).to(device)), dim=1)

            final_gen_distribution = p_gen * p_vocab_extended  # gen prob for words actually in the vocab

            # pay attention to nohs, but don't add its att to vocab probs (because we don't want the generate it
            # if don't mask, it may contribute to the first temp unk ID!

            nohs_masked_att_weights = att_weights.clone().masked_fill_(masks_nohs, 0.0)

            final_att_distribution = (1 - p_gen) * model.logsoftmax(nohs_masked_att_weights.squeeze(2)) # att over words in prev utt

            # final_att_distribution = (1 - p_gen) * att_weights.squeeze()  # att over words in prev utt
            # can include temp unks

            # scatter to extended vocab (attention of words in mapped prev utt)

            total_word_prob = final_gen_distribution.scatter_add(1, mapped_prev_words, final_att_distribution)

            if nohs_flag:
                # no unk added, but we have NOHS to account for
                # but because of NOHS index 2790 (vocablen-1) is going to be there in mapped_prev_words
                # that index is unwanted (we don't want the decoder to generate it)
                # final att distr is masked against nohs anyway
                total_word_prob = total_word_prob[:, :-1]

            if coverage_flag:
                # cur_covloss = torch.min(att_weights.squeeze(), coverage_vector).sum(dim=1)
                #
                # covloss += cur_covloss

                coverage_vector = coverage_vector + att_weights

            word_pred =  total_word_prob

            word_pred = top_scores.expand_as(word_pred) + word_pred

            if gen_len == 0:
                # all same

                # static std::tuple<Tensor, Tensor> at::topk(const Tensor &self, int64_t k,
                # int64_t dim = -1, bool largest = true, bool sorted = true)

                top_scores, top_words = word_pred[0].topk(beam_k, 0, True, True)

            else:
                # unrolled
                top_scores, top_words = word_pred.view(-1).topk(beam_k, 0, True, True)

            # vocab - 1 to exclude <NOHS>
            sentence_index = top_words / word_pred.shape[1]  # which sentence it will be added to
            word_index = top_words % word_pred.shape[1]  # predicted word

            gen_len += 1

            # add the newly generated word to the sentences
            gen_sentences_k = torch.cat((gen_sentences_k[sentence_index], word_index.unsqueeze(1)), dim=1)

            # there could be incomplete sentences
            incomplete_sents_inds = [inc for inc in range(len(gen_sentences_k)) if
                                     eos_token not in gen_sentences_k[inc]]

            complete_sents_inds = list(set(range(len(word_index))) - set(incomplete_sents_inds))

            # save the completed sentences
            if len(complete_sents_inds) > 0:
                completed_sentences.extend(gen_sentences_k[complete_sents_inds].tolist())
                completed_scores.extend(top_scores[complete_sents_inds])

                beam_k -= len(complete_sents_inds)  # fewer, because we closed at least 1 beam

            if beam_k == 0:
                break

            # continue generation for the incomplete sentences
            gen_sentences_k = gen_sentences_k[incomplete_sents_inds]

            # use the ongoing hidden states of the incomplete sentences
            h1, c1 = h1[sentence_index[incomplete_sents_inds]], c1[sentence_index[incomplete_sents_inds]],
            coverage_vector = coverage_vector[sentence_index[incomplete_sents_inds]]

            top_scores = top_scores[incomplete_sents_inds].unsqueeze(1)
            decoder_input = word_index[incomplete_sents_inds]

        if len(completed_scores) == 0:

            empty_count += 1
            #print('emptyseq', empty_count)

            # all incomplete here

            completed_sentences.extend((gen_sentences_k[incomplete_sents_inds].tolist()))
            completed_scores.extend(top_scores[incomplete_sents_inds])

        sorted_scores, sorted_indices = torch.sort(torch.tensor(completed_scores), descending=True)

        best_seq = completed_sentences[sorted_indices[0]]

        hypothesis = ''

        for g in range(len(best_seq)):

            if best_seq[g] not in [vocab.word2index['<sos>'], vocab.word2index['<eos>'], vocab.word2index['<pad>']]:

                if best_seq[g] in vocab.index2word and best_seq[g] != len(vocab)-1:  # not nohs, above that value are other temp unks

                    hypothesis += vocab.index2word[best_seq[g]]

                else:
                    oov_index = unk_words[0][best_seq[g] - len(vocab) + 1]  # ind - 2790 + 1 # unk[0] single batch
                    hypothesis += vocab_full[int(oov_index)]

                if g < len(best_seq) - 1:
                    hypothesis += ' '

        # remove sos and pads # I want to check eos
        hypotheses.append(hypothesis)
        # print('HYP', hypothesis)
        # print(reference_chain)

        if not os.path.isfile('speaker_outputs/refs_' + file_name + '.json'):
            # Reference
            references.append(reference_chain)

        if print_gen:
            # Reference
            target_utt_list = target_utterance.tolist()[0]
            reference = [vocab_full.index2word[w] for w in target_utt_list if w not in
                         [vocab_full.word2index['<sos>'], vocab_full.word2index['<eos>'], vocab_full.word2index['<pad>']]]

            reference_string = [' '.join(reference)]

            print('REF:', reference_string) # single one
            print('HYP:', hypothesis)

    if os.path.isfile('speaker_outputs/refs_' + file_name + '.json'):
        with open('speaker_outputs/refs_' + file_name + '.json', 'r') as f:
            references = json.load(f)
    else:
        with open('speaker_outputs/refs_' + file_name + '.json', 'w') as f:
            json.dump(references, f)
    #
    # if os.path.isfile('speaker_outputs/refs_BERT_' + file_name + '.json'):
    #     with open('speaker_outputs/refs_BERT_' + file_name + '.json', 'r') as f:
    #         references_BERT = json.load(f)
    # else:
    #     references_BERT = [r[0] for r in references]
    #     with open('speaker_outputs/refs_BERT_' + file_name + '.json', 'w') as f:
    #         json.dump(references_BERT, f)

    # Calculate scores
    metrics_dict = nlgeval_obj.compute_metrics(references, hypotheses)
    print(metrics_dict)

    (P, R, Fs), hashname = score(hypotheses, references, lang='en', return_hash=True, model_type="bert-base-uncased")
    print(f'{hashname}: P={P.mean().item():.6f} R={R.mean().item():.6f} F={Fs.mean().item():.6f}')

    if args.metric == 'cider':
        selected_metric_score = metrics_dict['CIDEr']
        print(round(selected_metric_score, 5))

    elif args.metric == 'bert':
        selected_metric_score = Fs.mean().item()
        print(round(selected_metric_score, 5))

    # from https://github.com/Maluuba/nlg-eval
    # where references is a list of lists of ground truth reference text strings and hypothesis is a list of
    # hypothesis text strings. Each inner list in references is one set of references for the hypothesis
    # (a list of single reference strings for each sentence in hypothesis in the same order).

    if isValidation:
        has_best_score = False

        if selected_metric_score > best_score:
            best_score = selected_metric_score
            has_best_score = True

            with open('speaker_outputs/hyps_' + file_name + '.json', 'w') as f:
                json.dump(hypotheses, f)

        return best_score, selected_metric_score, metrics_dict, has_best_score

    if isTest:
        with open('speaker_outputs/hyps_' + file_name + '.json', 'w') as f:
            json.dump(hypotheses, f)


