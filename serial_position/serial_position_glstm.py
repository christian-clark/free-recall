import argparse
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F

import dictionary_corpus
from utils import get_batch, repackage_hidden


LIST_COUNT = 500
#LIST_COUNT = 5
PREFACE = "In the morning , I saw"
CONTINUATION = "In the afternoon , I again encountered"

# GLSTM model weights
CHECKPOINT = "/users/PAS2157/ceclark/git/cued-recall/cached_lms/hidden650_batch128_dropout0.2_lr20.0.pt"
# GLSTM vocab
DATA = "/users/PAS2157/ceclark/git/modelblocks-release/resource-glstm/src"
EVAL_BATCH_SIZE = 1
# length of chunks of the input sequence that get fed into the LSTM
SEQ_LEN = 100


def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


# TODO update name list to exclude names that aren't in vocab
def generate_name_lists(names_fn, names_per_list, dictionary):
    names = list()
    for l in open(names_fn):
        if l.strip() in dictionary.word2idx:
            names.append(l.strip())
    names = np.array(names)
    name_lists = list()
    for _ in range(LIST_COUNT):
        random_names = np.random.choice(names, names_per_list, replace=False).tolist()
        name_lists.append(random_names)
    return name_lists


def words_to_ids(dictionary, words):
    words_eos = words + ["<eos>"]
    ids = torch.LongTensor(len(words_eos))
    for i, word in enumerate(words_eos):
        assert word in dictionary.word2idx, "word not in dictionary: {}".format(word)
        ids[i] = dictionary.word2idx[word]
    return ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("names")
    parser.add_argument("--length", '-l', type=int, help="Number of pairs per list", default=10)
    parser.add_argument("--seed", '-s', type=int, help="random seed")
    parser.add_argument("--gpu", '-g', action="store_true", help="use GPU")
    args = parser.parse_args()
    if args.seed:
        np.random.seed(args.seed)

    with open(CHECKPOINT, 'rb') as f:
        eprint("Loading model from {}".format(CHECKPOINT))
        if args.gpu:
            model = torch.load(f)
        else:
            # to convert model trained on cuda to cpu model
            model = torch.load(f, map_location = lambda storage, loc: storage)

    model.eval()
    hidden = model.init_hidden(EVAL_BATCH_SIZE)

    if args.gpu:
        device = "cuda"
        model.cuda()
    else:
        device = "cpu"
        model.cpu()

    model = model.to(device)
    dictionary = dictionary_corpus.Dictionary(DATA)
    vocab_size = len(dictionary)

    print("names tgtIx baselineSurp surp surpRatio")
    name_lists = generate_name_lists(args.names, args.length, dictionary)
    for name_list in name_lists:
        input_words = PREFACE.split()

        assert len(name_list) >= 3, "logic assumes 3+ names"
        for n_ix, name in enumerate(name_list):
            #name_locs.append(len(input_ids))
            # "and" before final conjunct
            if n_ix == len(name_list) - 2:
                input_words.extend([name, ",", "and"])
            elif n_ix == len(name_list) - 1:
                input_words.extend([name, "."])
            else:
                input_words.extend([name, ","])


        # process main sequence
        input_words.extend(CONTINUATION.split())
        model_input = words_to_ids(dictionary, input_words).unsqueeze(-1)
        model_input = model_input.to(device)
        surprisals = list()
        curr_hidden = tuple(x.clone() for x in hidden)
        for i in range(0, model_input.size(0) - 1, SEQ_LEN):
            data, targets = get_batch(model_input, i, SEQ_LEN)
            output, curr_hidden = model(data, curr_hidden)
            output_flat = output.view(-1, vocab_size)
            softmax_probs = F.softmax(output_flat, dim=1).detach()
            curr_log_probs_np = -1 * np.log2(softmax_probs.cpu().numpy())
            targets_np = targets.cpu().numpy()
            for scores, correct_label in zip(curr_log_probs_np, targets_np):
                surp = scores[correct_label]
                surprisals.append(surp)
            curr_hidden = repackage_hidden(curr_hidden)
        final_scores = curr_log_probs_np[-1]

        # process baseline sequence
        baseline_input_words = CONTINUATION.split()
        model_input = words_to_ids(dictionary, baseline_input_words).unsqueeze(-1)
        model_input = model_input.to(device)
        surprisals = list()
        curr_hidden = tuple(x.clone() for x in hidden)
        for i in range(0, model_input.size(0) - 1, SEQ_LEN):
            data, targets = get_batch(model_input, i, SEQ_LEN)
            output, curr_hidden = model(data, curr_hidden)
            output_flat = output.view(-1, vocab_size)
            softmax_probs = F.softmax(output_flat, dim=1).detach()
            curr_log_probs_np = -1 * np.log2(softmax_probs.cpu().numpy())
            targets_np = targets.cpu().numpy()
            for scores, correct_label in zip(curr_log_probs_np, targets_np):
                surp = scores[correct_label]
                surprisals.append(surp)
            curr_hidden = repackage_hidden(curr_hidden)
        baseline_final_scores = curr_log_probs_np[-1]

        for nix, name in enumerate(name_list):
            target_id = dictionary.word2idx[name]
            target_surp = final_scores[target_id]
            baseline_target_surp = baseline_final_scores[target_id]

            print(
                ','.join(name_list), nix, baseline_target_surp, target_surp,
                target_surp/baseline_target_surp
            )



if __name__ == "__main__":
    main()
