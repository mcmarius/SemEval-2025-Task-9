import csv
import string
import os
import random
import re

import editdistance
import numpy as np
import sentence_transformers
import torch
import tqdm

from nltk.corpus import stopwords
from torch import nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score

from simplemma import lemmatize, text_lemmatizer
# from lemminflect import getLemma

# from suffix_trees import STree

from utils import read_clean_file, process_missing_entries, get_category_mappings, make_unique_list, make_clean_text


torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)

stopwords_list = set(stopwords.words("english"))
cos_sim = nn.CosineSimilarity(dim=1)


def compute_score(hazards_true, products_true, hazards_pred, products_pred):
  # compute f1 for hazards:
  f1_hazards = f1_score(
    hazards_true,
    hazards_pred,
    average='macro'
  )
  print(f"f1 hazards: {f1_hazards}")

  f1_prod = f1_score(
    products_true,
    products_pred,
    average='macro'
  )
  print(f"f1 products: {f1_prod}")

  # compute f1 for products:
  f1_products = f1_score(
    products_true[hazards_pred == hazards_true],
    products_pred[hazards_pred == hazards_true],
    average='macro'
  )

  return (f1_hazards + f1_products) / 2.


def label_in_text(model, word, match_words):
    encoded_word = torch.tensor(model.encode(word)).unsqueeze(0)
    other_words = torch.tensor(model.encode(match_words.split(" ")))
    matr = cos_sim(encoded_word, other_words)
    sims, sim_ind = torch.max(matr, dim=0)
    # print(sims)
    # print(sim_ind)
    # print(matr)
    # print(word)
    # print(match_words)
    return float(sims) > 0.8
    # exit(0)
    # for other_word in match_words.split(" "):
    #     encoded_other_word = torch.tensor(model.encode(other_word)).unsqueeze(0)
    #     if float(cos_sim(encoded_word, encoded_other_word)[0]) > 0.7:
    #         return True
    # return False


def process_entries(model, entries, gold_key, eval_key, have_labels=True, use_lemmatization=False, reorder_labels=None, skip_reorder_cat=False, reorder_labels_cat=False, predict_cat_first=False, label_set='', topk_true=1, use_stopwords=True):
    if label_set:
        gold_labels_file = f'data/{gold_key}{label_set}_labels.csv'
    else:
        gold_labels_file = f'data/{gold_key}_labels.csv'
    with open(gold_labels_file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        gold_labels = []
        # label2id = {}
        for row in reader:
            gold_labels.append(row[gold_key])
            # label2id[row[gold_key]] = row['id']
    label2id = {gold_labels[k]: k for k in range(len(gold_labels))}
    id2label = {k: gold_labels[k] for k in range(len(gold_labels))}
    parsed_entries = process_missing_entries(entries, eval_key)
    # print(gold_labels)
    # return
    # extracted_entries = list(entry[eval_key] for entry in entries)
    extracted_entries = parsed_entries
    mappings_data_file = 'data/incidents_train.csv'
    if label_set == 'valid':
        mappings_data_file = 'data/incidents_valid.csv'
    if label_set == 'test':
        mappings_data_file = 'data/incidents_test_labeled.csv'
    elif label_set == 'train_valid':
        mappings_data_file = 'data/incidents_train_valid.csv'
    elif label_set == 'train_valid_test':
        mappings_data_file = 'data/incidents_train_valid_test.csv'
    elif label_set == 'train_test':
        mappings_data_file = 'data/incidents_train_test.csv'
    cat2all, all2cat = get_category_mappings(gold_key, mappings_data_file)
    if use_lemmatization:
        cat2all = {k: [" ".join(lemmatize(word, lang='en', greedy=True) for word in labels.split(" ") if word) for labels in cat2all[k]] for k in cat2all}
    all_categories = list(cat2all.keys())
    id2label_cat = {k: all_categories[k] for k in range(len(all_categories))}
    ##############
    # if reorder_labels_cat or reorder_labels:
    if reorder_labels:
        if gold_key == 'product':
            # Order the following product categories from most specific to more general. Repsond only with a python list, lowercase, no explanations.
            # adjusted to move "other" as last element
            # ordered_categories = ['confectionery', 'ices and desserts', 'sugars and syrups', 'fats and oils', 'herbs and spices', 'honey and royal jelly', 'cocoa and cocoa preparations, coffee and tea', 'dietetic foods, food supplements, fortified foods', 'food additives and flavourings', 'food contact materials', 'feed materials', 'soups, broths, sauces and condiments', 'nuts, nut products and seeds', 'fruits and vegetables', 'seafood', 'cereals and bakery products', 'meat, egg and dairy products', 'prepared dishes and snacks', 'non-alcoholic beverages', 'alcoholic beverages', 'other food product / mixed']
            ordered_categories = ['honey and royal jelly', 'food additives and flavourings', 'food contact materials', 'nuts, nut products and seeds', 'feed materials', 'cocoa and cocoa preparations, coffee and tea', 'confectionery', 'soups, broths, sauces and condiments', 'sugars and syrups', 'fruits and vegetables', 'seafood', 'meat, egg and dairy products', 'fats and oils', 'pet feed', 'dietetic foods, food supplements, fortified foods', 'prepared dishes and snacks', 'ices and desserts', 'non-alcoholic beverages', 'alcoholic beverages', 'cereals and bakery products', 'herbs and spices', 'other food product / mixed']

        else:
            # Order the following hazard categories by importance and health risks, from most important to less important: 'chemical', 'food additives and flavourings', 'biological', 'organoleptic aspects', 'migration', 'foreign bodies', 'other hazard', 'allergens', 'packaging defect', 'fraud'. Respond only with a Python list, no explanations.
            ordered_categories = ["biological", "chemical", "allergens", "food additives and flavourings", "migration", "foreign bodies", "organoleptic aspects", "packaging defect", "fraud", "other hazard"]
        ordered_categories = {cat: i for i, cat in enumerate(ordered_categories)}
    ##############
    # Reorder labels
    if reorder_labels:
        if have_labels:
            if label_set == 'test':
                raw_data = read_clean_file('data/incidents_test_labeled.csv')
            elif label_set == 'valid':
                raw_data = read_clean_file('data/incidents_valid.csv')
            else:
                raw_data = read_clean_file('data/incidents_train.csv')
        else:
            # raw_data = read_clean_file('data/incidents_validation.csv')
            raw_data = read_clean_file('data/incidents_test.csv')
        if reorder_labels == 'reorder_':
            with open(f'data/{gold_key}{label_set}_labels_sorted_reverse.txt') as label_file:
                # ordered_labels = {label.strip(): i for i, label in enumerate(label_file.readlines())}
                ordered_labels = [label.strip() for label in label_file.readlines()]
        if reorder_labels == 'reorder_simple_':
            with open(f'data/{gold_key}{label_set}_labels_sorted2.txt') as label_file:
                # ordered_labels = {label.strip(): i for i, label in enumerate(label_file.readlines())}
                ordered_labels = [label.strip() for label in label_file.readlines()]
        elif reorder_labels == 'reorder_len_':
            with open(f'data/{gold_key}{label_set}_labels_sorted_len_reverse.txt') as label_file:
                # ordered_labels = {label.strip(): i for i, label in enumerate(label_file.readlines())}
                ordered_labels = [label.strip() for label in label_file.readlines()]
            # ordered_labels = [label.strip() for label in ordered_labels]
            # print(ordered_labels['other'])
            # exit(0)
        if gold_key == 'hazard':
            if not skip_reorder_cat:
                ordered_labels = sorted(ordered_labels, key=lambda x: ordered_categories[all2cat[x]])
        ordered_labels = {label: i for i, label in enumerate(ordered_labels)}
    ##############
    # new_entries = []
    # extremities_stopwords = {'defect. ', 'defects. ', 'contamination. ', '.\ncntamination.', '. contamination.'}
    # for i, entry in enumerate(extracted_entries):
    #     # if i < 53:
    #     #     continue
    #     new_entry = entry
    #     # print(f"old: {entry}")
    #     for word in extremities_stopwords:
    #         if new_entry.startswith(word):
    #             new_entry = new_entry[len(word):].strip()
    #         if new_entry.endswith(word):
    #             new_entry = new_entry[:-len(word)].strip()
    #     new_entries.append(new_entry)
    #     # print(f"new: {new_entry}")
    #     # if i > 60:
    #     #     exit(0)
    # extracted_entries = new_entries
    ################
    # Lemmatization
    if use_lemmatization:
        # extracted_entries = [" ".join(lemmatize(word, lang='en', greedy=True) for word in entry.split(" ") if word) for entry in extracted_entries]
        gold_labels = [" ".join(lemmatize(word, lang='en', greedy=True) for word in entry.split(" ") if word) for entry in gold_labels]
        # gold_labels = [" ".join(getLemma(word, upos='NOUN')[0] for word in entry.split(" ") if word) for entry in gold_labels]
        # if gold_key == 'product':
        #     extracted_entries = [" ".join(getLemma(word, upos='NOUN')[0] for word in re.split("[^a-zA-Z]+", entry) if word) for entry in extracted_entries]
        # else:
        extracted_entries = [" ".join(lemmatize(word, lang='en', greedy=True) for word in re.split("[^a-zA-Z]+", entry) if word) for entry in extracted_entries]
        # extracted_entries = [" ".join(make_unique_list([word for word in entry.split(" ")])) for entry in extracted_entries]
        # gold_labels = [" ".join(lemmatize(word, lang='en', greedy=True) for word in re.split("[^a-zA-Z]+", entry) if word) for entry in gold_labels]
    ##############
    #
    # extracted_entries = [" ".join(sorted(lemmatize(word, lang='en', greedy=True) for word in entry.split(" ") if word and word not in stopwords_list)) for entry in extracted_entries]
    # gold_labels = [" ".join(sorted(lemmatize(word, lang='en', greedy=True) for word in entry.split(" ") if word and word not in stopwords_list)) for entry in gold_labels]
    #
    # extracted_entries = [" ".join(text_lemmatizer(entry, lang='en', greedy=True)) for entry in extracted_entries]
    # gold_labels = [" ".join(text_lemmatizer(entry, lang='en', greedy=True)) for entry in gold_labels]
    #
    # extracted_entries = [" ".join([word for word in re.split("[^a-zA-Z]+", entry) if word not in specific_stopwords]) for entry in extracted_entries]
    # gold_labels = [" ".join([word for word in re.split(" ", entry) if word not in specific_stopwords]) for entry in gold_labels]
    if gold_key == 'hazard':
        specific_stopwords = {'products', 'thereof', 'containing', 'may', 'contain', 'including', 'spp', 'spp.', 'issues', 'abnormal', 'bad', 'fragment', 'matter', 'shaving', 'wire', 'contamination', 'defect', 'hazard', 'defects', 'hazards', ''}#, 'defects', 'hazards', ''}#, 'defects', 'hazards'} # , 'packaging', 'incorrect', 'unauthorized'}
    else:
        specific_stopwords = {'synonyms', 'food', 'product', 'category', 'definition', 'two', 'short'}
    if use_stopwords:
        extracted_entries = [" ".join([word for word in entry.split(" ") if word and word not in specific_stopwords]) for entry in extracted_entries]

    # workaround_9_01 = True
    # if workaround_9_01 and gold_key == 'product':
    #     # extracted_entries = [". ".join(entry.split(". ")[0:3]) for entry in extracted_entries]
    #     # extracted_entries = [entry.split(". -")[0] for entry in extracted_entries]
    #     # extracted_entries = [entry.split(" - ")[0] for entry in extracted_entries]
    #     extracted_entries = [entry.replace("uncommon", "").replace("common", "") for entry in extracted_entries]

    # if gold_key == 'product':
    #     extracted_entries = [entry[0:200] for entry in extracted_entries]
    # extracted_entries = [" ".join([word for word in re.split("[^a-zA-Z]+", entry) if word and word not in specific_stopwords]) for entry in extracted_entries]
    # gold_labels = [" ".join([word for word in entry.split(" ") if word not in specific_stopwords]) for entry in gold_labels]
    ##############
    # cuttoff = 100
    # print(extracted_entries[90:100])
    # exit(0)
    gold_label_embeddings = model.encode(gold_labels)
    gold_category_embeddings = model.encode(all_categories)
    extracted_label_embeddings = model.encode(extracted_entries)
    # extracted_label_embeddings = model.encode(extracted_entries[0:cuttoff])
    num_correct = 0
    num_wrong = 0
    preds = []
    golds = []
    probs = []
    preds_text = []
    golds_text = []
    debug_print = False
    for i, extracted_entry in enumerate(tqdm.tqdm(extracted_label_embeddings)):
        if debug_print:
            if i < 0:
                continue
        if predict_cat_first:
            matr_cat = cos_sim(torch.tensor(extracted_entry).unsqueeze(0), torch.tensor(gold_category_embeddings))
            sims_cat, sim_cat_ind = torch.max(matr_cat, dim=0)
            tops_cat, tops_cat_ind = torch.topk(matr_cat, 10, dim=0)
            probs_text_cat = ", ".join([f"{id2label_cat[int(x)]} ({round(float(tops_cat[i]), 5)})" for i, x in enumerate(tops_cat_ind)])
            # category = id2label_cat[int(sim_cat_ind)]
            # if debug_print:
            #     # print(f"got cat {category} with sublabels: {cat2all[category][0:10]} ({len(cat2all[category])} total)")
            #     print(f"predicted cats: {probs_text_cat}")
            # exit(0)
            # if reorder_labels_cat:
            #     old_category = category
            #     for j, top_cat_ind in enumerate(tops_cat_ind):
            #         if sims_cat - tops_cat[j] < 0.005 and ordered_categories[id2label_cat[int(top_cat_ind)]] < ordered_categories[category]:
            #             category = id2label_cat[int(top_cat_ind)]
            #             print(f"changed category from {old_category} to {category}")
            #             break
            # tops_cat_ind_reordered = sorted(tops_cat_ind, key=lambda x: ordered_categories[id2label_cat[int(x)]])
            # if debug_print:
            #     print(f"Tops cat ind reordered: {[id2label_cat[int(x)] for x in tops_cat_ind_reordered]}")
            # break_all = False
            # for j, top_cat_ind in enumerate(tops_cat_ind_reordered):
            #     for k, prob in enumerate(tops_cat):
            #         if tops_cat_ind[k] == top_cat_ind and sims_cat - prob < 0.05:
            #             category = id2label_cat[int(top_cat_ind)]
            #             break_all = True
            #             if debug_print:
            #                 print(f"changed category to {category}")
            #             break
            #     if break_all:
            #         break
            # exit(0)
            sims = 0
            sim_ind = -1
            topk_limit = 10
            id2full_id = {}
            matr = None
            for j, top_cat_ind in enumerate(tops_cat_ind):
            # for j, category in enumerate(all_categories):
                # if sims_cat - tops_cat[j] > 0.01:
                #     break
                category = id2label_cat[int(top_cat_ind)]
                label_inds = []
                for j, label in enumerate(gold_labels):
                    if label in cat2all[category]:
                        label_inds.append(j)
                # print(f"{j} cat: {category}, label inds: {label_inds}")
                # print(gold_labels[label_inds[1]])
                filtered_gold_embeddings = gold_label_embeddings[label_inds]
                # id2label_cand = {i: gold_labels[k] for i, k in enumerate(label_inds)}
                id2full_id_candidate = {i: k for i, k in enumerate(label_inds)}
                # print(id2full_id)
                # print(id2label_cand)
                # exit(0)
                # print(filtered_gold_embeddings.shape)
                matr_candidate = cos_sim(torch.tensor(extracted_entry).unsqueeze(0), torch.tensor(filtered_gold_embeddings))
                sims_candidate, sim_ind_candidate = torch.max(matr_candidate, dim=0)
                if sims_candidate > sims:
                    topk_limit = min(10, len(label_inds))
                    matr = matr_candidate
                    sims = sims_candidate
                    id2full_id = id2full_id_candidate
                    sim_ind = id2full_id[int(sim_ind_candidate)]
                    # if debug_print:
                    #     print(f"changed cat to {category}, pred is now {id2label[sim_ind]}")
                # predicted_label = id2label_cand[int(sim_ind)]
                # tops, tops_ind = torch.topk(matr, topk_limit, dim=0)
            # sim_ind = id2full_id[int(sim_ind)]
            tops, tops_ind = torch.topk(matr, topk_limit, dim=0)
            probs_text = ", ".join([f"{id2label[id2full_id[int(x)]]} ({round(float(tops[i]), 5)})" for i, x in enumerate(tops_ind)])
        else:
            matr = cos_sim(torch.tensor(extracted_entry).unsqueeze(0), torch.tensor(gold_label_embeddings))
            topk_limit = 10
            sims, sim_ind = torch.max(matr, dim=0)
            tops, tops_ind = torch.topk(matr, topk_limit, dim=0)
            probs_text = ", ".join([f"{id2label[int(x)]} ({round(float(tops[i]), 5)})" for i, x in enumerate(tops_ind)])
        predicted_label = id2label[int(sim_ind)]
        # print(sims)
        # print(tops)
        # # print(tops_ind)
        if debug_print:
            print(f'text: {extracted_entries[i]}')
            print(f'true label: {entries[i][gold_key]}')
            print(f'predicted label: {predicted_label}')
        # if debug_print:
        #     print(f'predicted labels: {probs_text[0:80]}')
        tops_ind_final = tops_ind
        if reorder_labels:
            sim_ind_init = sim_ind
            if debug_print:
                print(f"{i} Tops ind: {[id2label[int(x)] for x in tops_ind]}")
            tops_ind_reordered = sorted(tops_ind, key=lambda x: ordered_labels[id2label[int(x)]])
            tops_ind_final = tops_ind_reordered
            if debug_print:
                print(f"Tops ind reordered: {[id2label[int(x)] for x in tops_ind_reordered]}")
            # sim_ind = tops_ind_reordered[0]
            match_text = entries[i][eval_key].lower()
            raw_title = raw_data[i]['title'].lower()
            raw_text = raw_data[i]['text'].lower()
            # raw_text = make_clean_text(raw_data[i]['text'], gold_key).lower()
            match_text_words = re.split("[^a-zA-Z]+", match_text)
            raw_text_words = re.split("[^a-zA-Z]+", raw_text)
            #####
            count_matching_first = 0
            # for word in re.split("[^a-zA-Z]+", id2label[int(sim_ind)]):
            #     # for other_word in match_text_words:
            #     #     if editdistance.eval(word, other_word) < 2:
            #     #         count_matching_first += 1
            #     # for other_word in raw_text_words:
            #     #     if editdistance.eval(word, other_word) < 2:
            #     #         count_matching_first += 1
            #     # count_matching_first += max(match_text_words.count(word), raw_text_words.count(word))
            #     count_matching_first += raw_text_words.count(word)
            # if debug_print:
            #     print(f"count first: {count_matching_first}")
            #####
            stop_counting = False
            count_max = 0
            # TODO: always count the matches for top match
            for j, top_ind in enumerate(tops_ind_reordered):
                # if top_ind == sim_ind_init:
                #     continue
                # if id2label[int(top_ind)].lower() in entries[i][eval_key].lower():
                # TO DO?? if label_in_text(label, text)
                # if in text and if new label is more specific
                top_text = id2label[int(top_ind)]
                top_text_lower = top_text.lower()
                # if len(STree.STree([top_text, match_text]).lcs()) > len(top_text) / 2:
                # if top_text in match_text and len(id2label[int(top_ind)]) > len(id2label[int(sim_ind)]):
                # if top_text in raw_text and len(id2label[int(top_ind)]) > len(id2label[int(sim_ind)]):
                # if top_text in match_text and ordered_labels[id2label[int(top_ind)]] < ordered_labels[id2label[int(sim_ind)]]:
                #     sim_ind = top_ind
                #     break
                count_matching = 0
                other_prob = 0
                # potential_matches = top_text_lower.split()
                potential_matches = re.split("[^a-zA-Z]+", top_text_lower)
                # potential_matches.remove('') if '' in potential_matches else None
                # potential_matches = [lemmatize(word, lang='en', greedy=True) for word in potential_matches]
                for word in specific_stopwords | stopwords_list:
                    if use_stopwords and word in potential_matches:
                        potential_matches.remove(word)
                if debug_print:
                    print(f"potential_matches: {potential_matches}, text: {match_text_words}")
                for word in potential_matches:
                    # word_count = max(match_text_words.count(word), raw_text_words.count(word))
                    word_count = raw_text_words.count(word)
                    # word_count = 0
                    # for other_word in match_text_words:
                    #     if editdistance.eval(word, other_word) < 2:
                    #         word_count += 1
                    # for other_word in raw_text_words:
                    #     if editdistance.eval(word, other_word) < 2:
                    #         word_count += 1
                    # word_count = match_text_words.count(word) + raw_text_words.count(word)
                    if debug_print:
                        print(f"word: {word}, count: {word_count}, sim_ind_init: {sim_ind_init}, top_ind: {top_ind}")
                    if word_count > 0: # and (not stop_counting or sim_ind_init == top_ind):
                    # break_all = False
                    # if word in re.split("[^a-zA-Z]+", match_text) or word in re.split("[^a-zA-Z]+", raw_text):
                    # if word in re.split("[^a-zA-Z]+", raw_text) or word in re.split("[^a-zA-Z]+", match_text): # or label_in_text(model, word, match_text)): # [0:2*len(raw_text)//3]: # raw_title + " " + match_text + " " + raw_text:
                        for k, prob in enumerate(tops):
                            # do not change label if the difference is too big
                            if debug_print:
                                print(f"word: {word}, sim: {sims}, prob: {prob}, {tops_ind[k]}, {top_ind}")
                            if tops_ind[k] == top_ind and sims - prob < 0.1:
                                count_matching += word_count
                                other_prob = prob
                                # stop_counting = True
                                break
                            # early stopping since word-by-word similarity is slow
                            # if prob < 0.4:
                            #     break_all = True
                            #     break
                    # if break_all:
                    #     break
                                # print(f"found {word}")
                        # found = True
                        # break
                if sim_ind_init == top_ind:
                    count_matching_first = count_matching
                if count_max < count_matching:
                    count_max = count_matching
                # this limit is used to handle cases where we have exact match with single word labels, but we also want to prevent
                # labels with more than one word from a partial match
                match_limit = 1
                if len(potential_matches) > 1:
                    match_limit = 2
                if debug_print:
                    print(f"count is {count_matching}, top: {top_text}, actual label: {id2label[int(sim_ind)]}, match limit: {match_limit}, o1: {ordered_labels[top_text]}, o2: {ordered_labels[id2label[int(sim_ind)]]}")

                if count_max == count_matching:
                    if (count_matching > 1) or \
                        (count_matching >= match_limit and \
                            (ordered_labels[top_text] < ordered_labels[id2label[int(sim_ind_init)]] or count_matching > count_matching_first)):
                    # if count_matching > 1 or (count_matching >= match_limit and ordered_labels[top_text] < ordered_labels[id2label[int(sim_ind)]]):
                        if debug_print:
                            print(f"{i} change from {id2label[int(sim_ind)]} to {top_text}")
                        sim_ind = top_ind
                        # break
                elif count_matching > 0 and count_matching_first == 0 and sims - other_prob < 0.01:
                    if debug_print:
                        print(f"{i} change from {id2label[int(sim_ind)]} to {top_text}")
                    sim_ind = top_ind
                # if count_matching_first > 0 and sim_ind_init == top_ind and sims > 0.5:
                #     if debug_print:
                #         print(f"{i} change from {id2label[int(sim_ind)]} to {top_text}")
                #     sim_ind = top_ind
                #     break
        if debug_print:
            if i > 5:
                exit(0)
        probs.append(probs_text)
        # exit(0)
        # take into account reordering
        predicted_label = id2label[int(sim_ind)]
        true_in_topk = False
        if have_labels and topk_true > 0:
            true_in_topk = entries[i][gold_key] in [id2label[int(top_ind)] for top_ind in tops_ind_final[:topk_true]]
        if true_in_topk:
            preds_text.append(entries[i][gold_key])
        else:
            preds_text.append(predicted_label)
        if have_labels:
            if id2label[int(sim_ind)] == entries[i][gold_key] or true_in_topk:
                num_correct += 1
            else:
                num_wrong += 1
        # if predict_cat_first:
        #     preds.append(id2full_id[int(sim_ind)])
        # else:
        preds.append(int(sim_ind))
        if have_labels:
            if label2id.get(entries[i][gold_key]):
                golds.append(label2id[entries[i][gold_key]])
            else:
                golds.append(0) # irrelevant since we do not use this; kept for legacy for now
            golds_text.append(entries[i][gold_key])
        else:
            golds.append(0)
            golds_text.append('missing')
        # if i > 50:
        #     break
    print(f'correct: {num_correct}; wrong: {num_wrong}')
    return np.array(preds), golds, id2label, probs, golds_text, preds_text





def main(have_gold_labels=True):
    # model_name = 'whaleloops/phrase-bert'
    # model_name = 'sentence-transformers/all-mpnet-base-v2'
    # hazard_model_name = 'sentence-transformers/all-mpnet-base-v2'
    hazard_model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    # hazard_model_name = 'trained_models/cosent-pos2-hazard'
    # hazard_model_name = 'sentence-transformers/paraphrase-MiniLM-L12-v2'
    # hazard_model_name = 'thenlper/gte-large'
    product_model_name = 'thenlper/gte-small'
    product_model_name = 'thenlper/gte-base'
    product_model_name = 'thenlper/gte-large'
    # product_model_name = 'Dizex/FoodBaseBERT-NER' # really bad
    # product_model_name = 'BAAI/bge-base-en-v1.5'
    # product_model_name = 'trained_models/cosent-lr2-w2-pos4-neutr1-neg1-1epoch-product-base'
    # product_model_name = 'trained_models/cosent-3sample-minority-lr2-w2-pos2-neutr1-neg1-1epoch-product-base'
    # product_model_name = 'trained_models/angloss-lr2-w2-pos2-neutr1-neg1-1epoch-product-base'
    # product_model_name = 'sentence-transformers/paraphrase-MiniLM-L12-v2'
    # product_model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    # model_name = 'kamalkraj/BioSimCSE-BioLinkBERT-BASE'
    model_name2 = hazard_model_name.replace('/', '_')
    if hazard_model_name != product_model_name:
        model_name2 += product_model_name.replace('/', '_')
    os.makedirs(f'results/{model_name2}', exist_ok=True)
    os.makedirs(f'submissions/{model_name2}', exist_ok=True)
    # file_name = 'parsed_data.csv'
    # hazard_file_name = 'parsed_data_prompt_h10_f10-q4-k-m.csv'
    # hazard_file_name = 'parsed_data_prompt_h14.csv'
    # hazard_file_name = 'parsed_data_f24_h17.csv'
    hazard_file_name = 'parsed_data_h25.csv'
    hazard_file_name = 'parsed_data_h25_ollama_repro.csv'  # <- used in paper
    # hazard_file_name = 'parsed_data_h25_2_ollama_regex_more_words.csv'
    # hazard_file_name = 'parsed_data_h25_4_ollama_regex.csv'
    # hazard_file_name = 'parsed_data_h25_4_ollama_regex_valid.csv'
    hazard_file_name = 'parsed_data_h25_ollama_regex_valid_v4.csv'  # <- used in paper
    hazard_file_name = 'parsed_data_h25_ollama_regex_test2.csv'  # <- used in paper
    # hazard_file_name = 'parsed_data_no_prod_regex_test_no_clean.csv'
    # hazard_file_name = 'parsed_data_h25_ollama_regex_test_labeled.csv'
    # hazard_file_name = 'parsed_data_h29_ollama_regex_more_words.csv'
    # hazard_file_name = 'parsed_data_h25_validation.csv'
    # hazard_file_name = 'parsed_data_f25_h18.csv'
    # hazard_file_name = 'parsed_data_f24_h17_validation.csv'
    # hazard_file_name = 'parsed_data_f23_h16_first1.csv'
    # hazard_file_name = 'parsed_data_validation_h14_f10_q6.csv'
    hazard_entries = read_clean_file(hazard_file_name)
    # product_file_name = hazard_file_name
    # product_file_name = 'parsed_data_f24_h17.csv'
    # product_file_name = 'parsed_data_f24_ollama.csv'
    product_file_name = 'parsed_data_f24_ollama_no_regex.csv'
    # product_file_name = 'parsed_data_f24_ollama_regex_valid.csv'
    # product_file_name = 'parsed_data_f44_ollama_no_regex_valid.csv'
    product_file_name = 'parsed_data_f24_ollama_no_regex_valid_v2.csv'
    product_file_name = 'parsed_data_f24_ollama_no_regex_test2.csv'
    # product_file_name = 'parsed_data_no_prod_regex_test_no_clean.csv'
    # product_file_name = 'parsed_data_f24_ollama_no_regex_test_labeled.csv'
    # product_file_name = 'parsed_data_f24_ollama_no_regex_repro.csv'
    # product_file_name = 'parsed_data_f24_ollama_no_regex_q8.csv'
    # product_file_name = 'parsed_data_f42_ollama_regex.csv'
    # product_file_name = 'parsed_data_f36.csv'
    # product_file_name = 'parsed_data_f36_first100.csv'
    # product_file_name = 'parsed_data_f24_h17_validation.csv'
    # product_file_name = 'parsed_data_f25_h18.csv'
    # product_file_name = 'parsed_data_f25_h18_validation.csv'
    # product_file_name = 'parsed_data_f31_validation.csv'
    # product_file_name = 'parsed_data_prompt_h10_f10-q4-k-m.csv'
    # product_file_name = 'parsed_data_prompt_f10.csv'
    # product_file_name = 'parsed_data_validation_h14_f10_q6.csv'
    product_entries = read_clean_file(product_file_name)
    if 'validation' in hazard_file_name or 'validation' in product_file_name or 'test.csv' in hazard_file_name or 'test.csv' in product_file_name:
        have_gold_labels = False
    # have_gold_labels = False
    eval_label_set = 'train_valid'
    eval_label_set = 'train_valid_test'
    # eval_label_set = 'test_labeled'
    # eval_label_set = 'valid'
    # def read_all_gold_labels(gold_key):
    #     all_gold_labels_file = f'data/{gold_key}{eval_label_set}_labels.csv'
    #     with open(all_gold_labels_file) as csvfile:
    #         reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
    #         all_gold_labels = []
    #         label2id = {}
    #         for row in reader:
    #             all_gold_labels.append(row[gold_key])
    #             label2id[row[gold_key]] = row['id']
    #     return all_gold_labels, label2id
    # all_hazard_gold_labels, hazard_all_gold_label2id = read_all_gold_labels('hazard')
    # print(sorted(hazard_all_gold_label2id.keys()))
    # print(sorted(all_hazard_gold_labels))
    # exit(0)
    # all_product_gold_labels, product_all_gold_label2id = read_all_gold_labels('product')

    label_set = 'train_valid'
    label_set = 'train_valid_test'
    # label_set = 'valid'
    label_set = 'test'
    # label_set = ''
    # hazard_pred, hazard_gold = process_entries(model, hazard_entries, 'hazard-category', 'extracted_hazard')
    # product_pred, product_gold = process_entries(model, product_entries, 'product-category', 'extracted_product')
    # hazard_pred, hazard_gold = process_entries(model, entries, 'hazard-category', 'title')
    # product_pred, product_gold = process_entries(model, entries, 'product-category', 'title')
    # print(f'f1 category: {compute_score(hazard_gold, product_gold, hazard_pred, product_pred)}')
    hazard_exist = False
    use_cache = True
    use_cache = False
    # reorder_labels = 'reorder_len_'
    reorder_labels = 'reorder_'
    reorder_labels = 'reorder_simple_'
    reorder_labels = ''
    pred_cat_first = 'pred_cat_'
    pred_cat_first = ''
    skip_reorder_cat = False
    # skip_reorder_cat = True
    use_stopwords = True
    # use_stopwords = False
    use_lemmatization = True
    # use_lemmatization = False
    topk_true = 10
    if os.path.isfile(f'results/{model_name2}/hazard_{reorder_labels}{pred_cat_first}{hazard_file_name}') and use_cache:
        hazard_exist = True
        hazard_pred = []
        hazard_pred_text = []
        hazard_gold = []
        hazard_gold_text = []
        with open(f'results/{model_name2}/hazard_{reorder_labels}{pred_cat_first}{hazard_file_name}', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                hazard_gold.append(row['hazard-gold'])
                hazard_pred.append(row['hazard-pred'])
                # hazard_pred_text.append(row['hazard-pred-text'])
                # hazard_gold_text.append(row['hazard-gold-text'])
        hazard_pred_text = hazard_pred
        hazard_gold_text = hazard_gold
        hazard_pred = np.array(hazard_pred)
        hazard_gold = np.array(hazard_gold)
    else:
        # reorder_labels = ''
        model = SentenceTransformer(hazard_model_name)
        hazard_pred, hazard_gold, id2label, probs, hazard_gold_text, hazard_pred_text = process_entries(model, hazard_entries, 'hazard', 'extracted_hazard', have_labels=have_gold_labels, use_lemmatization=use_lemmatization, reorder_labels=reorder_labels, predict_cat_first=pred_cat_first, label_set=label_set, topk_true=topk_true, skip_reorder_cat=skip_reorder_cat, use_stopwords=use_stopwords)
        # hazard_pred_text = [id2label[pred] for pred in hazard_pred]
        # hazard_gold_text = [id2label[pred] for pred in hazard_gold]
        # hazard_gold = [hazard_all_gold_label2id[pred] for pred in hazard_gold_text]
        # hazard_pred = [hazard_all_gold_label2id[pred] for pred in hazard_pred_text]
        # hazard_gold = np.array(hazard_gold)
        # hazard_pred = np.array(hazard_pred)
        hazard_gold = np.array(hazard_gold_text)
        hazard_pred = np.array(hazard_pred_text)
        with open(f'results/{model_name2}/hazard_{reorder_labels}{pred_cat_first}{hazard_file_name}', 'w', newline='') as csvfile:
            fieldnames = ['hazard-gold', 'hazard-pred',
                          # 'hazard-gold-text', 'hazard-pred-text',
                          'probs']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
            writer.writeheader()
            for i in range(len(hazard_pred)):
                writer.writerow({
                    'hazard-gold': hazard_gold[i], 'hazard-pred': hazard_pred[i],
                    # 'hazard-gold-text': hazard_gold_text[i], 'hazard-pred-text': hazard_pred_text[i],
                    'probs': probs[i]
                })
    product_exist = False
    use_cache = True
    use_cache = False
    # pred_cat_first = 'pred_cat_'
    pred_cat_first = ''
    if os.path.isfile(f'results/{model_name2}/product_{pred_cat_first}{product_file_name}') and use_cache:
        product_exist = True
        product_pred = []
        product_pred_text = []
        product_gold = []
        product_gold_text = []
        with open(f'results/{model_name2}/product_{pred_cat_first}{product_file_name}', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                product_gold.append(row['product-gold'])
                product_pred.append(row['product-pred'])
                # product_pred_text.append(row['product-pred-text'])
                # product_gold_text.append(row['product-gold-text'])
        product_pred_text = product_pred
        product_gold_text = product_gold
        product_pred = np.array(product_pred)
        product_gold = np.array(product_gold)
    else:
        model = SentenceTransformer(product_model_name)
        product_pred, product_gold, id2label, probs, product_gold_text, product_pred_text = process_entries(model, product_entries, 'product', 'extracted_product', have_labels=have_gold_labels, use_lemmatization=False, reorder_labels_cat=False, predict_cat_first=pred_cat_first, label_set=label_set, topk_true=topk_true)
        # product_pred_text = [id2label[pred] for pred in product_pred]
        # product_gold_text = [id2label[pred] for pred in product_gold]
        # product_gold = [product_all_gold_label2id[pred] for pred in product_gold_text]
        # product_pred = [product_all_gold_label2id[pred] for pred in product_pred_text]
        # product_gold = np.array(product_gold)
        # product_pred = np.array(product_pred)
        product_gold = np.array(product_gold_text)
        product_pred = np.array(product_pred_text)
        with open(f'results/{model_name2}/product_{pred_cat_first}{product_file_name}', 'w', newline='') as csvfile:
            fieldnames = ['product-gold', 'product-pred',
                          # 'product-gold-text', 'product-pred-text',
                          'probs']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
            writer.writeheader()
            for i in range(len(product_pred)):
                writer.writerow({
                    'product-gold': product_gold[i], 'product-pred': product_pred[i],
                    # 'product-gold-text': product_gold_text[i], 'product-pred-text': product_pred_text[i],
                    'probs': probs[i]
                })

    submission_file_name = hazard_file_name
    if submission_file_name != product_file_name:
        submission_file_name = submission_file_name.split('.')[0] + '_' + product_file_name
    if not have_gold_labels:
        # submission_file_name = 'validation_' + submission_file_name
        submission_file_name = 'test_' + submission_file_name
    with open(f'submissions/{model_name2}/{submission_file_name}', 'w', newline='') as csvfile:
        fieldnames = ['hazard-category', 'product-category', 'hazard', 'product']
        # if have_gold_labels:
        #     fieldnames = ['hazard-category', 'hazard-category-gold', 'product-category', 'product-category-gold', 'hazard', 'hazard-gold','product', 'product-gold']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
        writer.writeheader()
        #
        mappings_data_file = 'data/incidents_train.csv'
        if label_set == 'valid':
            mappings_data_file = 'data/incidents_valid.csv'
        elif label_set == 'train_valid':
            mappings_data_file = 'data/incidents_train_valid.csv'
        elif label_set == 'train_valid_test':
            mappings_data_file = 'data/incidents_train_valid_test.csv'
        elif label_set == 'test':
            mappings_data_file = 'data/incidents_test_labeled.csv'
        cat2all_hazard, all2cat_hazard = get_category_mappings('hazard', mappings_data_file)
        cat2all_product, all2cat_product = get_category_mappings('product', mappings_data_file)
        for i in range(min(len(product_pred), len(hazard_pred))):
            row = {'hazard-category': all2cat_hazard[hazard_pred_text[i]],
                             'product-category': all2cat_product[product_pred_text[i]],
                             'hazard': hazard_pred_text[i], 'product': product_pred_text[i]}
                    # if have_gold_labels:
                    #     row['product-category-gold'] = all2cat_product[product_gold_text[i]]
                    #     row['hazard-category-gold'] = all2cat_hazard[hazard_gold_text[i]]
                    #     row['product-gold'] = product_gold_text[i]
                    #     row['hazard-gold'] = hazard_gold_text[i]
            writer.writerow(row)
    if have_gold_labels:
        f1 = compute_score(hazard_gold, product_gold, hazard_pred, product_pred)
        print(f'score vector: {f1}')
        acc = accuracy_score(hazard_gold, hazard_pred)
        print(f'acc hazard: {acc}')
        acc = accuracy_score(product_gold, product_pred)
        print(f'acc prod: {acc}')
        with open(f'results/{model_name2}/score_{hazard_file_name.split(".")[0]}.txt', 'w') as f:
            f.write(f'f1: {f1}')
        #

        # label2id_gold = {hazard_gold_text[k]: k for k in range(len(hazard_gold_text))}
        hazard_cats = list(cat2all_hazard.keys())
        product_cats = list(cat2all_product.keys())
        # label2id_hazard_cat = {hazard_cats[k]: k for k in range(len(hazard_cats))}
        # label2id_product_cat = {product_cats[k]: k for k in range(len(product_cats))}
        # use true labels now
        mappings_data_file = f'data/incidents_{eval_label_set}.csv'
        cat2all_hazard_gold, all2cat_hazard_gold = get_category_mappings('hazard', mappings_data_file)
        cat2all_product_gold, all2cat_product_gold = get_category_mappings('product', mappings_data_file)
        hazard_cats_pred = []
        hazard_cats_gold = []
        product_cats_pred = []
        product_cats_gold = []
        for i in range(len(hazard_pred)):
            # hazard_cats_gold.append(label2id_hazard_cat[all2cat_hazard_gold[hazard_gold_text[i]]])
            # hazard_cats_pred.append(label2id_hazard_cat[all2cat_hazard[hazard_pred_text[i]]])
            hazard_cats_gold.append(all2cat_hazard_gold[hazard_gold_text[i]])
            hazard_cats_pred.append(all2cat_hazard[hazard_pred_text[i]])
        for i in range(len(product_pred)):
            # product_cats_gold.append(label2id_product_cat[all2cat_product_gold[product_gold_text[i]]])
            # product_cats_pred.append(label2id_product_cat[all2cat_product[product_pred_text[i]]])
            product_cats_gold.append(all2cat_product_gold[product_gold_text[i]])
            product_cats_pred.append(all2cat_product[product_pred_text[i]])
        hazard_cats_gold = np.array(hazard_cats_gold)
        hazard_cats_pred = np.array(hazard_cats_pred)
        product_cats_gold = np.array(product_cats_gold)
        product_cats_pred = np.array(product_cats_pred)
        f1 = compute_score(hazard_cats_gold, product_cats_gold, hazard_cats_pred, product_cats_pred)
        print(f'score cat: {f1}')
        acc = accuracy_score(hazard_cats_gold, hazard_cats_pred)
        print(f'acc cat hazard: {acc}')
        acc = accuracy_score(product_cats_gold, product_cats_pred)
        print(f'acc cat prod: {acc}')

if __name__ == '__main__':
    main()
