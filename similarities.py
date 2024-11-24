import csv
import string
import os
import random

import numpy as np
import sentence_transformers
import torch
import tqdm

from torch import nn
from sentence_transformers import SentenceTransformer
from sklearn.metrics import f1_score, accuracy_score

torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)

def read_file(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        entries = [row for row in reader]
        # header = entries[0]
        # entries = entries[1:]
        return entries


def compute_score(hazards_true, products_true, hazards_pred, products_pred):
  # compute f1 for hazards:
  f1_hazards = f1_score(
    hazards_true,
    hazards_pred,
    average='macro'
  )

  # compute f1 for products:
  f1_products = f1_score(
    products_true[hazards_pred == hazards_true],
    products_pred[hazards_pred == hazards_true],
    average='macro'
  )

  return (f1_hazards + f1_products) / 2.


def process_entries(model, entries, gold_key, eval_key, have_labels=True):
    if not have_labels:
        with open(f'data/{gold_key}_labels.csv') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            gold_categories = []
            label2id = {}
            for row in reader:
                gold_categories.append(row[gold_key])
                label2id[row[gold_key]] = row['id']
    else:
        gold_categories = list(set(entry[gold_key] for entry in entries))
        label2id = {gold_categories[k]: k for k in range(len(gold_categories))}
        if not os.path.isfile(f'data/{gold_key}_labels.csv'):
            with open(f'data/{gold_key}_labels.csv', 'w', newline='') as csvfile:
                fieldnames = [gold_key, 'id']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
                writer.writeheader()
                for line in gold_categories:
                    writer.writerow({gold_key: line, 'id': label2id[line]})
    label2id = {gold_categories[k]: k for k in range(len(gold_categories))}
    id2label = {k: gold_categories[k] for k in range(len(gold_categories))}
    parsed_entries = []
    for entry in entries:
        new_entry = entry[eval_key].lower().replace('\n', ' ').strip(string.punctuation).strip()
        if 'missing' in new_entry:
            new_entry = new_entry.replace('missing', '').strip(string.punctuation).strip()
            if new_entry == '':
                new_entry = entry['title']
        parsed_entries.append(new_entry)
    # return
    # extracted_entries = list(entry[eval_key] for entry in entries)
    extracted_entries = parsed_entries
    gold_label_embeddings = model.encode(gold_categories)
    extracted_label_embeddings = model.encode(extracted_entries)
    cos_sim = nn.CosineSimilarity(dim=1)
    num_correct = 0
    num_wrong = 0
    preds = []
    golds = []
    for i, extracted_entry in enumerate(tqdm.tqdm(extracted_label_embeddings)):
        max_similarity = 0
        predicted_label = None
        matr = cos_sim(torch.tensor(extracted_entry).unsqueeze(0), torch.tensor(gold_label_embeddings))
        sims, sim_ind = torch.max(matr, dim=0)
        if have_labels:
            if id2label[int(sim_ind)] == entries[i][gold_key]:
                num_correct += 1
            else:
                num_wrong += 1
        preds.append(int(sim_ind))
        if have_labels:
            golds.append(label2id[entries[i][gold_key]])
        else:
            golds.append(0)
    #     # if i > 50:
    #     #     break
    print(f'correct: {num_correct}; wrong: {num_wrong}')
    return np.array(preds), np.array(golds), id2label





def main(have_gold_labels=True):
    # model_name = 'whaleloops/phrase-bert'
    # model_name = 'sentence-transformers/all-mpnet-base-v2'
    # hazard_model_name = 'sentence-transformers/all-mpnet-base-v2'
    hazard_model_name = 'sentence-transformers/paraphrase-MiniLM-L6-v2'
    # hazard_model_name = 'sentence-transformers/paraphrase-MiniLM-L12-v2'
    # hazard_model_name = 'thenlper/gte-large'
    # product_model_name = 'thenlper/gte-small'
    # product_model_name = 'thenlper/gte-base'
    product_model_name = 'thenlper/gte-large'
    # product_model_name = 'sentence-transformers/paraphrase-MiniLM-L12-v2'
    # model_name = 'kamalkraj/BioSimCSE-BioLinkBERT-BASE'
    model_name2 = hazard_model_name.replace('/', '_')
    if hazard_model_name != product_model_name:
        model_name2 += product_model_name.replace('/', '_')
    os.makedirs(f'results/{model_name2}', exist_ok=True)
    os.makedirs(f'submissions/{model_name2}', exist_ok=True)
    # file_name = 'parsed_data.csv'
    # hazard_file_name = 'parsed_data_prompt_h10_f10-q4-k-m.csv'
    hazard_file_name = 'parsed_data_prompt_h14.csv'
    # hazard_file_name = 'parsed_data_validation_h14_f10_q6.csv'
    product_file_name = hazard_file_name
    hazard_entries = read_file(hazard_file_name)
    # product_file_name = 'parsed_data_prompt_h10_f10-q4-k-m.csv'
    product_file_name = 'parsed_data_prompt_f10.csv'
    # product_file_name = 'parsed_data_validation_h14_f10_q6.csv'
    product_entries = read_file(product_file_name)
    if 'validation' in hazard_file_name or 'validation' in product_file_name:
        have_gold_labels = False
    # hazard_pred, hazard_gold = process_entries(model, hazard_entries, 'hazard-category', 'extracted_hazard')
    # product_pred, product_gold = process_entries(model, product_entries, 'product-category', 'extracted_food')
    # hazard_pred, hazard_gold = process_entries(model, entries, 'hazard-category', 'title')
    # product_pred, product_gold = process_entries(model, entries, 'product-category', 'title')
    # print(f'f1 category: {compute_score(hazard_gold, product_gold, hazard_pred, product_pred)}')
    hazard_exist = False
    if os.path.isfile(f'results/{model_name2}/hazard_{hazard_file_name}'):
        hazard_exist = True
        hazard_pred = []
        hazard_pred_text = []
        hazard_gold = []
        with open(f'results/{model_name2}/hazard_{hazard_file_name}', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                hazard_gold.append(row['hazard-gold'])
                hazard_pred.append(row['hazard-pred'])
                hazard_pred_text.append(row['hazard-pred-text'])
        hazard_pred = np.array(hazard_pred)
        hazard_gold = np.array(hazard_gold)
    else:
        model = SentenceTransformer(hazard_model_name)
        hazard_pred, hazard_gold, id2label = process_entries(model, hazard_entries, 'hazard', 'extracted_hazard', have_labels=have_gold_labels)
        hazard_pred_text = [id2label[pred] for pred in hazard_pred]
        with open(f'results/{model_name2}/hazard_{hazard_file_name}', 'w', newline='') as csvfile:
            fieldnames = ['hazard-gold', 'hazard-pred', 'hazard-gold-text', 'hazard-pred-text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
            writer.writeheader()
            for i in range(len(hazard_pred)):
                writer.writerow({
                    'hazard-gold': hazard_gold[i], 'hazard-pred': hazard_pred[i],
                    'hazard-gold-text': id2label[hazard_gold[i]], 'hazard-pred-text': id2label[hazard_pred[i]]
                })
    product_exist = False
    if os.path.isfile(f'results/{model_name2}/product_{product_file_name}'):
        product_exist = True
        product_pred = []
        product_pred_text = []
        product_gold = []
        with open(f'results/{model_name2}/product_{product_file_name}', newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
            for row in reader:
                product_gold.append(row['product-gold'])
                product_pred.append(row['product-pred'])
                product_pred_text.append(row['product-pred-text'])
        product_pred = np.array(product_pred)
        product_gold = np.array(product_gold)
    else:
        model = SentenceTransformer(product_model_name)
        product_pred, product_gold, id2label = process_entries(model, product_entries, 'product', 'extracted_food', have_labels=have_gold_labels)
        product_pred_text = [id2label[pred] for pred in product_pred]
        with open(f'results/{model_name2}/product_{product_file_name}', 'w', newline='') as csvfile:
            fieldnames = ['product-gold', 'product-pred', 'product-gold-text', 'product-pred-text']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
            writer.writeheader()
            for i in range(len(product_pred)):
                writer.writerow({
                    'product-gold': product_gold[i], 'product-pred': product_pred[i],
                    'product-gold-text': id2label[product_gold[i]], 'product-pred-text': id2label[product_pred[i]]})

    submission_file_name = hazard_file_name
    if submission_file_name != product_file_name:
        submission_file_name = submission_file_name.split('.')[0] + '_' + product_file_name
    if not have_gold_labels:
        submission_file_name = 'validation_' + submission_file_name
    with open(f'submissions/{model_name2}/{submission_file_name}', 'w', newline='') as csvfile:
        fieldnames = ['hazard-category', 'product-category', 'hazard', 'product']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
        writer.writeheader()
        for i in range(len(hazard_pred)):
            writer.writerow({'hazard-category': 'other', 'product-category': 'other', 'hazard': hazard_pred_text[i], 'product': product_pred_text[i]})
    if have_gold_labels:
        f1 = compute_score(hazard_gold, product_gold, hazard_pred, product_pred)
        print(f'f1 vector: {f1}')
        acc = accuracy_score(hazard_gold, hazard_pred)
        print(f'acc hazard: {acc}')
        acc = accuracy_score(product_gold, product_pred)
        print(f'acc prod: {acc}')
        with open(f'results/{model_name2}/score_{hazard_file_name.split(".")[0]}.txt', 'w') as f:
            f.write(f'f1: {f1}')

if __name__ == '__main__':
    main()
