import csv
import random
import os

from collections import Counter

import numpy as np
import torch
import tqdm

from datasets import Dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader

from sklearn.model_selection import KFold

from textattack.augmentation import Augmenter, EmbeddingAugmenter
from textattack.constraints.semantics.sentence_encoders import SBERT
from textattack.transformations import CompositeTransformation, WordDeletion, WordInnerSwapRandom

from utils import read_clean_file, process_missing_entries, get_category_mappings


torch.manual_seed(0)
torch.backends.cudnn.benchmark = False

random.seed(0)
np.random.seed(0)


def main_train_unsupervised(model_name, train_sentences, output_model_name, batch_size=128):
    # Define your sentence transformer model using CLS pooling
    # model_name = "distilroberta-base"
    word_embedding_model = models.Transformer(model_name, max_seq_length=32)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

    # Define a list with sentences (1k - 100k sentences)
    # train_sentences = [
    #     "Your set of sentences",
    #     "Model will automatically add the noise",
    #     "And re-construct it",
    #     "You should provide at least 1k sentences",
    # ]

    # Convert train sentences to sentence pairs
    train_data = [InputExample(texts=[s, s]) for s in train_sentences]

    # DataLoader to batch your data
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

    # Use the denoising auto-encoder loss
    train_loss = losses.MultipleNegativesRankingLoss(model)

    # Call the fit method
    model.fit(
        train_objectives=[(train_dataloader, train_loss)], epochs=2, show_progress_bar=True
    )

    model.save(f"trained_models/simcse-{output_model_name}")


def make_labeled_examples(entries, sentences, eval_key, pos_example_multiplier=1, neutral_examples=1, negative_examples=1):
    sent1 = []
    sent2 = []
    labels = []
    cat2all, _ = get_category_mappings(eval_key)
    # neutral_examples = 1
    # negative_examples = 1
    for i, entry in enumerate(entries):
        source_sentence = entry[f'extracted_{eval_key}']
        for _ in range(pos_example_multiplier):
            sent1.append(source_sentence)
            sent2.append(sentences[i])
            labels.append(1)
        nr = 0
        # same category
        for other_type in cat2all[entry[f'{eval_key}-category']]:
            if other_type == entry[eval_key]:
                continue
            # different label in same category
            if nr >= neutral_examples:
                break
            nr += 1
            sent1.append(source_sentence)
            sent2.append(other_type)
            labels.append(-0.5)
        nr = 0
        for other_category in cat2all:
            if other_category == entry[f'{eval_key}-category']:
                continue
            # different category altogether
            # TODO: should also make sure that the cosine similarity between the categories is small or even negative
            for other_type in cat2all[other_category]:
                if nr >= negative_examples:
                    break
                nr += 1
                sent1.append(source_sentence)
                sent2.append(other_type)
                labels.append(-1)
    return sent1, sent2, labels


def make_resampled_dataset(entries, sentences, eval_key, model_name=''):
    # https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.RandomOverSampler.html
    gold_labels = list(set(entry[eval_key] for entry in entries))
    label2id = {gold_labels[k]: k for k in range(len(gold_labels))}
    id2label = {v: k.lower() for k, v in label2id.items()}
    X = []
    y = []
    for i, entry in enumerate(entries):
        X.append(i)
        y.append(label2id[entry[eval_key]])
    X = np.array(X).reshape(-1, 1)
    # from imblearn.over_sampling import RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    # from imblearn.over_sampling import SMOTE
    # ros = RandomOverSampler(random_state=0, sampling_strategy='minority')
    num_resamples_dict = {label2id[k]: 1 for k in gold_labels}

    new_entries = []
    new_sents = []

    max_examples = 3
    transformation = CompositeTransformation([WordDeletion(), WordInnerSwapRandom()])
    constraints = []
    # constraints = [SBERT()]
    # augmenter = Augmenter(
    #     transformation=transformation,
    #     # constraints=constraints,
    #     pct_words_to_swap=0.5,
    # )
    augmenter = EmbeddingAugmenter()
    all_labels = [entry[eval_key] for entry in entries]
    counts = Counter(all_labels)
    top_entries = [k[0] for k in sorted(counts.items(), key=lambda key: key[1], reverse=True) if k[1] > max_examples]
    num_resamples_dict = {label2id[k]: max_examples for k in top_entries}
    if os.path.isfile(f'data/aug/{eval_key}-{max_examples}.csv'):
        new_entries = read_clean_file(f'data/aug/{eval_key}-{max_examples}.csv')
        new_sents = [entry[eval_key] for entry in new_entries]
    else:
        for label in tqdm.tqdm(all_labels):
            if label not in top_entries:
                # num_resamples_dict[label] = counts[label]
                augmenter.transformations_per_example = max_examples - counts[label]
                selected_entries = [entry for entry in entries if entry[eval_key] == label]
                for selected_entry in selected_entries:
                    # print(f'calling with {selected_entry[f'extracted_{eval_key}']}')
                    aug_sents = augmenter.augment(selected_entry[f'extracted_{eval_key}'])
                    for i, new_sent in enumerate(aug_sents):
                        new_entries.append({eval_key: label, f'extracted_{eval_key}': new_sent, f'{eval_key}-category': selected_entry[f'{eval_key}-category']})
                        new_sents.append(label)
        with open(f'data/aug/{eval_key}-{max_examples}.csv', 'w', newline='') as csvfile:
            fieldnames = [eval_key, f'extracted_{eval_key}', f'{eval_key}-category']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
            writer.writeheader()
            for row in new_entries:
                writer.writerow(row)

    return new_entries, new_sents

    rus = RandomUnderSampler(random_state=0, sampling_strategy=num_resamples_dict)
    # ros = SMOTE(random_state=0)
    # X2, y2 = ros.fit_resample(X, y)
    X2, y2 = rus.fit_resample(X, y)

    for i in range(len(y2)):
        new_entries.append(entries[X2[i][0]])
        new_sents.append(id2label[y2[i]])
        # if i > 20:
        #     break
    # print(X.shape)
    # print(X2.shape)
    # print(len(y2))
    # print(new_entries[0:5])
    # print(new_sents[0:5])
    # print('....')
    # print(sentences[-15:])
    # print(new_sents[-15:])
    return new_entries, new_sents


def train_supervised(model_name, eval_key, sent1, sent2, labels, batch_size=32, eval_dataset=None):
    model = SentenceTransformer(model_name)
    dataset = Dataset.from_dict({
        "sentence1": sent1,
        "sentence2": sent2,
        "score": labels,
    })
    if eval_dataset:
        train_dataset = dataset
    else:
        ds = dataset.train_test_split(test_size=0.1, seed=0)
        train_dataset = ds['train']
        eval_dataset = ds['test']
    num_epochs = 1
    loss = losses.CoSENTLoss(model)
    # loss = losses.AnglELoss(model)
    args = SentenceTransformerTrainingArguments(
        f'checkpoints/cosent-pos2-{eval_key}',
        per_device_train_batch_size=batch_size,
        num_train_epochs=num_epochs,
        eval_strategy='steps',
        save_strategy='steps',
        save_total_limit = 1,
        load_best_model_at_end=True,
        warmup_ratio=0.2,
        learning_rate=2e-5,
        eval_steps=100,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        args=args,
    )
    trainer.train()
    model.save_pretrained(f"trained_models/cosent-3sample-minority-lr2-w2-pos2-neutr1-neg1-{num_epochs}epoch-{eval_key}-base")


def make_augmented_dataset(entries, sentences, eval_key):
    pass


def main_kf(model_name=''):
    product_file_name = 'parsed_data_f24_h17.csv'
    product_entries = read_clean_file(product_file_name)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    # transformation = CompositeTransformation([WordDeletion(), WordInnerSwapRandom()])
    # constraints = [SBERT(model_name=model_name)]
    # constraints = []
    # augmenter = Augmenter(
    #     transformation=transformation,
    #     # constraints=constraints,
    #     pct_words_to_swap=0.5,
    #     transformations_per_example=4
    # )
    # TODO: set this dynamically based on a Counter depending on class frequency
    # augmenter.transformations_per_example = n
    for fold, (train_index, val_index) in enumerate(kf.split(product_entries)):
        train_data = [product_entries[i] for i in train_index]
        train_sents = process_missing_entries(train_data, 'product')
        # TODO: cache aug data
        new_train_data, new_train_sents = make_resampled_dataset(train_data, train_sents, 'product', 'thenlper/gte-base')

        val_data = [product_entries[i] for i in val_index]
        val_sents = process_missing_entries(val_data, 'product')
        sent1, sent2, labels = make_labeled_examples(new_train_data, new_train_sents, 'product', pos_example_multiplier=2, neutral_examples=1, negative_examples=1)
        sent1_eval, sent2_eval, labels_eval = make_labeled_examples(val_data, val_sents, 'product', pos_example_multiplier=2, neutral_examples=1, negative_examples=1)
        eval_dataset = Dataset.from_dict({
            "sentence1": sent1_eval,
            "sentence2": sent2_eval,
            "score": labels_eval,
        })
        train_supervised('thenlper/gte-base', 'product', sent1, sent2, labels, batch_size=32, eval_dataset=eval_dataset)
        return


def main():
    # hazard_file_name = 'parsed_data_prompt_h14.csv'
    # hazard_entries = read_clean_file(hazard_file_name)
    # hazard_sentences = process_missing_entries(hazard_entries, 'hazard')
    # # main_train_unsupervised('sentence-transformers/paraphrase-MiniLM-L6-v2', hazard_sentences, 'hazard')
    # sent1, sent2, labels = make_labeled_examples(hazard_entries, hazard_sentences, 'hazard')
    # train_supervised('sentence-transformers/paraphrase-MiniLM-L6-v2', 'hazard', sent1, sent2, labels, batch_size=256)
    # product_file_name = 'parsed_data_prompt_f10.csv'
    product_file_name = 'parsed_data_f24_h17.csv'
    product_entries = read_clean_file(product_file_name)
    product_sentences = process_missing_entries(product_entries, 'product')
    # main_train_unsupervised('thenlper/gte-base', product_sentences, 'product', batch_size=64)
    product_entries, product_sents = make_resampled_dataset(product_entries, product_sentences, 'product', 'thenlper/gte-base')
    return
    sent1, sent2, labels = make_labeled_examples(product_entries, product_sentences, 'product', pos_example_multiplier=2, neutral_examples=1, negative_examples=1)
    train_supervised('thenlper/gte-base', 'product', sent1, sent2, labels, batch_size=32)


if __name__ == "__main__":
    # main()
    main_kf()
