import random

import numpy as np
import torch

from datasets import Dataset

from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, SentenceTransformerTrainingArguments, InputExample
from sentence_transformers import models, losses
from torch.utils.data import DataLoader

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


def make_labeled_examples(entries, sentences, eval_key, pos_example_multiplier=1):
    sent1 = []
    sent2 = []
    labels = []
    cat2all, _ = get_category_mappings(eval_key)
    neutral_examples = 1
    negative_examples = 1
    for i, entry in enumerate(entries):
        for _ in range(pos_example_multiplier):
            sent1.append(sentences[i])
            sent2.append(entry[eval_key])
            labels.append(1)
        nr = 0
        for other_type in cat2all[entry[f'{eval_key}-category']]:
            if other_type == entry[eval_key]:
                continue
            if nr >= neutral_examples:
                break
            nr += 1
            sent1.append(sentences[i])
            sent2.append(other_type)
            labels.append(-0.5)
        nr = 0
        for other_category in cat2all:
            if other_category == entry[f'{eval_key}-category']:
                continue
            for other_type in cat2all[other_category]:
                if nr >= negative_examples:
                    break
                nr += 1
                sent1.append(sentences[i])
                sent2.append(other_type)
                labels.append(-1)
    return sent1, sent2, labels


def make_resampled_dataset(entries, sentences, eval_key):
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
    from imblearn.over_sampling import RandomOverSampler
    # from imblearn.over_sampling import SMOTE
    ros = RandomOverSampler(random_state=0, sampling_strategy='minority')
    # ros = SMOTE(random_state=0)
    X2, y2 = ros.fit_resample(X, y)
    new_entries = []
    new_sents = []
    for i in range(len(y2)):
        new_entries.append(entries[X2[i][0]])
        new_sents.append(id2label[y2[i]])
        if i > 20:
            break
    # print(X.shape)
    # print(X2.shape)
    # print(len(y2))
    # print(new_entries[0:5])
    # print('....')
    # print(sentences[-15:])
    # print(new_sents[-15:])


def train_supervised(model_name, eval_key, sent1, sent2, labels, batch_size=256):
    model = SentenceTransformer(model_name)
    dataset = Dataset.from_dict({
        "sentence1": sent1,
        "sentence2": sent2,
        "score": labels,
    })
    ds = dataset.train_test_split(test_size=0.1, seed=0)
    train_dataset = ds['train']
    eval_dataset = ds['test']
    num_epochs = 1
    # loss = losses.CoSENTLoss(model)
    loss = losses.AnglELoss(model)
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
        eval_steps=50,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        loss=loss,
        args=args,
    )
    trainer.train()
    model.save_pretrained(f"trained_models/angloss-lr2-w2-pos4-neutr1-neg1-{num_epochs}epoch-{eval_key}-base")



def main():
    # hazard_file_name = 'parsed_data_prompt_h14.csv'
    # hazard_entries = read_clean_file(hazard_file_name)
    # hazard_sentences = process_missing_entries(hazard_entries, 'hazard')
    # # main_train_unsupervised('sentence-transformers/paraphrase-MiniLM-L6-v2', hazard_sentences, 'hazard')
    # sent1, sent2, labels = make_labeled_examples(hazard_entries, hazard_sentences, 'hazard')
    # train_supervised('sentence-transformers/paraphrase-MiniLM-L6-v2', 'hazard', sent1, sent2, labels, batch_size=256)
    product_file_name = 'parsed_data_prompt_f10.csv'
    product_entries = read_clean_file(product_file_name)
    product_sentences = process_missing_entries(product_entries, 'product')
    # main_train_unsupervised('thenlper/gte-base', product_sentences, 'product', batch_size=64)
    make_resampled_dataset(product_entries, product_sentences, 'product')
    # sent1, sent2, labels = make_labeled_examples(product_entries, product_sentences, 'product', pos_example_multiplier=4)
    # train_supervised('thenlper/gte-base', 'product', sent1, sent2, labels, batch_size=128)


if __name__ == "__main__":
    main()
