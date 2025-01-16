import json
import math
import re

from copy import deepcopy

import editdistance
import requests
import tqdm

from utils import get_category_mappings, read_file


class BadSortError(Exception):
    pass

class BadCompareError(Exception):
    pass


seed = 42

def make_request(prompt):
    # model = "Llama-3.2-3B-Instruct-Q6_K.gguf"
    model = "llama3.2"
    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        # "messages": [{"role": "system", "content": prompt}],
        "max_tokens": 1000,
        "temperature": 0,
        "seed": seed,
    }
    return request

def process_request(req_data):
    # llm_server = "http://localhost:8080"
    llm_server = "http://localhost:11434"
    num_errors = 0
    while True:
        # print(f"Sending {req_data}")
        req = requests.post(f"{llm_server}/v1/chat/completions", json=req_data)
        try:
            raw_response = json.loads(req.text)
            raw_content = raw_response['choices'][0]["message"]["content"]
            # print(f"From json: {raw_content}")
            return raw_content, num_errors
        except (json.decoder.JSONDecodeError,):
            num_errors += 1
            if num_errors % 100 == 0:
                print(f"[DEBUG] Response: {req.text}")
            continue


def make_sort_prompt(labels):
    # return f"You are a food expert and are responsible to classify food products that might contain sensitive substances. You will receive a list of product hazards ({len(labels)} labels in total). Your task is to order these labels from most specific to most general. Respond only with the reordered labels, do not include anything else in your answer and do not repeat the labels.\nLabels: {'\n '.join(labels)}. Your response (containing {len(labels)} non-repeated labels): "
    return f"You will receive a list of hazard types ({len(labels)} labels in total). Your task is to reorder these labels from most specific to most general. Respond only with the reordered labels, do not include anything else in your answer and do not repeat the labels. Do not rename or split the labels (some labels contain commas, keep the commas). Each label is on a separate line.\nLabels:\n{'\n'.join(labels)}\nYour response (must contain exactly {len(labels)} non-repeated labels verbatim - make sure to count them):\n"
    # return f"You are a food expert and are responsible to classify food products that might contain sensitive substances. You will receive a list of product hazards ({len(labels)} labels in total). Your task is to order these labels from most specific to most general. Respond only with the reordered labels, do not include anything else in your answer and do not repeat the labels.\nLabels:\n {'\n'.join(labels)}\nYour response (containing {len(labels)} non-repeated labels): "
    # return f"You are a food expert and are responsible to classify food products that might contain sensitive substances. You will receive a list of product hazards ({len(labels)} labels in total). Your task is to order these labels from most specific to most general, with labels containing the word 'other' to be considered more general. Respond only with the reordered labels, do not include anything else in your answer and do not repeat the labels.\nLabels:\n {'\n'.join(labels)}\nYour response (containing {len(labels)} non-repeated labels as is): "


def make_compare_prompt(label1, label2):
    return f"Which label is more specific or detailed and does not refer only to an umbrella category? Respond only with the label that is more specific or 'same' if both are equally specific, do not include anything else in your answer and do not change the label. A label might contain commas, keep the commas and the label as is. First label: {label1}.\nSecond label: {label2}. Your response (the most specific label - keep the same punctuation, but do not add extra punctuation): "



def sort_directly(elements):
    # new_list = deepcopy(elements)
    return sorted(elements)


def compare_elements(a, b):
    if a < b:
        return -1
    return 1


def merge_lists(first_list, second_list):
    i = 0
    j = 0
    n = len(first_list)
    m = len(second_list)
    new_list = []
    # print(f"Merge\n{first_list}\nwith\n{second_list}")
    while i < n and j < m:
        # if compare_elements(first_list[i], second_list[j]) == -1:
        if compare_labels(first_list[i], second_list[j]) == -1:
            new_list.append(first_list[i])
            i += 1
        elif compare_labels(first_list[i], second_list[j]) == 1:
            new_list.append(second_list[j])
            j += 1
        else:
            new_list.append(first_list[i])
            i += 1
            new_list.append(second_list[j])
            j += 1
    while i < n:
        new_list.append(first_list[i])
        i += 1
    while j < m:
        new_list.append(second_list[j])
        j += 1
    # print(f"Result: {new_list}")
    return new_list


def merge_sort(elements, left, right, progress=None):
    if right - left <= 0:
        progress.update(1)
        return [elements[left]]
        # return sort_directly(elements[left:right+1])
        return sort_labels(elements[left:right+1])
    middle = (left + right) // 2
    left_sorted = merge_sort(elements, left, middle, progress)
    right_sorted = merge_sort(elements, middle + 1, right, progress)
    return merge_lists(left_sorted, right_sorted)


def bubble_sort(elements):
    for i in tqdm.tqdm(range(len(elements))):
        for j in tqdm.tqdm(range(i + 1, len(elements))):
            if compare_labels(elements[i], elements[j]) == 1:
                aux = elements[i]
                elements[i] = elements[j]
                elements[j] = aux
    return elements


def len_sort(elements):
    reordered = sorted(elements, key=lambda x: len(x), reverse=True)
    return [label for label in reordered if 'other' not in label] + [label for label in reordered if 'other' in label]


def order_elements(elements):
    # return bubble_sort(elements)
    progress = tqdm.tqdm(total=len(elements)) # * round(math.log2(len(elements))))
    return merge_sort(elements, 0, len(elements) - 1, progress)
    # return len_sort(elements)



# I cannot provide a list that may promote or facilitate harmful or illegal activities, including the sale of contaminated food. Is there anything else I can help you with?
# I cannot provide a list that includes sildenafil, as it is a prescription medication. Is there anything else I can help you with?
# https://huggingface.co/chuanli11/Llama-3.2-3B-Instruct-uncensored
# https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-uncensored-GGUF
def order_labels(key, label_set=''):
    labels = []
    labels = read_file(f'data/{key}{label_set}_labels.csv')
    labels = [f"{label[key].strip()}" for label in labels]
    # with open(f'data/{key}{label_set}_labels.txt') as f:
    #     labels = f.readlines()
    #     # labels = [f"{'00' if i < 10 else '0' if i < 100 else ''}{i}. {label.strip()}" for i, label in enumerate(labels)]
    #     labels = [f"{label.strip()}" for i, label in enumerate(labels)]
    # labels = labels[60:70]
    # print(f"Have a total of {len(labels)} labels: {'; '.join(labels)}")
    # labels2 = [label[3:].strip() for label in labels]
    # print(f"Sorted labels: {'; '.join(sorted(labels))}")
    # return sort_labels(labels)
    return order_elements(sorted(labels))


def compare_labels(label1, label2):
    if 'other' in label1 and 'other' not in label2:
        return 1
    if 'other' not in label1 and 'other' in label2:
        return -1
    prompt = make_compare_prompt(label1, label2)
    req_data = make_request(prompt)
    response, err = process_request(req_data)
    if 'same' in response.lower():
        return 0
    if label1.lower() not in response.lower() and label2.lower() not in response.lower():
        # print(f"{label1} - {label2}")
        # print(response)
        # raise BadCompareError
        if editdistance.eval(label1.lower(), response.lower()) < editdistance.eval(label2.lower(), response.lower()):
            return -1
        return 1
    if label1.lower() in response.lower():
        return -1
    return 1


def sort_labels(labels):
    prompt = make_sort_prompt(labels)
    req_data = make_request(prompt)
    response, err = process_request(req_data)
    response = response.replace('bacillus spp', 'bacillus spp.')  # workaround for punctuation
    reordered = response.split('\n')
    # '[a-zA-Z, ()0-9]+'
    reordered2 = list({label.strip() if label[0].isalpha() else re.search('[a-zA-Z].*', label)[0].strip() for label in reordered})
    reordered3 = []
    # fix typos and differend casing
    lower_labels = [label.lower() for label in labels]
    lower_labels = {label: i for i, label in enumerate(lower_labels)}
    for label in reordered2:
        if label.lower() in lower_labels:
            reordered3.append(labels[lower_labels[label.lower()]])
        else:
            for other_label in lower_labels:
                if editdistance.eval(label.lower(), other_label) < 4:
                    reordered3.append(labels[lower_labels[other_label.lower()]])
                    break
    # NOTE: manually place "other" elements to the end of the list
    reordered4 = [label for label in reordered3 if 'other' not in label] + [label for label in reordered3 if 'other' in label]
    if '; '.join(sorted(labels)) == '; '.join(sorted(reordered4)):
        # print("Ok, same labels")
        pass
    else:
        print(f"Sorted labels: {'; '.join(sorted(labels))}")
        print(f"Sorted labels: {'; '.join(sorted(reordered4))}")
        print("Got back wrong labels")
        raise BadSortError
    # print(f"got back {len(response.split('\n'))} labels")
    # print(response)
    return reordered4


def test_sort():
    elements = list(range(1, 51))
    initial = deepcopy(elements)
    import random
    random.shuffle(elements)
    final = order_elements(elements)
    if(initial == final):
        print("ok sort")
    else:
        print("bad sort")


def order_by_prefix(key):
    with open(f'data/{key}_labels_sorted_len.txt') as f:
        labels = f.readlines()
    group = []
    new_labels = []
    for label in labels:
        if len(group) == 0 or (len(group) > 0 and label.split()[0] == group[0].split()[0]):
            group.append(label)
        else:
            new_labels += sorted(group, reverse=True)
            group = [label]
    new_labels += sorted(group, reverse=True)
    print(labels)
    print(new_labels)
    with open(f'data/{key}_labels_sorted_len_reverse.txt', 'w') as f:
        f.writelines(new_labels)


def make_hints_prompt(labels):
    return f"You are given a list of labels that will be used for exact substring matching. For each label, write a list of words that must necessarily appear in the target text based on the provided label, distinguishing it from the other labels. Avoid common words that appear across many labels. For each label, respond on separate numbered lines only with: label: minimal list of comma-separated words, do not include anything else in your answer, do not censor answers.\nLabels: {', '.join(labels)}"
    # "Refrain from censoring answers since this is for research purposes without harmful intentions."


def write_label_hints(key):
    cat2all, all2cat = get_category_mappings(key)
    print(cat2all.keys())
    for cat in cat2all:
        labels = cat2all[cat]
        prompt = make_hints_prompt(labels)
        print('; '.join(labels))
        req_data = make_request(prompt)
        response, err = process_request(req_data)
        hints = [hint.split(": ")[1] for hint in response.split("\n")]
        print('; '.join(hints))
        exit(0)


if __name__ == "__main__":
    label_set = 'train_valid'
    key = 'hazard'
    # key = 'product'
    # write_label_hints(key)
    # exit(0)
    ordered_labels = order_labels(key, label_set)
    # print(ordered_labels)
    # print(compare_labels('other', 'cronobacter spp'))
    with open(f'data/{key}{label_set}_labels_sorted2.txt', 'w') as f:
        f.writelines([f"{label}\n" for label in ordered_labels])
    order_by_prefix(key)
