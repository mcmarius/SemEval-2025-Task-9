import csv
import re
import string

from copy import deepcopy

def read_file(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        entries = [row for row in reader]
        # header = entries[0]
        # entries = entries[1:]
        return entries


def read_clean_file(file_name):
    return read_file(file_name)


def fix_file(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        entries = [row for row in reader]
        header = list(entries[0].keys())
    out_header = []
    for key in header:
        if key == 'extracted_food':
            out_header.append('extracted_product')
        else:
            out_header.append(key)
    with open(f'fixed_{file_name}', 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=out_header, delimiter=',', quotechar='"')
            writer.writeheader()
            for row in entries:
                product = row.pop('extracted_food')
                row['extracted_product'] = product
                writer.writerow(row)


# merge_files('data/incidents_train.csv', 'data/incidents_valid.csv', 'data/incidents_train_valid.csv')
def merge_files(file1, file2, new_filename):
    rows1 = read_file(file1)
    rows2 = read_file(file2)
    new_rows = rows1
    for i, row in enumerate(rows1):
        new_rows[i][''] = i
    for i, row in enumerate(rows2):
        new_row = row
        new_row[''] = i + len(rows1) + 1
        new_rows.append(new_row)
    with open(new_filename, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=new_rows[0].keys(), delimiter=',', quotechar='"')
            writer.writeheader()
            for row in new_rows:
                writer.writerow(row)


# write_label_file('hazard', 'train_valid')
# write_label_file('product', 'train_valid')
def write_label_file(gold_key, label_set=''):
    entries = read_file('data/incidents_train_valid.csv')
    if label_set:
        gold_labels_file = f'data/{gold_key}{label_set}_labels.csv'
    else:
        gold_labels_file = f'data/{gold_key}_labels.csv'
    gold_labels = list(set(entry[gold_key] for entry in entries))
    label2id = {gold_labels[k]: k for k in range(len(gold_labels))}
    with open(gold_labels_file, 'w', newline='') as csvfile:
        fieldnames = [gold_key, 'id']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames, delimiter=',', quotechar='"')
        writer.writeheader()
        for line in gold_labels:
            writer.writerow({gold_key: line, 'id': label2id[line]})


def process_missing_entries(entries, eval_key):
    parsed_entries = []
    for entry in entries:
        new_entry = entry[eval_key].lower().replace('\n', ' ').strip(string.punctuation).strip()
        if 'missing' in new_entry:
            new_entry = new_entry.replace('missing', '').strip(string.punctuation).strip()
            if new_entry == '':
                new_entry = entry['title']
            # elif 'hazard' in eval_key:
            #     new_entry += entry['title']
        # else:
        #     if 'hazard' in eval_key:
        #         new_entry += entry['title']
        parsed_entries.append(new_entry)
    return parsed_entries


def get_category_mappings(eval_key, data_file='data/incidents_train.csv'):
    train_examples = read_file(data_file)
    categories = list(set(example[f'{eval_key}-category'] for example in train_examples))
    # ids = list(set(example[eval_key] for example in train_examples))
    cat2all = {key: list(set(example[eval_key] for example in train_examples if example[f'{eval_key}-category'] == key)) for key in categories}
    all2cat = {}
    for example in train_examples:
        all2cat[example[eval_key]] = example[f'{eval_key}-category']
    return cat2all, all2cat


def make_unique_list(elements):
    # we do not use a set because we want to preserve the initial order
    unique_rows = []
    for row in elements:
        if row not in unique_rows:
            unique_rows.append(row.strip())
    return unique_rows



# [r"What are the defects\?(.+(?=What are the hazards\?)[^\.])?", ["What should consumers do", "Where the product"]],
hazard_regexes = [
    [r"WASHINGTON.{,1550}(?#join lines)", []], # this is kind of hard-coded (example 1601)
    # [r"WASHINGTON.{,1550}(?#join lines ignore boxes containing)", []], # this is kind of hard-coded (example 1601)
    [r"FOR IMMEDIATE RELEASE.{,1550}(?#join lines)", []],
    [r"((?:Problem|Description): .*)", []],
    [r"(What are the defects\?.+)", ["What should consumers do", "Where the product"]],
    [r"^.{150,}$", []], # was 250; was 200
    [r"(Problem/Reason for Recall:.+How/When Discovered:.+)Establishment(?#join lines)", []],
]
product_regexes = [
    [r"Date published.*Product description(?#skip)", []],
    [r"WASHINGTON.{,600}(?#join lines)", []],
    [r"Product\(s\) Recalled:.*(?#join lines)", []],
    [r"^Product: (.*)$", []],
    [r"^.{150,}$", []],
    [r"^.*recall.*$", []],
    # [r"(Problem/Reason for Recall:.+How/When Discovered:.+)Establishment(?#join lines)", []],
]


def apply_regex_rules(rows, key):
    if key == 'hazard':
        regexes = hazard_regexes
    else:
        regexes = product_regexes
    result = []
    match_count = 0
    MAX_CHARS = 1550
    for regex, splits in regexes:
        working_rows = deepcopy(rows)
        # if "ignore boxes containing" in regex:
        #     working_rows = [row for row in working_rows if "boxes containing" not in row]
        if "join lines" in regex:
            working_rows = ['. '.join(working_rows)]
        for row in working_rows:
            result_match = re.findall(regex, row)
            if result_match:
                if "skip" in regex:
                    return []
                result_str = result_match[0]
                for split in splits:
                    result_str = result_str.split(split)[0]
                # print(row)
                # print(regex)
                # cut off extra long lines
                result.append(result_str[0:MAX_CHARS].strip())
                match_count += 1
                # break
            # print(result)
            # print("--------")
        if match_count >= 1:
            return result
    for row in rows:
        result.append(row[0:MAX_CHARS])
    return rows



def make_clean_text(text, key, regex_filter=True):
    MAX_PROMPT_CHARS = 3000
    rows = text.split("\n")
    # remove duplicate entries
    unique_rows = make_unique_list(rows)
    # check if we can rely on simple regex rules
    if regex_filter:
        unique_rows = apply_regex_rules(unique_rows, key)
    clean_rows = unique_rows
    # clean html tags
    clean_rows = []
    # print(unique_rows)
    for row in unique_rows:
        if re.findall(r'<.+>', row):
            matches = re.findall(r'>([A-Z][a-z]*+)<', row)
            if matches:
                clean_rows.append(matches[0])
        else:
            if re.findall(r'[A-Z][a-z]+', row):
                clean_rows.append(row)
    if not clean_rows:
        clean_rows = unique_rows
    return "\n".join(make_unique_list(clean_rows))[0:MAX_PROMPT_CHARS]
