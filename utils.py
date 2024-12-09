import csv
import string


def read_file(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar='"')
        entries = [row for row in reader]
        # header = entries[0]
        # entries = entries[1:]
        return entries


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


def read_clean_file(file_name):
    return read_file(file_name)


def process_missing_entries(entries, eval_key):
    parsed_entries = []
    for entry in entries:
        new_entry = entry[eval_key].lower().replace('\n', ' ').strip(string.punctuation).strip()
        if 'missing' in new_entry:
            new_entry = new_entry.replace('missing', '').strip(string.punctuation).strip()
            if new_entry == '':
                new_entry = entry['title']
        parsed_entries.append(new_entry)
    return parsed_entries


def get_category_mappings(eval_key):
    train_examples = read_file('data/incidents_train.csv')
    categories = list(set(example[f'{eval_key}-category'] for example in train_examples))
    # ids = list(set(example[eval_key] for example in train_examples))
    cat2all = {key: list(set(example[eval_key] for example in train_examples if example[f'{eval_key}-category'] == key)) for key in categories}
    all2cat = {}
    for example in train_examples:
        all2cat[example[eval_key]] = example[f'{eval_key}-category']
    return cat2all, all2cat
