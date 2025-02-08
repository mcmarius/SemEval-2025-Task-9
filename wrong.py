import csv
import os
import re

from utils import read_file

def wrong_examples(in_file):
    key = 'hazard' if 'hazard' in in_file else 'product'
    #
    out_file = in_file.replace('results', 'wrong')
    results = read_file(in_file)
    if 'validation' in in_file:
    	text_data = 'data/incidents_validation.csv'
    elif 'test' in in_file:
    	text_data = 'data/incidents_test.csv'
    else:
        text_data = 'data/incidents_train.csv'
    #
    train_examples = read_file(text_data)
    #
    new_rows = []
    # nr = 0
    exclude_other = 'exclude_other'
    for i, row in enumerate(results):
        # if row[f'{key}-gold-text'] != row[f'{key}-pred-text']:
        if row[f'{key}-gold'] != row[f'{key}-pred']:
            if exclude_other:
                # if 'other' in row[f'{key}-gold-text'] and 'other' not in row[f'{key}-pred-text']:
                if 'other' in row[f'{key}-gold'] and 'other' not in row[f'{key}-pred']:
                    continue
            new_rows.append(row)
            new_rows[-1]['id'] = i
            new_rows[-1]['title'] = train_examples[i]['title']
            new_rows[-1]['text'] = train_examples[i]['text']
            # if float(re.search("[0-9].[0-9]+", row['probs'])[0]) < 0.5:
            #     nr += 1
    # print(f"nr: {nr}")
    #
    if exclude_other:
        out_file = out_file.split('.csv')[0] + '_exclude_other.csv'
    with open(out_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=list(new_rows[0].keys()), delimiter=',', quotechar='"')
        writer.writeheader()
        for row in new_rows:
            writer.writerow(row)
    #

# TODO: the 387, 535 examples do not actually contain the label: salmonella is only mentioned in another report
# 456, 1013, 1070, 1071, 1246, 1309, 1324 (?), 1325, 1663 (?), 1734, 1808, 2881, 2989, 3087 (?), 3118 (?), 3151, 3225, 3376, 3443, 3444, 4697 (?), 4946, 4953: wrong label
# 3454, 3495, 3507, 4123 (?), 4171 (?), 4279, 4286, 4584, 5062: ours is more specific
# investigate example 855, 928, 3229
# 874: în text zice că _nu_ e salmonella, gold label e salmonella
# 1180: does not want to expand e. coli, should do it in another step
# 1700: incomplete title
# 1907, 2184, 2507, 2940, 3601, 4049, 4112, 4809, 4865: multiple causes
# 1991, 2119, 2258, 3379: multiple causes, but the gold label is the real cause; idea for prompt: if multiple causes, choose the root cause
#   - 4268: reverse??
#   - 4394: in between???
# 2329, 2561: need better embeddings
# struggle w/ more abstract labels (e.g. compositional deviation): 4424
# 4433: we find cronobacter, but in text it says it is not tested
# 4953: information only available in an external (not) linked source

if __name__ == "__main__":
    model_str = 'sentence-transformers_paraphrase-MiniLM-L6-v2thenlper_gte-large'
    # file_str = 'hazard_parsed_data_f24_h17.csv'
    # file_str = 'product_parsed_data_f24_h17.csv'
    # file_str = 'hazard_reorder_simple_parsed_data_f24_h17.csv'
    file_str = 'hazard_reorder_simple_parsed_data_h25.csv'
    file_str = 'hazard_reorder_simple_parsed_data_h25_ollama_repro.csv'
    file_str = 'hazard_reorder_simple_parsed_data_h29_ollama_regex.csv'
    file_str = 'hazard_reorder_simple_parsed_data_h25_2_ollama_regex_more_words.csv'
    file_str = 'hazard_reorder_simple_parsed_data_h25_3_ollama_regex.csv'
    file_str = 'hazard_reorder_simple_parsed_data_h25_ollama_regex_test2.csv'
    file_str = 'hazard_parsed_data_h25_ollama_regex_test2.csv'
    file_str = 'product_parsed_data_f24_ollama_no_regex_test2.csv'
    # file_str = 'product_parsed_data_f36.csv'
    # file_str = 'product_parsed_data_f36.csv'
    # file_str = 'product_parsed_data_f24_ollama_no_regex_repro.csv'
    # file_str = 'product_parsed_data_f38_ollama_regex.csv'
    # file_str = 'product_parsed_data_f24_h17.csv'
    os.makedirs(f'wrong/{model_str}', exist_ok=True)
    wrong_examples(f'results/{model_str}/{file_str}')
