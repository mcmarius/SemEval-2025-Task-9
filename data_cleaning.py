import csv
import json

# import evaluate
import requests
import tqdm
# import unidecode

seed = 42

def read_file(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        entries = [row for row in reader]
        header = entries[0]
        entries = entries[1:]
        return header, entries

def make_clean_text(text):
    rows = text.split("\n")
    clean_rows = []
    # remove duplicate entries
    for row in rows:
        if row not in clean_rows:
            clean_rows.append(row.strip())
    return "\n".join(clean_rows)

def make_food_prompt(example):
    # prompt = f'Given the following text, extract the name of the food product. If the name of the food product is not specifically mentioned, respond with "missing". The text: "{example}". Format the answer as follows and do not include anything else: Product: <product>'
    # prompt = f'Text: "{example}". Is there a commercial food product name specifically mentioned? Respond only with <name> or "missing".'
    prompt = f'Text: "{example}". Can you infer what specific food item is being described explicitly in this text? Respond only with the food item or "missing".' # prompt 10
    # prompt = f'{example}\nCan you infer what specific food item is being described explicitly in this text? Respond only with the food item or "missing".' # prompt 11
    # prompt = f'{example}\nCan you infer what food item is being described explicitly in this text? Respond only with the food item or "missing".' # prompt 12
    # prompt = f'{example}\nCan you determine what food item is being described explicitly in this text? Respond with the full food item description (verbatim) or respond with "missing" if the information is not present. Do not write any numbers or additional text.' # prompt 13
    # prompt = f'{example}\nCan you determine what product (food item) is being described explicitly in this text? Respond with the full product description (verbatim) or respond with "missing" if the information is not present. Do not write any numbers or additional text.' # prompt 14
    # prompt = f'Text: "{example}". Can you infer what specific product (food item) is being described explicitly in this text? Respond with the product (food item) verbatim or respond with "missing" if the information is not present.' # prompt 15
    # prompt = f"{example}. Extract the specific product (food item) that is being described or mentioned in the text or title provided. Respond with the product (food item) verbatim. Do not write any numbers or additional text." # prompt 16
    # prompt = f"{example}. Can you infer what specific food item is being described explicitly in this text? Respond only with the food item or 'missing'. If the product is a proper name, also give a brief explanation." # prompt 17
    # prompt = f"{example}. Can you infer what food item is being described explicitly in this text? Respond only with the food item and a description of up to 10 words or 'missing'." # prompt 18
    # prompt = f"{example}. Can you infer what specific food item is being described explicitly in this text? Respond only with the food item or 'missing'. If it is a brand name or an uncommon product, add a description of up to 10 words " # prompt 19
    # prompt = f"{example}. Can you infer what specific food item or product is being described explicitly in this text? Respond only with the food item. Respond with 'missing' if there is no mention. Respond with 'missing' if only a product category is given." # prompt 20
    # prompt = f"{example}. Extract the most specific food item or product mentioned in the text. Respond with 'missing' if there is no mention. Do not include anything else." # prompt 21
    return prompt

def make_hazard_prompt(example):
    # prompt = f'Text: "{example}". Is there a food hazard specifically mentioned? Extract the most specific occurrence. Respond only with <name> or "missing".'
    # prompt = f'Text: "{example}". Can you infer what specific food hazard is being described explicitly in this text? Respond only with the food hazard or "missing".' # prompt 10
    # prompt = f'{example}\nCan you infer what specific food hazard is being described explicitly in this text? Respond only with the food hazard or "missing".' # prompt 11
    # prompt = f'{example}\nCan you infer what food hazard is being described explicitly in this text? Respond only with the food hazard or "missing".' # prompt 12
    # prompt = f'{example}\nCan you infer what product hazards, defects or problems are being described explicitly in this text? Extract the most detailed occurrences with all necessary extra information. Respond only with the reported issues (details). Respond with "missing" if the information is not present. Do not write any additional text.' # prompt 13
    prompt = f'Text: "{example}". Can you infer what specific food hazard or problem is being described explicitly in this text? Respond only with the food hazard  verbatim and also verbatim description if present. Respond with "missing" only if no information is available.' # prompt 14
    return prompt


def make_request(prompt):
    # model = "Llama-3.2-3B-Instruct-Q6_K.gguf"
    model = "llama3.2"
    # model = "llama3.2-q6-k"
    # prompt = f'Given the following text, extract the food product and the related food hazard. The text: "{example}". Format the answer as follows and do not include anything else: ' + '{"food": "<food>", "hazard": "<hazard>"}'
    # prompt = f'Given the following text, extract the food product and the related food hazard. The text: "{example}". Format the answer as follows and do not include anything else: Product: <food>\nHazard: <hazard>'
    # prompt = f'Given the following text, extract the food product and the related food hazard. Do not split words. The hazard might be mentioned multiple times: extract the most specific mention. Ask yourself if the product is actually a food item and if the hazard is actually a hazard. If either the food or the hazard is not mentioned, respond with "missing" in that category. The text: "{example}". Format the answer as follows and do not include anything else: Product: <food>\nHazard: <hazard>'

    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 30,
        "temperature": 0,
        "seed": seed,
    }
    return request

def process_request(req_data):
    llm_server = "http://localhost:8080"
    # llm_server = "http://localhost:11434"
    while True:
        # print(f"Sending {req_data}")
        req = requests.post(f"{llm_server}/v1/chat/completions", json=req_data)
        try:
            raw_response = json.loads(req.text)
            raw_content = raw_response['choices'][0]["message"]["content"]
            # print(f"From json: {raw_content}")
            return raw_content, False
            # content = json.loads(raw_content)
            # food = raw_content
            # lines = raw_content.split("\n")
            # food = lines[0]
            # hazard = lines[1]
            # print(f"Got parsed content: {content}")
            # print(f"Got summary: {summary}\n\n")
            # out_row = row[0:6] + row[7:11]
            # out_row += [food] #, hazard]
            # out_entries.append(out_row)
        except (json.decoder.JSONDecodeError,):
            # aaa = bytes(req.text, encoding='utf8')
            # print(f"[DEBUG] Response: {req.text}")
            lines = req.text.split("\n")
            relevant_line = ""
            for line in lines:
                if "content" in line:
                    relevant_line = line.split('": "')[1].split('",')[0]
                    break
            try:
                relevant_line = bytes(relevant_line, "utf-8").decode("unicode_escape")
            except UnicodeDecodeError:
                pass
            relevant_line = ". ".join(relevant_line.split("\n")[0:2])
            # print(f"Manual parse: {relevant_line}")
            # num_errors += 1
            return relevant_line, True
            if num_errors % 100 == 0:
                print("[DEBUG] reached 100 errors")
            continue
        if req.status_code == 200:
            break
        # print(f"[WARNING] Got {req.status_code}. Retrying...")

def main_eval_loop(file_name, include_product=True, include_hazard=True, add_title=False):
    header, entries = read_file(file_name)
    out_header = header[0:6] + header[7:11]
    if include_product:
        out_header += ['extracted_food']
    if include_hazard:
        out_header += ['extracted_hazard']
    out_entries = []
    num_errors = 0
    print("data loaded")
    try:
        for i, row in enumerate(tqdm.tqdm(entries)):
            # if i < 60:
            #     continue
            clean_text = make_clean_text(row[6])
            if add_title:
                clean_text = f"{row[5]}. {clean_text}"
            out_row = row[0:6] + row[7:11]
            if include_product:
                prompt = make_food_prompt(clean_text)
                req_data = make_request(prompt)
                # print(f"made prompt with {clean_text}")
                food, err = process_request(req_data)
                if err:
                    num_errors += 1
                out_row += [food]
            if include_hazard:
                prompt = make_hazard_prompt(clean_text)
                req_data = make_request(prompt)
                hazard, err = process_request(req_data)
                if err:
                    num_errors += 1
                out_row += [hazard]
            out_entries.append(out_row)
            # if i > 100:
            #     break

    finally:
        print(f"Total errors: {num_errors}")
        # out_file = 'parsed_data_f21.csv'
        out_file = 'parsed_data_f10.csv'
        # out_file = 'parsed_data_h14.csv'
        # out_file = 'parsed_data_validation_h14_f10_q6.csv'
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"')
            writer.writerow(out_header)
            for row in out_entries:
                writer.writerow(row)


if __name__ == "__main__":
    # main_eval_loop('data/incidents_train.csv', include_hazard=False, add_title=True)
    main_eval_loop('data/incidents_train.csv', include_hazard=False, add_title=False)
    # main_eval_loop('data/incidents_train.csv', include_product=False, add_title=False)
    # main_eval_loop('data/incidents_validation.csv', add_title=False)
