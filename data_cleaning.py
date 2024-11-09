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
    for row in rows:
        if row not in clean_rows:
            clean_rows.append(row)
    return "\n".join(clean_rows)

def make_food_prompt(example):
    # prompt = f'Given the following text, extract the name of the food product. If the name of the food product is not specifically mentioned, respond with "missing". The text: "{example}". Format the answer as follows and do not include anything else: Product: <product>'
    prompt = f'Text: "{example}". Is there a commercial food product name specifically mentioned? Respond only with <name> or "missing".'
    return prompt

def make_hazard_prompt(example):
    prompt = f'Text: "{example}". Is there a food hazard specifically mentioned? Extract the most specific occurrence. Respond only with <name> or "missing".'
    return prompt


def make_request(prompt):
    model = "Llama-3.2-3B-Instruct-Q6_K.gguf"
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

def process_request(req_data, num_errors):
    llm_server = "http://localhost:8080"
    while True:
        req = requests.post(f"{llm_server}/v1/chat/completions", json=req_data)
        try:
            raw_response = json.loads(req.text)
            raw_content = raw_response['choices'][0]["message"]["content"]
            print(f"From json: {raw_content}")
            return raw_content
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
            # print(f"Manual parse: {relevant_line}")
            return relevant_line
            num_errors += 1
            if num_errors % 100 == 0:
                print("[DEBUG] reached 100 errors")
            continue
        if req.status_code == 200:
            break
        # print(f"[WARNING] Got {req.status_code}. Retrying...")

def main_eval_loop(file_name):
    header, entries = read_file(file_name)
    out_header = header[0:6] + header[7:11] + ['extracted_food', 'extracted_hazard']
    out_entries = []
    num_errors = 0
    print("data loaded")
    for i, row in enumerate(tqdm.tqdm(entries)):
        # if i < 45:
        #     continue
        clean_text = make_clean_text(row[6])
        prompt = make_food_prompt(clean_text)
        req_data = make_request(prompt)
        # print(f"made prompt with {clean_text}")
        food = process_request(req_data, num_errors)
        prompt = make_hazard_prompt(clean_text)
        req_data = make_request(prompt)
        hazard = process_request(req_data, num_errors)
        out_row = row[0:6] + row[7:11]
        out_row += [food, hazard]
        out_entries.append(out_row)
        if i > 100:
            break
    print(f"Total errors: {num_errors}")

    with open('parsed_data.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quotechar='"')
        writer.writerow(out_header)
        for row in out_entries:
            writer.writerow(row)


if __name__ == "__main__":
    main_eval_loop('data/incidents_train.csv')
