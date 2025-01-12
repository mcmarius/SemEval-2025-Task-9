import csv
import json
import re

# import evaluate
import requests
import tqdm
# import unidecode

from copy import deepcopy

from utils import make_clean_text

seed = 42

def read_file(file_name):
    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        entries = [row for row in reader]
        header = entries[0]
        entries = entries[1:]
        return header, entries



def make_food_prompt(example, title=""):
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
    prompt = f"Text: {title}. {example}.\nExtract all sentences and phrases that contain mentions of commercial products, food, beverages, or supplements. Write only the verbatim sentences and phrases every time they are mentioned. Do not include any other text." # prompt 22
    prompt = f"Text: {title}. {example}.\nExtract all mentions of commercial products, food, beverages, or supplements as a comma-separated list. Respond only with the verbatim items separated by comma, ordered from the most specific to generic. Do not include any other text." # prompt 23

    prompt = f"Article: {title}\n{example}.\n\nYou are given an article about food-incident reports. Your task is to find what product is described and extract it as it is found in the text, followed by a brief description. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product>. <product description>.<end of your response>" # prompt 24
    return prompt
    # prompt = f"Title: {title}\nArticle: {example}.\n\nYou are given an article about food-incident reports. Your task is to find what products are described and extract them as they are found in the text, followed by a brief description for each product. Do not include any numbers. Write the product in the title: <product in title>. Then, for each product, respond in the following format: <product>. <product dictionary definition>.<end of your response>. Do not respond with anything else." # prompt 25
    # prompt = f"Title: {title}\nArticle: {example}.\n\nYou are given an article about food-incident reports. Your task is to find what commercial products, foods, supplements or desserts are described and extract them as they are found in the text, followed by a brief description for each product. Do not include any numbers. Write at most 5 products found in the title and text in thr following format: <product 1, product 2>. Then, for each product, respond in the following format: <product>. <product dictionary definition>.<end of your response>. Do not respond with anything else. Make sure to extract the product and not the ingredients." # prompt 26
    # prompt = f"Title: {title}\nArticle: {example}.\n\nYou are given an article about food-incident reports. Your task is to extract commercial products, foods, supplements or desserts are written, with all words found in the article, but make sure not to confuse them with the ingredients, which might also be mentioned. Do not include any numbers, organizations or agencies. For each product, respond in the following format, ordered by most general to specific: <product sentence with details maximum 5 words>. <product dictionary definition>.<end of your response>. Do not respond with anything else." # prompt 27
    # prompt = f"Article: {title}.\n{example}.\n\nYou are given an article about food-incident reports. Your task is to extract commercial products, foods, supplements or desserts are written, with all words found in the article, but make sure not to confuse them with the ingredients, which might also be mentioned. Do not include any numbers, store names, organizations or agencies. For each product, respond in the following format, ordered by most specific to more general: <product sentence with specific details maximum 3 words>. <product dictionary definition>.<end of your response>. Do not respond with anything else." # prompt 28
    # prompt = f"Article: {title}.\n{example}.\n\nYou are given an article about food-incident reports. Your task is to extract commercial products, foods, supplements or desserts are written, with all words found in the article, but make sure not to confuse them with the ingredients, contaminants, substances, chemicals or hazards, which might also be mentioned. Do not include any numbers, store names, organizations or agencies. For each product, respond in the following format, ordered by most specific to more general: <product sentence with specific details maximum 3 words>. <product dictionary definition>.<end of your response>. Do not respond with repeated lines, notes, 'here are the extracted...' or anything else." # prompt 29
    # prompt = f"Text: {title}.\n{example}.\n\nRewrite the previous text in the following format for each product:- <product> <product dictionary definition><end of your response>. Do this for each product mentioned. Do not include ingredients, company names, contaminants, substances, chemicals, hazards, numbers, instructions, countries, diseases or anything else, which might also be mentioned. Do not include anything else in your answer." # prompt 30
    # prompt = f"Text: {title}.\n{example}.\n\nRemove all content from the text that mentions anything else besides a product being recalled. Do not include sentences containing problems, hazards, contaminants, steps for consumers, return instructions, dates, best before, batch details, quantities, categorization, company names, shop names. Respond directly with the product(s) name(s), do not include phrases like 'Here is the text with...'." # prompt 31
    #     meaning in 10 words max
    prompt = f"Article: {title}\n{example}.\n\nYou are given an article about food-incident reports. Your task is to find what product is described and extract it as it is found in the text, followed by a brief description. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product>. <product description>.<end of your response>" # prompt 32
    prompt = f"Article: {title}\n{example}.\n\nYou are given an article about food-incident reports. Your task is to find what product is described and extract it as it is found in the text, followed by the most 1-2 common synonyms. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product>  <synonyms>" # prompt 33
    #     Ignore the food incident or hazard if mentioned, focus on the product.
    # prompt = f"Article about food: {title}\n{example}.\n\nFind what product(s) are described or mentioned. Give one or two synonym terms. Give a short definition. Do not include any numbers. Respond in the following format and do not respond with anything else:\nfull product name and description. 1 or 2 synonyms. short definition of product" # prompt 34
    # prompt = f"Article about food: {title}\n{example}.\n\nFind what product(s) are described or mentioned. Give one or two synonyms terms. Determine the product category. Do not include any numbers. Respond in the following format and do not respond with anything else:\nfull product name and description. 1 or 2 synonyms. product category" # prompt 35
    prompt = f"Article about food: {title}\n{example}.\n\nFind what product(s) are described or mentioned. Give two synonyms in simple words. Determine the product category. Give a short definition. Do not include any numbers. Do not include food risks, recalls, incidents or hazard. Focus only on finding the product. Respond in the following format and do not respond with anything else:\nfull product name and description verbatim. two synonyms. product category. short definition of product, ten words max" # prompt 36
    # prompt = f"Article about food: {title}\n{example}.\n\nFind what product(s) are described or mentioned. Give one or two synonyms. Give a short definition. Do not include any numbers. Respond in the following format and do not respond with anything else:\nfull product name and description. 1 or 2 synonyms. short definition of product 10 words max" # prompt 37

    prompt = f"Article: {title}\n{example}.\n\nYou are given an article about food-incident reports. Your task is to find what product is described and extract it as it is found in the text with all characteristics (if present), followed by a brief description with synonyms. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description with synonyms>" # prompt 38: prompt 24 + synonyms

    # prompt = f"Article: {title}\n{example}\n\nThis is an article about food-incident reports. Your task is to find what product is described with all characteristics (if present), then write a brief description and use synonyms. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description with synonyms>" # prompt 39 old1
    # prompt = f"Article: {title}\n{example}\n\nThis is an article about food-incident reports. Your task is to find what product is described with all characteristics (if present), then write a brief description and use synonyms. Make sure the definition and synonyms do not change the original meaning   . Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description with synonyms>" # prompt 39 old2
    # prompt = f"Article: {title}\n{example}\n\nThis is an article about incident reports. Your task is to find what product is described with all characteristics (if present), then write a brief description and use synonyms. Make sure the definition and synonyms do not change the original meaning. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description with synonyms>" # prompt 39 old3
    # prompt = f"Article: {title}\n{example}\n\nYour task is to find what product is described with all characteristics (if present), then write a brief description and use synonyms. Make sure the definition and synonyms do not change the original meaning. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description with synonyms>" # prompt 39 old4
    # prompt = f"Text: {title}\n{example}\n\nYour task is to find what product is described with all characteristics (if present), then write a brief description and use synonyms that preserve the exact meaning. Make sure the definition and synonyms do not change the original meaning. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description with synonyms>" # prompt 39 old5
    # prompt = f"Text: {title}\n{example}\n\nYour task is to find what product is described with all characteristics (if present), write a brief description and then a few synonyms that preserve the exact meaning. Make sure the definition and synonyms do not change the original meaning. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description>. <synonyms>" # prompt 39 old6
    prompt = f"Text: {title}\n{example}\n\nYour task is to find what product is described with all characteristics (if present), write a brief description and then a list of related terms that preserve the exact meaning. Make sure the definition and synonyms do not change the original meaning. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description>" # prompt 39 old7 37.99 first50 27/23; 32.27 first100 51/49
    # prompt = f"Text: {title}\n{example}\n\nYour task is to find what product is described with all characteristics (if present), then write a brief description. Include a list of related terms that preserve the exact meaning if the product does not use common words. Make sure the definition and terms do not change the original meaning. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description>" # prompt 39 old8 31.48 first50 23/27
    # prompt = f"Text: {title}\n{example}\n\nYour task is to find what product is described with all characteristics (if present). Write a brief description if the product does not use common words. Make sure the definition does not change the original meaning. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description>" # prompt 39 old9 30.80 first50 27/23
    # prompt = f"Text: {title}\n{example}\n\nFind what product is described with all characteristics (if present) as is, verbatim. Write a brief description if the product does not use basic, common words and make sure this definition does not change the original meaning. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product with characteristics>. <product description>" # prompt 39 old10 27.94 25/25
    #
    # prompt = f"Text: {title}\n{example}.\n\nFind what product is described, with all characteristics (if present), followed by a brief description that includes alternative and semantically equivalent words (if any). Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product>. <product description>" # prompt 40 old1
    # prompt = f"Text: {title}\n{example}.\n\nFind what product is described, with all characteristics and properties, followed by a brief description that includes alternative and semantically equivalent words (if any). Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product>. <product description>" # prompt 40 old2
    prompt = f"Text: {title}\n{example}.\n\nFind what product is described, with all characteristics and properties verbatim, followed by a brief description that includes alternative and semantically equivalent terms. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product>. <product description>" # prompt 40 old3 36.04 first50 28/22; 30.18 first100 49/51
    prompt = f"Text: {title}\n{example}.\n\nFind what product is described, with all characteristics verbatim as in the text, followed by alternative and semantically equivalent terms. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product> - <alternative equivalent terms>" # prompt 41 old2 bad
    prompt = f"Text: {title}\n{example}.\n\nExtract the product described as is in the text with all characteristics mentioned, followed by alternative and semantically equivalent terms that do not alter the meaning or add any details. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<product found in text> - <alternative equivalent terms>" # prompt 41 old3 bad
    prompt = f"Text: {title}\n{example}.\n\nExtract the product found in the text, keeping all words as is. Include a short definition. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<full product text> - <short definition>" # prompt 41 old4
    prompt = f"Text: {title}\n{example}.\n\nExtract the product found in the text, keeping all words as is. Include a short definition only for foreign and uncommon products. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<full product text> - <short definition>" # prompt 41 old5
    prompt = f"Text: {title}\n{example}.\n\nExtract the product found in the text, keeping all words as is. Include a short definition only for foreign products. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<full product text>< - short definition if foreign product>" # prompt 41 old6
    prompt = f"Text: {title}\n{example}.\n\nExtract the product found in the text, keeping all words as is. If the product consists only of proper names, also provide a short definition. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<full product text>< - short definition only if needed>" # prompt 41
    # prompt = f"Text: {title}\n{example}.\n\nExtract the product found in the text, keeping all words as is. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<full product text> <product name common usage>" # prompt 41 bad
    prompt = f"Text: {title}\n{example}.\n\nExtract the product found in the text, keeping all words as is. Determine if the product has a common name. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<full product text>. <common/uncommon>" # prompt 41 old8 35.66 top100 54/46; 40.12 full, but they seem reasonable
    prompt = f"Extract the product name(s) mentioned or described in the text below. Also write a few keywords for the product. Do not include any numbers. Respond in the following format and do not respond with anything else: <product text> <keywords>\nText: {title} {example}" # prompt 42 old1
    prompt = f"Extract the product name(s) mentioned or described in the text below. Also write a few keywords for the product. Make sure the keywords refer to the product, not the ingredients. Do not include any numbers. Respond in the following format and do not respond with anything else: <product text> <keywords>\nText: {title} {example}" # prompt 42 old2
    prompt = f"Extract the product name(s) mentioned or described in the text below. Also write a few keywords for the product. Make sure the keywords refer only to the product, not the ingredients. Do not include any numbers. Respond in the following format and do not respond with anything else: <product text> <keywords>\nText: {title} {example}" # prompt 42 old3
    return prompt

def make_hazard_prompt(example, title=""):
    # prompt = f'Text: "{example}". Is there a food hazard specifically mentioned? Extract the most specific occurrence. Respond only with <name> or "missing".'
    # prompt = f'Text: "{example}". Can you infer what specific food hazard is being described explicitly in this text? Respond only with the food hazard or "missing".' # prompt 10
    # prompt = f'{example}\nCan you infer what specific food hazard is being described explicitly in this text? Respond only with the food hazard or "missing".' # prompt 11
    # prompt = f'{example}\nCan you infer what food hazard is being described explicitly in this text? Respond only with the food hazard or "missing".' # prompt 12
    # prompt = f'{example}\nCan you infer what product hazards, defects or problems are being described explicitly in this text? Extract the most detailed occurrences with all necessary extra information. Respond only with the reported issues (details). Respond with "missing" if the information is not present. Do not write any additional text.' # prompt 13
    prompt = f'Text: "{example}". Can you infer what specific food hazard or problem is being described explicitly in this text? Respond only with the food hazard  verbatim and also verbatim description if present. Respond with "missing" only if no information is available.' # prompt 14
    prompt = f"Text: {title}. {example}.\nExtract ALL sentences and phrases that contain mentions of hazards or problems related to commercial products, food or health. Respond only with the verbatim sentences and phrases every time they are mentioned. Do not include any other text." # prompt 15
    prompt = f"Text: {title}. {example}.\nExtract all mentions of hazards or problems related to commercial products, food or health as a comma-separated list. Respond only with the verbatim items separated by comma, ordered from most specific to generic. Do not include any other text." # prompt 16

    prompt = f"Article: {title}\n{example}.\n\nYou are given an article about food-incident reports. Your task is to extract the problem(s) with the product as briefly as possible, preserving the words in the article. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<problem>. <problem description>.<end of your response>" # prompt 17
    # prompt = f"Article: {title}\n{example}.\n\nYou are given an article about food-incident reports. Your task is to extract the problem(s) description(s) with the product as briefly as possible, preserving the words in the article. Do not include any numbers. Respond in the following format and do not respond with anything else:\n<problem description><: details if present>.<end of your response>" # prompt 18
    prompt = f"Article: {title}\n{example}.\n\nYou are given a snippet from an article about food-incident reports. Your task is to extract the most specific problem description with the product in maximum 1-7 words. Do not use filler words such as food, product, hazard, problem. Add a short description of up to 5-10 words. Respond in the following format: <problem in 1-7 words max>. <description of 10 words max> (optional)<end of your answer>. Preserve the words in the article. Do not include any numbers. Do not mention the product. Do not mention the product name. Do not respond with anything else." # prompt 19 old
    #
    prompt = f"Article: {title}\n{example}.\n\nYou are given an article about food-incident reports. Extract the most specific problem with the product in maximum 1-5 words. Use only nouns and adjectives. Be as brief as possible. You may add a description of up to 5-10 words. You must follow all of the following restrictions: Preserve the words in the article. Do not include any numbers. Do not mention the product. Do not mention the food or product name. Do not use filler words such as food, product, hazard, problem. Respond in the following format: <problem in 1-5 words max>. <description of 10 words max> (optional)<end of your answer>. Do not respond with anything else." # prompt 20 old
    prompt = f"Article: {title}\n{example}.\n\nYou are given an article about food-incident reports. Extract the most specific problem with the product in maximum 1-10 words. Use only nouns and adjectives. Be as brief as possible. You must follow all of the following restrictions: Preserve the words in the article. Do not include any numbers. Do not mention the product. Do not mention the food or product name. Do not use filler words such as food, product, hazard, problem. Respond in the following format: <problem in 1-10 words><end of your answer>. Do not respond with anything else." # prompt 20
    #     Only include words directly related to product problems or hazards.
    #     Make sure to keep essential details from descriptions (all of it if it is short)
    prompt = f"Text:\n{title}\n{example}.\n\nThe above text is about food product incidents. Extract the most specific 3-15 words that describe the problem with the product. Copy the words from the given text. Do not include any numbers. Do not mention the product. Do not mention the food or product name. Respond in the following format: <problem in 3 to 15 words><end of your answer> Do not respond with anything else." # prompt 21
    #
    # this one sometimes writes haikus
    prompt = f"Text:\n{title}\n{example}.\n\nExtract the most 2 to 8 specific words that describe the problem with the product (might be almost all of them for short texts). Copy the words from the given text. Do not include any numbers. Do not mention the product. Do not mention the food or product name. Respond in the following format: <2 to 8 words><end of your answer> Do not respond with anything else." # prompt 22
    # return prompt
    #
    if "Description: " in example or len(example) < 100:
        max_words = 3  # was 3
    elif len(example) < 200:
        max_words = 5  # was 5
    elif len(example) < 500:
        max_words = 6  # was 6
    elif len(example) < 1000:
        max_words = 7 # was 7
    else:
        max_words = 15 # was 8
    # print(f"len: {len(example)}")
    prompt = f"Text about food hazards:\n{title}; {example}.\n\nExtract at most {max_words} specific words that describe the product hazard (almost all for short texts). Copy the words from the given text, no new words. Keep scientific terms. Do not mention the food or product name. Respond like this on one line: <phrase of {max_words} words max> Do not respond with anything else." # prompt 23
    #
    prompt = f"Text about food hazards:\n{title}; {example}.\nExtract the product hazard as detailed as possible. Include scientific terms. Copy the words from the given text, no new words. Do not mention the food or product name. Respond like this on one line: <exact hazard problem detailed in {max_words} words> Do not respond with anything else." # prompt 24
    prompt = f"Article about food-incident reports: {title}\n{example}.\n\nYour task is to extract the problem(s) with the product as briefly as possible, preserving the words in the article. Keep scientific terms. Respond in the following format and do not respond with anything else:\n<problem>. <problem description in {max_words} words>.<end of your response>" # prompt 25
    prompt = f"Article about food-incident reports: {title}\n{example}.\n\nYour task is to extract the problem(s) with the product as briefly as possible, preserving the words in the article. Keep scientific terms. Respond in the following format and do not respond with anything else:\n<problem>. <problem description in {max_words} words>." # prompt 25-4; 3-5-6-7-15w
    return prompt
    if "Description: " in example or len(example) < 100:
        max_words = 5  # was 3
    elif len(example) < 200:
        max_words = 10  # was 5
    elif len(example) < 500:
        max_words = 12  # was 6
    elif len(example) < 1000:
        max_words = 14 # was 7
    else:
        max_words = 16 # was 8
    prompt = f"Article about food-incident reports: {title}\n{example}.\n\nYour task is to extract the problem(s) with the product as briefly as possible, preserving the words in the article. Keep scientific terms. Respond in the following format and do not respond with anything else:\n<problem>. <problem description in {max_words} words>." # prompt 25-2
    prompt = f"Article about food-incident reports: {title}\n{example}.\n\nYour task is to extract the problem with the product as briefly as possible, preserving the words in the article with all details. Keep scientific terms. Respond in the following format and do not respond with anything else:\n<problem>. <problem description>." # prompt 25-3
    return prompt
    # prompt = f"Article about food-incident reports: {title}\n{example}.\n\nYour task is to extract the problem(s) with the product as briefly as possible, preserving the words in the article, expanding all abbreviations to their full form. Keep scientific terms and expand abbreviations. Do not respond with one letter abbreviations. Respond in the following format and do not respond with anything else:\n<problem>. <problem description in {max_words} words>.<end of your response>" # prompt 26
    prompt = f"Article about food-incident reports: {title}\n{example}.\n\nYour task is to extract the problem(s) with the product as briefly as possible, preserving the words in the article. Keep scientific terms. Give full names to abbreviations of scientific terms, even if typically used contracted. Do not give notes. Respond in the following format and do not respond with anything else:\n<problem>. <problem description in {max_words} words>" # prompt 27
    prompt = f"Article about food-incident reports: {title}\n{example}.\n\nYour task is to extract the problem(s), risks, hazards with the product as briefly as possible, preserving the words in the article and essential details. Keep scientific terms. Respond in the following format and do not respond with anything else:\n<problem>. <problem description in {max_words} words>" # prompt 28
    # "risks, hazards" -> leads to censoring
    # "food-incident reports" -> also leads to censoring
    prompt = f"Text: {title}\n{example}.\n\nYour task is to extract the problem(s), risks, hazards with the product as briefly as possible, preserving the words in the article and details. Keep scientific terms. Respond in the following format and do not respond with anything else:\n<problem>. <problem description in {max_words} words>" # prompt 29
    return prompt


def make_common_prompt(example, title=""):
    # prompt = f"You are given an article about food-incident reports. Your first task is to find what product is described and extract it as it is found in the text, followed by a brief description. Your second task is to extract the problem(s) with the product as briefly as possible, preserving the words in the article. Do not include any numbers. Respond in the following format and do not respond with anything else:\nProduct: <product>. <product description>. Problem: <problem>. <problem description>.\n<end of your response>\n\nArticle: {title}\n{example}" # prompt 1
    prompt = f"Article: {title}\n{example}.\n\nYou are given an article about food-incident reports. Your first task is to find what product is described and extract it as it is found in the text, followed by a brief description. Your second task is to extract the problem(s) with the product as briefly as possible, preserving the words in the article. Do not include any numbers. Respond in the following format and do not respond with anything else:\nProduct: <product>. <product description>. Problem: <problem>; <problem description>.\n<end of your response>" # prompt 2
    return prompt


def make_request(prompt):
    # model = "Llama-3.2-3B-Instruct-Q6_K.gguf"
    model = "llama3.2"
    # model = "llama3.1:8b"
    # model = "llama3.2-q6-k"
    # model = "llama3.2:3b-instruct-q8_0"
    # prompt = f'Given the following text, extract the food product and the related food hazard. The text: "{example}". Format the answer as follows and do not include anything else: ' + '{"food": "<food>", "hazard": "<hazard>"}'
    # prompt = f'Given the following text, extract the food product and the related food hazard. The text: "{example}". Format the answer as follows and do not include anything else: Product: <food>\nHazard: <hazard>'
    # prompt = f'Given the following text, extract the food product and the related food hazard. Do not split words. The hazard might be mentioned multiple times: extract the most specific mention. Ask yourself if the product is actually a food item and if the hazard is actually a hazard. If either the food or the hazard is not mentioned, respond with "missing" in that category. The text: "{example}". Format the answer as follows and do not include anything else: Product: <food>\nHazard: <hazard>'

    request = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 50, # 20; 30 for f34
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
            num_errors += 1
            if num_errors % 100 == 0:
                print(f"[DEBUG] Response: {req.text}")
            continue
            # aaa = bytes(req.text, encoding='utf8')
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
            # if num_errors % 100 == 0:
            #     print("[DEBUG] reached 100 errors")
            continue
        if req.status_code == 200:
            break
        # print(f"[WARNING] Got {req.status_code}. Retrying...")

def main_eval_loop(file_name, include_product=True, include_hazard=True, add_title=False, common_prompt=False):
    header, entries = read_file(file_name)
    out_header = header[0:6] + header[7:11]
    if include_product or common_prompt:
        out_header += ['extracted_product']
    if include_hazard or common_prompt:
        out_header += ['extracted_hazard']
    out_entries = []
    num_errors = 0
    response = ""
    food = ""
    hazard = ""
    print("data loaded")
    cache_responses = {}
    num_cache_hits = 0
    debug_mode = False
    should_save = True
    if debug_mode:
        should_save = False
    try:
        for i, row in enumerate(tqdm.tqdm(entries)):
            if debug_mode:
                if i < 41:
                    continue
            title = row[5]
            # if add_title:
            #     clean_text = f"{row[5]}. {clean_text}"
            out_row = row[0:6] + row[7:11]
            # if i > 100:
            #     break
            #     exit(0)
            # continue
            if include_product:
                clean_text = make_clean_text(row[6], 'product', regex_filter=False)
                if not clean_text:
                    clean_text = title
                if "Recall Notification: FSIS" in title and "(" not in title:
                    cache_key = clean_text
                else:
                    cache_key = title + clean_text
                # print(f"made prompt with {cache_key}\n")
                # continue
                prompt = make_food_prompt(clean_text, title)
                # print(f"food prompt: {prompt}")
                if cache_responses.get(cache_key):
                    req_data = cache_responses[cache_key]
                    num_cache_hits += 1
                else:
                    req_data = make_request(prompt)
                    cache_responses[cache_key] = req_data
                # we cannot skip this step because the similarity search has trouble with peculiar terms
                # if len(clean_text) < 100:
                #     food = clean_text
                # else:
                food, err = process_request(req_data)
                if err:
                    num_errors += err
                # food = food.replace('**', '').replace("\n", " ").replace("  ", " ").replace("Synonyms: ", "").replace("Product category: ", "").replace("Definition: ", "").replace("Two synonyms: ", "").replace("Short definition: ", "").replace("Product Category: ", "").replace("Product Name: ", "").replace("Related terms: ", "").replace("Related Terms: ", "").replace("Uncommon", "").replace("Common", "").replace("<", "").replace(">", "")
                food = food.replace("\n", " ").replace("  ", " ").replace("Uncommon", "").replace("uncommon", "").replace("Common", "").replace("common", "").replace("<", "").replace(">", "")
                out_row += [food]
            if include_hazard:
                clean_text = make_clean_text(row[6], 'hazard')
                if debug_mode:
                    print(f"made prompt with {clean_text}\n")
                # if i > 45:
                #     break
                # continue
                prompt = make_hazard_prompt(clean_text, title)
                # print(f"hazard prompt: {prompt}")
                if cache_responses.get(prompt):
                    req_data = cache_responses[prompt]
                    num_cache_hits += 1
                else:
                    req_data = make_request(prompt)
                    cache_responses[prompt] = req_data
                hazard, err = process_request(req_data)
                if err:
                    num_errors += err
                out_row += [hazard]
            if common_prompt:
                prompt = make_common_prompt(clean_text, title)
                req_data = make_request(prompt)
                response, err = process_request(req_data)
                if err:
                    num_errors += 1
                # print(f"Reponse: {response}")
                # food, hazard = response.split("\n")
                food = response.split("Product: ")[1].split("Problem: ")[0]
                hazard = response.split("Problem: ")[1]
                out_row += [food, hazard]
            if debug_mode:
                print(f"got food: {food}\ngot hazard: {hazard}")

            out_entries.append(out_row)
            if debug_mode:
                if i > 0:
                    break
    except requests.exceptions.ConnectionError:
        print("connection error")
        should_save = False
    finally:
        if not should_save:
            exit(0)
        # print(f"Response: {response}")
        print(f"got food: {food}\ngot hazard: {hazard}")
        print(f"Total errors: {num_errors}")
        print(f"Total cache hits: {num_cache_hits}")
        # return
        # out_file = 'parsed_data_f21.csv'
        # out_file = 'parsed_data_f10.csv'
        # TODO repro f24 w/0 regex filter: need larger context; maybe only try to limit prompt to ~2k chars or smth
        out_file = 'parsed_data_f24_ollama_no_regex_repro.csv'
        out_file = 'parsed_data_f24_ollama_no_regex_valid.csv'
        # out_file = 'parsed_data_f24_ollama_no_regex_q8.csv'
        # out_file = 'parsed_data_f41_ollama_regex_q8.csv'
        # out_file = 'parsed_data_f42_ollama_regex.csv'
        # out_file = 'parsed_data_f36.csv'
        # out_file = 'parsed_data_f23_h16.csv'
        # out_file = 'parsed_data_f24_h17.csv'
        # out_file = 'parsed_data_f25_h18.csv'
        # out_file = 'parsed_data_f24_h17_validation.csv'
        # out_file = 'parsed_data_h29_ollama_regex_more_words.csv'
        # out_file = 'parsed_data_h25_4_ollama_regex_valid.csv'
        # out_file = 'parsed_data_h25_validation.csv'
        # out_file = 'parsed_data_c2.csv'
        # out_file = 'parsed_data_h14.csv'
        # out_file = 'parsed_data_validation_h14_f10_q6.csv'
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"')
            writer.writerow(out_header)
            for row in out_entries:
                writer.writerow(row)


def make_food_in_title_prompt(title):
    prompt = f"Does the following text mention a specific product? Text: {title}\nRespond only with yes or no, do not include anything else." # prompt 1 old1 bad
    prompt = f"Does the following text mention a name of a product? Text: {title}\nRespond only with yes or no, do not include anything else. Respond with yes only if you can find what product is mentioned." # prompt 1 old2 bad
    prompt = f"Is the text \"{title}\" sufficient to identify a product name and to also name that product without looking up other sources? Respond only with yes or no, do not include anything else." # prompt 1
    prompt = f"Example 1: for 'Recall Notification: FSIS-012-34', the answer is 'No' (need to look up the id). Example 2: 'recall of beef product', the answer is 'Yes' (beef mentioned). Example 3: 'recall of product', the answer is 'No' (nothing mentioned). Is the text \"{title}\" sufficient to identify a product name and to also name that product without looking up other sources? Respond only with yes or no, do not include anything else. Your answer (Yes/No): " # prompt 2
    return prompt


def main_filter_title(file_name, include_product=True, include_hazard=True):
    header, entries = read_file(file_name)
    out_header = ['id', 'title']
    if include_product:
        out_header += ['has_product']
    if include_hazard:
        out_header += ['has_hazard']
    out_entries = []
    should_save = True
    try:
        for i, row in enumerate(tqdm.tqdm(entries)):
            title = row[5]
            out_row = [row[0], title]
            if include_product:
                prompt = make_food_in_title_prompt(title)
                req_data = make_request(prompt)
                has_food, err = process_request(req_data)
                out_row += [has_food]
            if include_hazard:
                prompt = make_hazard_in_title_prompt(title)
                req_data = make_request(prompt)
                has_hazard, err = process_request(req_data)
                out_row += [has_hazard]
            out_entries.append(out_row)
    except requests.exceptions.ConnectionError:
        print("connection error")
        should_save = False
    finally:
        if not should_save:
            exit(0)
        out_file = 'title_data_f2_ollama.csv'
        with open(out_file, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile, delimiter=',', quotechar='"')
            writer.writerow(out_header)
            for row in out_entries:
                writer.writerow(row)


if __name__ == "__main__":
    # main_filter_title('data/incidents_train.csv', include_hazard=False)
    # main_eval_loop('data/incidents_train.csv', include_hazard=False, add_title=False)
    # main_eval_loop('data/incidents_train.csv', include_product=False)
    main_eval_loop('data/incidents_valid.csv', include_hazard=False)
    # main_eval_loop('data/incidents_valid.csv', include_product=False)
    # main_eval_loop('data/incidents_train.csv', add_title=False)
    # main_eval_loop('data/incidents_train.csv', include_product=False, include_hazard=False, common_prompt=True)
    # main_eval_loop('data/incidents_train.csv', include_hazard=False, add_title=False)
    # main_eval_loop('data/incidents_train.csv', include_product=False, add_title=False)
    # main_eval_loop('data/incidents_validation.csv', include_product=False, add_title=False)
