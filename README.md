# SemEval 2025 Task 9 UniBuc team

### Data cleaning

Dependencies:
- requests==2.32.3
- tqdm==4.67.0
- LLM server (ollama 0.3.13)

Start a LLM server, e.g.
```sh
ollama serve
```
If using another server, update the host/port in the `process_request` function.

Check that the `main` function and the `out_file` in the `data_cleaning.py` script contain the desired file names.

Run the data cleaning script:
```sh
python data_cleaning.py
```

### Prepare labels

⚠️ This should be refactored so less manual work is needed. For now, use the following steps:

If needed, merge initial data files (e.g. train + valid):
```python
from utils import merge_files
# merge_files(first_file, second_file, destination)
# example
merge_files('data/incidents_train.csv', 'data/incidents_valid.csv', 'data/incidents_train_valid.csv')
```

Write label files:
```python
from utils import write_label_file
write_label_file('hazard', 'train_valid')  # from the file above; might simply use 'train' or 'valid' or 'test' or other combination
write_label_file('product', 'train_valid')
```

The string `train_valid` (in the example above) is used as "label_set" in the code base.

### Order labels

Dependencies:
- editdistance==0.8.1
- requests==2.32.3
- tqdm==4.67.0
- LLM server (ollama 0.3.13)

Update the `label_set` in the `main` function in `order_labels.py` (optionally also update `key`). If updating the destination file, this also needs to be updated in the similarity script.

Run sorting:
```sh
python order_labels.py
```

### Run similarity classification script

Dependencies:
- numpy==1.26.4
- sentence-transformers==3.3.1
- torch==2.5.1
- tqdm==4.67.0
- nltk==3.9.1
- scikit-learn==1.5.2
- simplemma==1.1.2

Update `hazard_file_name` and `product_file_name` with the files resulted from data cleaning.

Run the script:
```sh
python similarity.py
```

### Analyze prediction mistakes

Update the file name in the `main` function in `wrong.py`.

Run the script:
```sh
python wrong.py
```

Check the `wrong` folder.

### Other scripts

The other scripts are legacy, with discarded attempts. They can be safely ignored.
