# Finetuning BioBert for arthropod species and gene Relation Extraction

Project aims to collect a literature corpus as our training and testing data with manual labeled relationship, from abstracts in the arthropod sciences. We finetuned the [BioBert](https://github.com/dmis-lab/biobert-pytorch) to perform relation extraction (RE) for relationship between arthropod species and gene name.

### Requirment
`Python3` and `Colab`<br>
The introduction of [Colab](https://colab.research.google.com/?utm_source=scs-index#scrollTo=5fCEDCU_qrC0).

### Installation
- seqeval : Used for evaluation (`pip install seqeval`)
- inflect (`pip install inflect`)
- nltk (`pip install nltk`)
- sklearn (`pip install scikit-learn`)
- transformers (`pip install transformers`)
- beautifulsoup4 (`pip install beautifulsoup4`)
- pandas (`pip install pandas`)

### Preprocess
We use TeamTat to label relationship between species and gene, "1" is positive, "0" is negative.
Use `preprocess_re.py` to preprocess dataset to BioBERT's input format.
- split our data into 5/10 folds (5/10 folds cross validation)
- leave-one-out cross validation

### Finetuning BioBERT
Put the dataset and `run_re.py`, `metrics.py`, `run_re_10cv.sh`, `run_re_colab.ipynb` on google drive, and run `run_re_colab.ipynb` on Colab.

`run_re_10cv.sh` store the training setting and data path. 

### Evaluation
Evaluate test predictions.

`python re_eval.py --output_path=${SAVE_DIR}/test_results.txt --answer_path=${DATA_DIR}/test_original.tsv`

`re_eval_10cv.sh` can 5-cv/10-cv/leave-one-out cv experiment result.


### The result of 5 folds cross validation

| Test Fold      |    Test Precision (%)   |    Test Recall (%)   |    Test F1 (%)   |Test Specificity(%)|
|----------------|:-----------------------:|:--------------------:|:----------------:|:-----------------:|
| fold_0         |          96.92          |         99.21        |       98.05      |       50.00       |
| fold_1         |          98.43          |         98.43        |       98.43      |       75.00       |
| fold_2         |          97.66          |         100.00       |       98.83      |       62.50       |
| fold_3         |          96.18          |         99.21        |       97.67      |       37.50       |
| fold_4         |          98.41          |         97.64        |       98.02      |       71.43       |
| Average        |          97.52          |         98.89        |       98.20      |       59.29       |

### The result of 10 folds cross validation

| Test Fold      |    Test Precision (%)   |    Test Recall (%)   |    Test F1 (%)   |Test Specificity(%)|
|----------------|:-----------------------:|:--------------------:|:----------------:|:-----------------:|
| fold_0         |          100.00         |         100.00       |       100.00     |       100.00      |
| fold_1         |          98.46          |         100.00       |       99.22      |       75.00       |
| fold_2         |          98.44          |         98.44        |       98.44      |       75.00       |
| fold_3         |          98.46          |         100.00       |       99.22      |       75.00       |
| fold_4         |          96.88          |         96.88        |       96.88      |       50.00       |
| fold_5         |          97.64          |         98.41        |       96.88      |       50.00       |
| fold_6         |          95.31          |         96.83        |       96.06      |       25.00       |
| fold_7         |          98.44          |         100.00       |       99.21      |       75.00       |
| fold_8         |          100.00         |         100.00       |       100.00     |       100.00      |
| fold_9         |          99.21          |         100.00       |       99.21      |       66.67       |
| Average        |          98.28          |         99.06        |       98.51      |       69.17       |


### The result of leave-one-out cross validation

| Test Fold      |    Test Precision (%)   |    Test Recall (%)   |    Test F1 (%)   |Test Specificity(%)|
|----------------|:-----------------------:|:--------------------:|:----------------:|:-----------------:|
| Average        |          98.12          |         98.74        |       98.43      |       69.23       |


## Citation
```bibtex
@article{lee2020biobert,
  title={BioBERT: a pre-trained biomedical language representation model for biomedical text mining},
  author={Lee, Jinhyuk and Yoon, Wonjin and Kim, Sungdong and Kim, Donghyeon and Kim, Sunkyu and So, Chan Ho and Kang, Jaewoo},
  journal={Bioinformatics},
  volume={36},
  number={4},
  pages={1234--1240},
  year={2020},
  publisher={Oxford University Press}
}
```
