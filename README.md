# GEOBERTje
**A Domain-Adapted Dutch Language Model Trained on Geological Borehole Descriptions**

<img src="https://raw.githubusercontent.com/hghcomphys/hghcomphys.github.io/master/assets/urls/GEOBERTJE.png" width="300">


## Description 
GEOBERTje is a language model built upon the [BERTje](https://github.com/wietsedv/bertje) architecture, comprising 109 million parameters. It has been further trained using masked language modeling on a dataset of approximately 300,000 borehole descriptions in the Dutch language from the Flanders region in Belgium. It can serve as the base language model for a variety of geological applications. For instance, by leveraging the model's understanding of geological terminology and borehole data, professionals can streamline the process of interpreting subsurface information and generating detailed 3D representations of geological structures. 
This capability opens up new possibilities for improved exploration, interpretation, and analysis in the field of geology. 

To showcase the potential application of GEOBERTje, we fine-tune it on a limited dataset of 3,000 labeled samples. This fine-tuning allowed the model to classify various lithological categories.
For example `Grijs kleiig zand, zeer fijn, met enkele grindjes` will be classified as main lithology: `fijn zand`, second lithology: `klei`, third lithology: `grind`.
Our classifier obtained higher accuracy than conventional rule-based approaches or zero-shot classification using GPT-4.

GEOBERTje is freely available on [Hugging Face](https://huggingface.co/hghcomphys/geobertje-base-dutch-uncased) and also our paper on [arxiv](https://arxiv.org/abs/2407.10991).

---

## Model Training
The following sections provide a detailed descriptions on data preparation, domain adaptation of BERTje for geology, and subsequent fine-tuning for tasks related to lithology classification.

### Environment Setup
We need to create the required Python environment and this can be done using [poetry](https://python-poetry.org/) as follows:

```bash
$ git clone https://github.com/VITObelgium/geobertje.git && cd geobertje
$ poetry install  
```
This will install [PyTorch](https://pytorch.org/get-started/locally/), [Transformers](https://pypi.org/project/transformers/), [datasets](https://pypi.org/project/datasets/), and few other small packages. Also, the local `lithonlp` package will be installed for use in examples scripts or notebooks (i.e. `import lithonlp`). 

_Note_: To install `poetry` try `pipx install poetry` or follow the official [documentation](https://python-poetry.org/docs/#installation). 

Then instead of using `python <script.py>` simply use `poetry run python <script.py>` to run the training or prediction scripts. Or, alternatively use `poetry shell`

__Note__: If you want to run the model, you can skip the next section and go directly to the `Running the model` section. 

### Getting Raw Dataset 
We can download [dataset](https://huggingface.co/datasets/hghcomphys/geological-borehole-descriptions-dutch) in `.csv` format from the Hugging Face using the following command:
```bash
$ huggingface-cli download hghcomphys/geological-borehole-descriptions-dutch --repo-type dataset --local-dir csvdata
```
We expect two csv files: `unlabeled_lithological_descriptions.csv` and `labeled_lithological_descriptions.csv` in the out put `csvdata` directory.
For more details about the dataset, please check the dataset's readme file.


### Domain Adaptation
The model domain adaptation technique is particularly useful when there is limited labeled data available for the target task. By leveraging the knowledge acquired during the pre-training on the **unlabelled** dataset, the adapted model can often achieve better performance on the specific task than a model fine-tune trained from a generic base model. The resulted model [GEOBERTje](https://huggingface.co/hghcomphys/geobertje-base-dutch-uncased) is available on Hugging Face.

The below example processes directly a raw input dataset, converts it to Hugging Face test and train subsets (see output `dataset`  and `tokenized_dataset` directories), and then training the `Bertje` base model using masked language model (training details are available in `trainer` directory). 
```bash
$ cd domain_adaptation
$ python pretrain-cli.py \
    --unlabeled-dataset-file ../csvdata/unlabeled_lithological_descriptions.csv
```
The optimal model checkpoint obtained can serve as a new domain-adapted base model for the subsequent fine-tuning training with labeled data.

### Dataset Preparation
The `dataset.py` script takes an input raw **labeled** data file (csv file), divides it into train and test subsets using the original `Bertje` tokenizer, and then proceeds to tokenize these subsets. 

To prepare the hugging face dataset from an input dataframe file for any of `HL`, `NL1`, and `NL2` target columns, use the following commands:
```bash
$ cd fine_tuning
$ python dataset-cli.py \ 
    --raw-dataset-file  ../csvdata/labeled_lithological_descriptions.csv \ 
    --target-column HL_cor 
```
All necessary parameters are extracted from the default configuration, although additional adjustments can be made using input keywords. 
The `class weights` and the output directory `tokenized_dataset` will be required during model training.

### Finetuning
Fine-tuning GEOBERTje for the lithological classification task can be done through the subsequent commands:
```bash
$ cd fine_tuning
$ python train-cli.py \ 
    --pretrained-base-model PATH_TO_GEOBERTJE \ 
    --target-column HL_cor 
```
This script trains the model using the input tokenized dataset directory and stores the model checkpoint within the trainer. 
These checkpoints can then be employed to create a classifier model. 
An extra attention should be paid when selecting checkpoints to avoid overfitting (e.g. by comparing train and evaluation loss values)

It is also recommended to utilize the output class weights derived from the dataset preparation to enhance the training algorithm's effectiveness in handling input datasets with imbalanced distributions.

---

## Running the model
We provide three ways of running the model: Python module, terminal (CLI) and web API.

### Python module
```python
from lithonlp.predict import DrillCoreClassifier

classifier = DrillCoreClassifier.from_directory(
    PATH_TO_MODEL,
)
print(classifier(
    `Grijs kleiig zand, zeer fijn, met enkele grindjes`,
     cutoffval=0.1,
)
```

### CLI
The classification model prediction for the input text can be obtained using the following command:
```bash
$ cd fine_tuning
$ python predict-cli.py \ 
    --bundled-model-path PATH_TO_MODEL \ 
    "geel grijs heteromorf zand met fijn grind"   
```
Output:
```bash
{
        'HL_cor': [{'label': 'zand_onb', 'score': 0.9575759172439575}], 
        'NL1_cor': [{'label': 'grind', 'score': 0.9712706208229065}], 
        'NL2_cor': [{'label': 'none', 'score': 0.540747344493866}]
}
```

### Web API
Trained model can be deployed using `FastAPI` as follows: 
```bash
$ uvicorn deploy:app --reload --host=0.0.0.0 --port=56123
```
You can then access it via `http://HOST_NAME:56123/docs#/predict`.

