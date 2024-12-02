# semeval_2021_task4_LLM

Understanding abstract meanings is crucial for advanced language comprehension. Despite extensive research, abstract words remain challenging due to their non-concrete, high-level semantics. SemEval-2021 Task 4 (ReCAM) evaluates models’ ability to interpret abstract concepts by presenting passages with questions and five abstract options in a cloze-style format. Key findings include: (1) Most large language models (LLMs), including GPT-4o, struggle with abstract meaning comprehension under zero-shot, one-shot, and few-shot settings, while fine-tuned models like BERT and RoBERTa perform better. (2) A proposed bi-directional attention classifier, inspired by human cognitive strategies, enhances fine-tuned models by dynamically attending to passages and options. This approach improves accuracy by 4.06\% on Task 1 and 3.41\% on Task 2, demonstrating its potential for abstract meaning comprehension.

## Dataset
Go to https://competitions.codalab.org/competitions/26153#learn_the_details-overview to learn more about SemEval-2021 shared task 4. The data and baseline code are available at https://github.com/boyuanzheng010/SemEval2021-Reading-Comprehension-of-Abstract-Meaning. The data are stored in JSONL format. Copy the data folder to your pwd.

## Environment Setup
This project requires pytorch-lightning, transformers and datasets (please use requirements.txt for installation)
```
pip install -r requirements.txt
```

## LLM Based
### Local LLM
Download LM studio(https://lmstudio.ai/), load the model locally and open the server, copy the url of server to --base_url, set --n_shots to do zero-shot/ one-shot / two-shots setting,

```
python inference_LLM.py --base_url http://localhost:1234/v1 \
                         --model_api_identifier gemma-2-9b-it \
                         --n_shots 2 \

```

### API Based LLM
Apply and pay for the OpenAI API(https://platform.openai.com/docs/api-reference/introduction), export the OPEN API KEY to your environment variables, then run,

```
python inference_LLM.py --base_url "" \
                        --model_api_identifier gpt-4o-mini \
                        --n_shots 2 \
```

## Finetuning the BERT
We try 3 different methods: Electra Pretrained model, Electra + Uni-Directional Attention Classifier and Electra + Bi-Directional Attention Classifier. You can run the corresponding code you interested. 

For example, To get our result for task1-task3 using Electra Pretrained model:
```
# subtask1
python Electra.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
# subtask2
python Electra.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task1 and validation on task2)
python Electra.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task2 and validation on task1)
python Electra.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
```
To get our result for task1-task3 using Electra + Uni-Directional Attention model:
```
# subtask1
python Electra_MAMC.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
# subtask2
python Electra_MAMC.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task1 and validation on task2)
python Electra_MAMC.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task2 and validation on task1)
python Electra_MAMC.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
```
To get our result for task1-task3 using Electra + Bi-Directional Attention model:
```
# subtask1
python Electra_DUMA.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
# subtask2
python Electra_DUMA.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task1 and validation on task2)
python Electra_DUMA.py 'data/training_data/Task_1_train.jsonl' 'data/training_data/Task_2_dev.jsonl'
# subtask3(training on task2 and validation on task1)
python Electra_DUMA.py 'data/training_data/Task_2_train.jsonl' 'data/training_data/Task_1_dev.jsonl'
```
If you are interested in the result of Roberta model, you can run Roberta.py like others. But the performance of Roberta is not so good.


## Run google colab files
We do all experiments in google colab. All files are under Google Colab NLP folders. You can download and open with google colab. You'd better upgrade to Colab Pro+ to run the code quicklly. The resulting screenshots are under Image folder.

 
