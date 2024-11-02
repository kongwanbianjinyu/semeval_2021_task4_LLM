# use local LLM model of LM studio to do inference

import argparse
import random
import jsonlines
from openai import OpenAI
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Local LLM inference for ReCAM tasks')

    parser.add_argument('--base_url', type=str,
                        default='http://localhost:1234/v1', # http://localhost:1234/v1 for Local LLM
                        help='Base URL for the local LLM API endpoint')

    parser.add_argument('--model_api_identifier', type=str,
                        default='meta-llama-3.1-8b-instruct', # gpt-3.5-turbo-instruct
                        help='API identifier for the local LLM model')
    
    parser.add_argument('--dataset_path', type=str, 
                        default='/Users/jiachenjiang/Documents/GitHub/semeval_2021_task4_LLM/data/training_data/Task_1_dev.jsonl',
                        help='Path to the dataset file')
    
    parser.add_argument('--prompt_style', type=str, default='generate',
                        choices=['fill-back-echo', 'complete-echo', 'generate'],
                        help='Style of prompt to use')
    
    parser.add_argument('--n_shots', type=int, default=1,
                        help='Number of few-shot examples to include in prompt')
    
    parser.add_argument('--do_strong_shuffle', action='store_true',
                        help='Force correct answer index to change for each example')
    
    parser.add_argument('--do_perm', action='store_true', 
                        help='Process every example with all possible answer orderings')

    args = parser.parse_args()
    return args

def read_recam(path, show_num=5):
    dataset = []
    with open(path, mode='r') as f:
        reader = jsonlines.Reader(f)
        count = 0
        for instance in reader:
            count += 1
            if count <= show_num:
                print(instance)
            dataset.append(instance)
    print(f"Total number of instances in dataset: {len(dataset)}")
    return dataset

class Prompt:
    def __init__(self, sample, n_shots=0, shots=[]):
        self.sample = sample
        self.n_shots = n_shots
        self.shots = shots

    def build_generate_prompt(self):
        sys_prompt = "Given the article below and the corresponding question, you are expected to choose the correct answer from five candidates to fill @placeholder of abstract concepts in cloze-style machine reading comprehension tasks. Output answer with a single number, choose option from [0,1,2,3,4] that best fits the @placeholder in question. I will provide you with a few-shot examples to help you understand the task better. "
        article = self.sample['article']
        question = self.sample['question']
        option0 = self.sample['option_0']
        option1 = self.sample['option_1']
        option2 = self.sample['option_2']
        option3 = self.sample['option_3']
        option4 = self.sample['option_4']
        
        few_shot_examples = ""
        for shot in self.shots:
            shot_article = shot['article']
            shot_question = shot['question']
            shot_option0 = shot['option_0']
            shot_option1 = shot['option_1']
            shot_option2 = shot['option_2']
            shot_option3 = shot['option_3']
            shot_option4 = shot['option_4']
            shot_answer = shot['label']
            few_shot_examples += f"article :{shot_article}, question:{shot_question}, 0:{shot_option0}, 1:{shot_option1}, 2:{shot_option2}, 3:{shot_option3}, 4:{shot_option4}, Answer: {shot_answer}\n"
        
        prompt = f"{sys_prompt} {few_shot_examples} article :{article}, question:{question}, 0:{option0}, 1:{option1}, 2:{option2}, 3:{option3}, 4:{option4}, Answer: "
        #print("Generate Style Prompt: ", prompt)
        return prompt
    
    def build_fill_back_echo_prompt(self):
        article = self.sample['article']
        question = self.sample['question']
        option0 = self.sample['option_0']
        option1 = self.sample['option_1']
        option2 = self.sample['option_2']
        option3 = self.sample['option_3']
        option4 = self.sample['option_4']
        
        prompt = f"article :{article}, question:{question}, 0:{option0}, 1:{option1}, 2:{option2}, 3:{option3}, 4:{option4}, Answer: "
        #print("Fill Back Echo Style Prompt: ", prompt)
        return prompt

class LMStudioModel:
    def __init__(self, args):
        self.use_local_model = args.base_url != ""
        if self.use_local_model:
            self.client = OpenAI(base_url = args.base_url, api_key="lm-studio")
        else:
            self.client = OpenAI()

    def _get_response(self, model, input, echo=True):
        # call the model
        if self.use_local_model:
            response = self.client.completions.create(
                model=model,
                prompt=input,
                temperature=0,
                max_tokens = 1,
                logprobs = 5,
                echo = echo
            )
        else:  
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": input}
                ],
                temperature=0,
                max_tokens = 1,
                logprobs = True,
                top_logprobs = 2,
            )

            # OPENAI API doesn't support completions for chat model anymore
            # response = self.client.completions.create(
            #     model=model,
            #     prompt=input,
            #     temperature=0,
            #     max_tokens = 0,
            #     #logprobs = 2,
            #     echo = True,
            #     logprobs = 5,
            #     logit_bias = {}
            # )
            
        return response

    def process_generate_style_prompt(self, model, prompt):
        result = {}
        response = self._get_response(model, prompt, echo=False)
        if self.use_local_model:
            #print("LLM Response: ", response)
            text = response.choices[0].text.strip()
        else:
            text = response.choices[0].message.content.strip()
        # Extract just the number, removing any trailing characters
        number = ''.join(char for char in text if char.isdigit())
        pred_choice = int(number) if number else 0  # Default to 0 if no number found
        result["pred_choice"] = pred_choice
        # Since logprobs is None in this response, we'll just store the prediction
        result["logprobs"] = None
    
        return result
    
    def process_fill_back_echo_style_prompt(self, model, prompt):
        result = {}
        response = self._get_response(model, prompt, echo=True)

        return response

        

def main(args):
    print(args)
    # load local model 
    model = LMStudioModel(args)
    # load dataset
    dataset = read_recam(args.dataset_path, show_num = 0)
    correct = 0
    total = 0
    for i, sample in tqdm(enumerate(dataset), total=len(dataset), desc="Processing samples"):
        label = int(sample['label'])
        prompt = Prompt(sample, args.n_shots, random.sample(dataset, args.n_shots))
        if args.prompt_style == 'generate':
            generate_style_prompt = prompt.build_generate_prompt()
            result = model.process_generate_style_prompt(args.model_api_identifier, generate_style_prompt)
            pred_choice = result["pred_choice"]
            print(generate_style_prompt + f"|| Correct Answer " + str(pred_choice) + " || " + str(label))
            if pred_choice == label:
                correct += 1
            total += 1
        elif args.prompt_style == 'fill-back-echo':
            fill_back_echo_style_prompt = prompt.build_fill_back_echo_prompt()
            result = model.process_fill_back_echo_style_prompt(args.model_api_identifier, fill_back_echo_style_prompt)
            print(result)
            
    
    accuracy = correct / total if total > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    args = parse_args()
    main(args)