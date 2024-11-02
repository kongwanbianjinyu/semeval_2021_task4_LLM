import jsonlines

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


if __name__ == '__main__':
    task1_dev_path = "/Users/jiachenjiang/Documents/GitHub/semeval_2021_task4_LLM/data/training_data/Task_1_dev.jsonl"
    task2_dev_path = "/Users/jiachenjiang/Documents/GitHub/semeval_2021_task4_LLM/data/training_data/Task_2_dev.jsonl"

    task1_dev_set = read_recam(task1_dev_path)
    task2_dev_set = read_recam(task2_dev_path)