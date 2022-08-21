from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from simpletransformers.classification import ClassificationModel
from sklearn.model_selection import train_test_split
from datasets import load_dataset

import pandas as pd

import logging
import pdb


def main():
    logging.basicConfig(level=logging.INFO)
    transformers_logger = logging.getLogger("transformers")
    transformers_logger.setLevel(logging.WARNING)

    # They are lot of arguments to play with
    '''
    args = {
       'output_dir': 'outputs/',
       'cache_dir': 'cache/',
       'fp16': True,
       'fp16_opt_level': 'O1',
       'max_seq_length': 256,
       'train_batch_size': 8,
       'eval_batch_size': 8,
       'gradient_accumulation_steps': 1,
       'num_train_epochs': 3,
       'weight_decay': 0,
       'learning_rate': 4e-5,
       'adam_epsilon': 1e-8,
       'warmup_ratio': 0.06,
       'warmup_steps': 0,
       'max_grad_norm': 1.0,
       'logging_steps': 50,
       'evaluate_during_training': False,
       'save_steps': 2000,
       'eval_all_checkpoints': True,
       'use_tensorboard': True,
       'overwrite_output_dir': True,
       'reprocess_input_data': False,
    }
    '''

    # Create a ClassificationModel

    # You can set class weights by using the optional weight argument

    misogynous_dataset = load_dataset("src/huggingface_datasets/misogynous_dataset.py")
    shaming_dataset = load_dataset("src/huggingface_datasets/shaming_dataset.py")
    stereotype_dataset = load_dataset("src/huggingface_datasets/stereotype_dataset.py")
    objectification_dataset = load_dataset("src/huggingface_datasets/objectification_dataset.py")
    violence_dataset = load_dataset("src/huggingface_datasets/violence_dataset.py")

    print(len(misogynous_dataset), len(shaming_dataset), len(stereotype_dataset), len(violence_dataset))
    datasets = [objectification_dataset]
    datasets_names = ["objectification_dataset"]
    labels_names = ["objectification"]

    assert len(datasets) == len(datasets_names)
    assert len(datasets_names) == len(labels_names)
    assert len(datasets) == len(labels_names)

    # https://snrspeaks.medium.com/fine-tuning-xlnet-model-for-text-classification-in-3-lines-of-code-1a7c3b320669

    df_path = "data/TRAINING_csvs/training_no_bad_lines.csv"
    df = pd.read_csv(df_path)
    print(datasets_names)
    print(len(datasets))
    for (dataset, dataset_name, label_name) in zip(datasets, datasets_names, labels_names):
        model = ClassificationModel('xlnet', 'xlnet-base-cased',
                                    args={'num_train_epochs': 10,
                                          'train_batch_size': 8,
                                          'eval_batch_size': 8,
                                          'max_seq_length': 64,
                                          'output_dir': "xlnet_" + dataset_name}, use_cuda=False)

        X = df['Text Transcription'].to_list()
        y = df[label_name].to_list()

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        train_df = pd.DataFrame(X_train)
        train_df['target'] = y_train

        eval_df = pd.DataFrame(X_test)
        eval_df['target'] = y_test

        # Train the model
        model.train_model(train_df)
        # Evaluate the model
        # result, model_outputs, wrong_predictions = model.eval_model(eval_df, acc=sklearn.metrics.accuracy_score)

        model.save_model("xlnet_checkpoints/" + dataset_name)


if __name__ == "__main__":
    main()
