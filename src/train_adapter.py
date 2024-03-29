# https://towardsdatascience.com/adapterhub-a-framework-for-adapting-transformers-a21d0ab202a0
# https://adapterhub.ml/
# https://docs.adapterhub.ml/
# https://github.com/Adapter-Hub/adapter-transformers/tree/master/notebooks
# https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/03_Adapter_Fusion.ipynb
from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
import numpy as np
# from adapter_transformers import AutoModelForSequenceClassification, AdapterType
from datasets import load_dataset
from transformers import RobertaConfig, RobertaModelWithHeads
from transformers import RobertaTokenizer
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction

import pdb


def train_adapter():
    """

    :return:
    """
    misogynous_dataset = load_dataset("src/huggingface_datasets/misogynous_dataset.py")
    shaming_dataset = load_dataset("src/huggingface_datasets/shaming_dataset.py")
    stereotype_dataset = load_dataset("src/huggingface_datasets/stereotype_dataset.py")
    objectification_dataset = load_dataset("src/huggingface_datasets/objectification_dataset.py")
    violence_dataset = load_dataset("src/huggingface_datasets/violence_dataset.py")

    datasets = [misogynous_dataset, shaming_dataset, stereotype_dataset, objectification_dataset, violence_dataset]
    datasets_names = ["misogynous_dataset", "shaming_dataset", "stereotype_dataset", "objectification_dataset",
                      "violence_dataset"]
    labels_names = ["misogynous_label", "shaming_label", "stereotype_label", "objectification_label", "violence_label"]

    # https://adapterhub.ml/
    # common_dataset = concatenate_datasets(dsets=[misogynous_dataset, shaming_dataset, stereotype_dataset,
    # objectification_dataset, violence_dataset])
    num_epochs = 1
    for (dataset, dataset_name, label_name) in zip(datasets, datasets_names, labels_names):
        print("*" * 100)
        print("epochs: ", num_epochs)
        print(dataset_name)
        print(dataset.num_rows)
        print(dataset['train'][0])
        print("*" * 100)

        tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        def encode_batch(batch):
            """Encodes a batch of input data using the model tokenizer."""
            return tokenizer(batch["text"], max_length=80, truncation=True, padding="max_length")

        # Encode the input data
        dataset = dataset.map(encode_batch, batched=True)
        # The transformers model expects the target class column to be named "labels"
        dataset.rename_column_(label_name, "labels")
        # Transform to pytorch tensors and only output the required columns
        dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

        # training

        config = RobertaConfig.from_pretrained(
            "roberta-base",
            num_labels=2,
        )
        model = RobertaModelWithHeads.from_pretrained(
            "roberta-base",
            config=config,
        )

        # Add a new adapter
        model.add_adapter(dataset_name)
        # Add a matching classification head
        # model.save_adapter("./final_adapter/" + dataset_name, dataset_name)
        # exit()
        model.add_classification_head(
            dataset_name,
            num_labels=2,
            id2label={0: "👎", 1: "👍"}
        )
        # Activate the adapter
        model.train_adapter(dataset_name)

        training_args = TrainingArguments(
            learning_rate=1e-4,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            logging_steps=200,
            output_dir="./training_output/" + dataset_name,
            overwrite_output_dir=True,
            # The next line is important to ensure the dataset labels are properly passed to the model
            remove_unused_columns=False,
        )

        def compute_accuracy(p: EvalPrediction):
            """

            :param p:
            :return:
            """
            preds = np.argmax(p.predictions, axis=1)
            return {"acc": (preds == p.label_ids).mean()}

        trainer = AdapterTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            compute_metrics=compute_accuracy,

        )

        trainer.train()
        trainer.evaluate()

        # classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer)
        # classifier("This is awesome!")

        model.save_adapter("./final_adapter/" + dataset_name, dataset_name)

        print("*" * 100)
        print("epochs: ", num_epochs)
        print(dataset_name)
        print(dataset.num_rows)
        print(dataset['train'][0])
        print("*" * 100)


def main():
    train_adapter()


if __name__ == '__main__':
    main()
