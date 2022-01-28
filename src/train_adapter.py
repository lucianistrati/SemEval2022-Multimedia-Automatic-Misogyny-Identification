# https://towardsdatascience.com/adapterhub-a-framework-for-adapting-transformers-a21d0ab202a0
# https://adapterhub.ml/
# https://docs.adapterhub.ml/
# https://github.com/Adapter-Hub/adapter-transformers/tree/master/notebooks
# https://colab.research.google.com/github/Adapter-Hub/adapter-transformers/blob/master/notebooks/03_Adapter_Fusion.ipynb
import numpy as np
# from adapter_transformers import AutoModelForSequenceClassification, AdapterType
from datasets import load_dataset
from transformers import RobertaConfig, RobertaModelWithHeads
from transformers import RobertaTokenizer
from transformers import TextClassificationPipeline
from transformers import TrainingArguments, AdapterTrainer, EvalPrediction
from transformers import AutoModelForSequenceClassification, AdapterType


def train_adapter_1():
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    model.add_adapter("sst-2", AdapterType.text_task, config="pfeiffer") # TypeError: add_adapter() got multiple values for argument 'config'
    model.train_adapters(["sst-2"])  # Train model ...

    model.save_adapter("adapters/text-task/sst-2/", "sst")
    # Push link to zip file to AdapterHub ...


def train_adapter_2():
    model = AutoModelForSequenceClassification.from_pretrained("roberta-base")
    model.load_adapter("sst", config="pfeiffer")

from datasets import concatenate_datasets
import pdb
# https://github.com/Adapter-Hub/adapter-transformers/blob/master/notebooks/01_Adapter_Training.ipynb
def train_adapter_3():
    # dataset preprocessing
    dataset = load_dataset("rotten_tomatoes")

    # pdb.set_trace()
    misogynous_dataset = load_dataset("src/huggingface_datasets/misogynous_dataset.py")
    shaming_dataset = load_dataset("src/huggingface_datasets/shaming_dataset.py")
    stereotype_dataset = load_dataset("src/huggingface_datasets/stereotype_dataset.py")
    objectification_dataset = load_dataset("src/huggingface_datasets/objectification_dataset.py")
    violence_dataset = load_dataset("src/huggingface_datasets/violence_dataset.py")

    datasets = [misogynous_dataset, shaming_dataset, stereotype_dataset, objectification_dataset, violence_dataset]
    datasets_names = ["misogynous_dataset", "shaming_dataset", "stereotype_dataset", "objectification_dataset", "violence_dataset"]
    labels_names = ["misogynous_label", "shaming_label", "stereotype_label", "objectification_label", "violence_label"]
    # common_dataset = concatenate_datasets(dsets=[misogynous_dataset, shaming_dataset, stereotype_dataset, objectification_dataset, violence_dataset])
    num_epochs = 1
    for (dataset, dataset_name, label_name) in zip(datasets, datasets_names, labels_names):
        print(dataset.num_rows)
        print(dataset['train'][0])

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
            per_device_train_batch_size=32,
            per_device_eval_batch_size=32,
            logging_steps=200,
            output_dir="./training_output/" + dataset_name,
            overwrite_output_dir=True,
            # The next line is important to ensure the dataset labels are properly passed to the model
            remove_unused_columns=False,
        )

        def compute_accuracy(p: EvalPrediction):
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

        classifier = TextClassificationPipeline(model=model, tokenizer=tokenizer, device=training_args.device.index)
        classifier("This is awesome!")

        model.save_adapter("./final_adapter", dataset_name)


def main():
    # train_adapter_1() # TypeError: add_adapter() got multiple values for argument 'config' L18
    # train_adapter_2() # OSError: No adapter with name 'sst' was found in the adapter index. L27
    train_adapter_3()

if __name__ == '__main__':
    main()
