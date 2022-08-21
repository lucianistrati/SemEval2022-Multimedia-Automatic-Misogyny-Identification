from typing import Dict, List, Set, Tuple, Optional, Any, Callable, NoReturn, Union, Mapping, Sequence, Iterable
from datasets import Features, ClassLabel, Array3D, load_metric, load_dataset
from transformers import ViTFeatureExtractor, TrainingArguments, Trainer
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import ViTForImageClassification, default_data_collator, ViTModel


import torch.nn as nn
import numpy as np

import pdb


def preprocess_images(examples):
    """

    :param examples:
    :return:
    """
    feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
    images = examples['img']
    images = [np.array(image, dtype=np.uint8) for image in images]
    images = [np.moveaxis(image, source=-1, destination=0) for image in images]
    inputs = feature_extractor(images=images)
    examples['pixel_values'] = inputs['pixel_values']

    return examples


class ViTForImageClassification2(nn.Module):
    """

    """

    def __init__(self, num_labels=10):
        """

        :param num_labels:
        """
        super(ViTForImageClassification2, self).__init__()
        self.vit = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, pixel_values, labels):
        outputs = self.vit(pixel_values=pixel_values)
        logits = self.classifier(outputs)
        loss = None

        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()

            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


def compute_metrics(eval_pred):
    """

    :param eval_pred:
    :return:
    """
    metric_name = "accuracy"
    metric = load_metric(metric_name)
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return metric.compute(predictions=predictions, references=labels)


def main():  # Killed
    metric_name = "accuracy"
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')  # try smaller model

    # import pdb
    # pdb.set_trace()
    # model.eval()
    #

    features = Features({
        'label': ClassLabel(names=["0", "1"]),
        'img': Array3D(dtype="int64", shape=(3, 32, 32)),
        'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    })

    # features = Features({
    #     'label': ClassLabel(names=['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship',
    #     'truck']),
    #      'img': Array3D(dtype="int64", shape=(3, 32, 32)),
    #     'pixel_values': Array3D(dtype="float32", shape=(3, 224, 224)),
    # })

    ds = load_dataset('src/huggingface_datasets/vision_dataset.py')

    train_ds = ds["train"]
    val_ds = ds["validation"]
    test_ds = ds["test"]

    preprocessed_train_ds = train_ds.map(preprocess_images, batched=True, features=features)
    preprocessed_val_ds = val_ds.map(preprocess_images, batched=True, features=features)
    preprocessed_test_ds = test_ds.map(preprocess_images, batched=True, features=features)

    data_collator = default_data_collator

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')

    model.train()

    args = TrainingArguments(
        'src/huggingface_datasets/vision_dataset.py',
        # evaluation_strategy = "epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=10,
        per_device_eval_batch_size=4,
        num_train_epochs=3,
        weight_decay=0.01,
        # load_best_model_at_end=True,
        metric_for_best_model=metric_name,
        logging_dir='logs'
    )

    trainer = Trainer(
        model,
        args,
        train_dataset=preprocessed_train_ds,
        eval_dataset=preprocessed_val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    trainer.train()

    outputs = trainer.predict(preprocessed_test_ds)
    y_pred = outputs.predictions.argmax(1)

    print(len(y_pred))
    # https://theaisummer.com/hugging-face-vit/


if __name__ == "__main__":
    main()
