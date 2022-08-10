import csv

import datasets
from datasets.tasks import TextClassification

_DESCRIPTION = """MAMI"""

_CITATION = """MAMI"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"


class MisogynousDataset(datasets.GeneratorBasedBuilder):
    """AG News topic classification dataset."""

    def _info(self):
        """

        :return:
        """
        labels = [0, 1]
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                    "misogynous_label": datasets.features.ClassLabel(names=labels),
                    "objectification_label": datasets.features.ClassLabel(names=labels),
                    "shaming_label": datasets.features.ClassLabel(names=labels),
                    "stereotype_label": datasets.features.ClassLabel(names=labels),
                    "violence_label": datasets.features.ClassLabel(names=labels),
                }
            ),
            homepage="http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
            citation=_CITATION,
            task_templates=[
                TextClassification(text_column="text", label_column="misogynous_label",
                                   labels=labels)],
        )

    def _split_generators(self, dl_manager):
        """

        :param dl_manager:
        :return:
        """
        train_path = "data/TRAINING/training_no_bad_lines.csv"
        test_path = "data/TRAINING/training_no_bad_lines.csv"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": test_path}),
        ]

    def _generate_examples(self, filepath):
        """

        :param filepath:
        :return:
        """
        labels = [0, 1]
        with open(filepath, encoding="utf-8") as csv_file:
            csv_reader = csv.reader(
                csv_file, quotechar='"', delimiter=",", quoting=csv.QUOTE_ALL,
                skipinitialspace=True
            )
            for id_, row in enumerate(csv_reader):
                misogynous_label = row[1]
                objectification_label = row[2]
                shaming_label = row[3]
                stereotype_label = row[4]
                violence_label = row[5]
                text = row[-1]
                if misogynous_label not in labels or objectification_label not in labels or shaming_label not in \
                        labels or stereotype_label not in labels or violence_label not in labels:
                    continue
                yield id_, {"text": text, "misogynous_label": misogynous_label,
                            "objectification_label": objectification_label,
                            "shaming_label": shaming_label,
                            "stereotype_label": stereotype_label,
                            "violence_label": violence_label}
