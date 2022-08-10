import datasets
from datasets.tasks import TextClassification
import csv

_DESCRIPTION = """MAMI"""

_CITATION = """MAMI"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"


class ObjectificationDataset(datasets.GeneratorBasedBuilder):
    """

    """

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
                    "objectification_label": datasets.features.ClassLabel(
                        names=labels),
                }
            ),
            homepage="http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
            citation=_CITATION,
            task_templates=[
                TextClassification(text_column="text", label_column="objectification_label",
                                   labels=labels)],
        )

    def _split_generators(self, dl_manager):
        """

        :param dl_manager:
        :return:
        """
        train_path = "data/TRAINING_csvs/training_no_bad_lines_train_train.csv"
        test_path = "data/TRAINING_csvs/training_no_bad_lines_train_val.csv"
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
                if row[5] == "objectification":
                    continue
                label = int(row[5])
                text = row[-1]
                if label not in labels:
                    continue
                yield id_, {"text": text, "objectification_label": label}
