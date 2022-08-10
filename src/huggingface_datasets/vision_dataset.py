import csv
import cv2
import datasets
from datasets.tasks import ImageClassification, TextClassification
import os


_DESCRIPTION = """MAMI"""

_CITATION = """MAMI"""

_TRAIN_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/train.csv"
_TEST_DOWNLOAD_URL = "https://raw.githubusercontent.com/mhjabreel/CharCnn_Keras/master/data/ag_news_csv/test.csv"


class VisionDataset(datasets.GeneratorBasedBuilder):
    """AG News topic classification dataset."""

    def _info(self):
        """

        :return:
        """
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "img": datasets.Value("binary"),
                    "label": datasets.features.ClassLabel(num_classes=2),
                }
            ),
            homepage="http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html",
            citation=_CITATION,
            task_templates=[ImageClassification(label_column="label")],
        )

    def _split_generators(self, dl_manager):
        """

        :param dl_manager:
        :return:
        """
        train_path = "data/TRAINING/training_no_bad_lines_train.csv"
        val_path = "data/TRAINING/training_no_bad_lines_val.csv"
        test_path = "data/TRAINING/training_no_bad_lines_test.csv"
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepath": val_path}),
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
                try:
                    label = int(row[4])
                except ValueError:
                    print("error")
                    continue

                img_path = row[3]
                img = cv2.imread(os.path.join("data/TRAINING/" + img_path))
                # print(row, label, img_path)
                if img is None:
                    continue
                if label not in labels:
                    continue
                # print(img.shape)
                yield id_, {"img": img, "label": label}
