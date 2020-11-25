import pandas as pd


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels, x0=None, y0=None, x1=None, y1=None):
        """Constructs a InputExample.
        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
            x0: (Optional) list. The list of x0 coordinates for each word.
            y0: (Optional) list. The list of y0 coordinates for each word.
            x1: (Optional) list. The list of x1 coordinates for each word.
            y1: (Optional) list. The list of y1 coordinates for each word.
        """
        self.guid = guid
        self.words = words
        self.labels = labels
        if x0 is None:
            self.bboxes = None
        else:
            self.bboxes = [[a, b, c, d] for a, b, c, d in zip(x0, y0, x1, y1)]

    def __str__(self) -> str:
        return '{},{},{}'.format(self.guid, self.words, self.labels)


if __name__ == "__main__":
    train_data = [
        [0, "Simple", "B-MISC"],
        [0, "Transformers", "I-MISC"],
        [0, "started", "O"],
        [1, "with", "O"],
        [0, "text", "O"],
        [0, "classification", "B-MISC"],
        [1, "Simple", "B-MISC"],
        [1, "Transformers", "I-MISC"],
        [1, "can", "O"],
        [1, "now", "O"],
        [1, "perform", "O"],
        [1, "NER", "B-MISC"],
    ]
    data = pd.DataFrame(train_data, columns=[
        "sentence_id", "words", "labels"])
    print(data.groupby(["sentence_id"]))
    tmp = [
        InputExample(guid=sentence_id, words=sentence_df["words"].tolist(
        ), labels=sentence_df["labels"].tolist(),)
        for sentence_id, sentence_df in data.groupby(["sentence_id"])
    ]
    for t in tmp:
        print(t)
    # for sentence_id, sentence_df in train_df.groupby(["sentence_id"]):
    #     print('--',sentence_id, sentence_df)
