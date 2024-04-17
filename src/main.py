import itertools
import json
import re

import evaluate
import numpy as np
import pandas as pd
import sklearn
import torch
import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


class Trainer(Trainer):
    loss_fn = torch.nn.CrossEntropyLoss()

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss = self.loss_fn(
            logits.view(-1, self.model.config.num_labels), labels.view(-1)
        )
        return (loss, outputs) if return_outputs else loss


class Dataset(torch.utils.data.Dataset):
    def __init__(self, segmented_words):
        self._segmented_words = segmented_words
        self._items = {}

        for w in segmented_words:
            segments = w.split("@")
            word = "".join(segments)
            # in between characters, there are len-1 posibilities
            segment_lens = [len(s) for s in segments]
            hyphen_pos = set(itertools.accumulate(segment_lens[:-1]))

            for i in range(1, len(word)):
                text = f"{word} {word_separator_token} {word[:i]}{morph_boundary_token}{word[i:]}"
                label = int(i in hyphen_pos)
                value_dict = {"text": text, "label": label}
                value_dict.update(tokenizer(text))
                self._items[len(self._items)] = value_dict
        super()

    def get_class_weights(self):
        weights = sklearn.utils.class_weight.compute_class_weight(
            "balanced", classes=[0, 1], y=[i["label"] for i in self._items.values()]
        )
        return weights

    def __len__(self):
        return len(self._items)

    def __getitem__(self, idx):
        return self._items[idx]


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    return clf_metrics.compute(predictions=predictions, references=labels)


def eval_word(ground, predicted):
    tp = fp = fn = 0
    gi = pi = 0
    while gi < len(ground) and pi < len(predicted):
        g = ground[gi]
        p = predicted[pi]
        if g == p:
            if g == "@":
                tp += 1
            gi += 1
            pi += 1
        elif g == "@":
            fn += 1
            gi += 1
        elif p == "@":
            fp += 1
            pi += 1
        else:
            assert False, (ground, predicted)
    assert gi == len(ground) and pi == len(predicted)
    return tp, fp, fn


def load_lang_data(lang):
    if lang in ("swati", "xhosa", "zulu", "ndebele"):
        fname_px = f"/data/{lang}.clean"
        data_train = pd.read_table(
            f"{fname_px}.train.conll", header=None, sep=r"\s\|\s", engine="python"
        )
        data_train[1] = data_train[1].apply(lambda s: re.sub("-", "@", s))
        data_dev = pd.read_table(
            f"{fname_px}.dev.conll", header=None, sep=r"\s\|\s", engine="python"
        )
        data_dev[1] = data_dev[1].apply(lambda s: re.sub("-", "@", s))
        data_test = pd.read_table(
            f"{fname_px}.test.conll", header=None, sep=r"\s\|\s", engine="python"
        )
        data_test[1] = data_test[1].apply(lambda s: re.sub("-", "@", s))
    elif lang in ("eng", "fin", "tur"):
        fname_px = f"/data/{lang}"
        data_train = pd.read_csv(f"{fname_px}_train.segmentation.csv", header=None)
        data_dev = pd.read_csv(f"{fname_px}_val.segmentation.csv", header=None)
        data_test = pd.read_csv(f"{fname_px}_test.segmentation.csv", header=None)
    return data_train, data_dev, data_test


def _esc_spec(c):
    if c == " ":
        return "_"
    return c


def test_model(model, test_data):
    TP = FP = FN = 0
    ACC = 0
    predictions = []

    model.eval()
    for ground in tqdm.tqdm(test_data):
        word = "".join(ground.split("@"))
        ww = []
        wit = iter(word)
        candidates = [
            f"{word} {word_separator_token} {word[:i]}{morph_boundary_token}{word[i:]}"
            for i in range(1, len(word))
        ]
        tokenized = tokenizer(candidates, return_tensors="pt", padding=True).to("cuda")
        with torch.no_grad():
            res = model(**tokenized).logits.argmax(1).cpu().tolist()
        for i in res:
            c = _esc_spec(next(wit))
            ww.append(c)
            if i == 1:
                ww.append("@")
        c = _esc_spec(next(wit))
        ww.append(c)
        predicted = "".join(ww)
        predictions.append(predicted)
        if ground == predicted:
            ACC += 1
        tp, fp, fn = eval_word(ground, predicted)
        TP += tp
        FP += fp
        FN += fn
    return ACC, TP, FP, FN, predictions


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("cis-lmu/glot500-base")
    word_separator_token = "<__word-separator>"
    morph_boundary_token = "<__morph-boundary>"
    assert tokenizer.add_tokens([morph_boundary_token, word_separator_token])

    for lang in ("eng", "fin", "tur", "swati", "xhosa", "zulu", "ndebele"):
        model = AutoModelForSequenceClassification.from_pretrained(
            "cis-lmu/glot500-base", num_labels=2
        )
        embs = model.resize_token_embeddings(len(tokenizer))
        embs.weight.data[-1] = 0  # embs.weight[238].detach()
        embs.weight.data[-2] = 0  # embs.weight[2203].detach()
        data_train, data_dev, data_test = load_lang_data(lang)

        d_train = Dataset(data_train[1].tolist())
        d_val = Dataset(data_dev[1].tolist())

        accuracy = evaluate.load("accuracy")
        precision = evaluate.load("precision")
        recall = evaluate.load("recall")
        f1 = evaluate.load("f1")
        clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        training_args = TrainingArguments(
            output_dir="/out/trained-model",
            learning_rate=2e-5,
            per_device_train_batch_size=256,
            per_device_eval_batch_size=256,
            num_train_epochs=30,
            logging_steps=250,
            weight_decay=0.01,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            metric_for_best_model="f1",
            greater_is_better=True,
            load_best_model_at_end=True,
            warmup_steps=20,
        )

        model.cuda()
        model.train()
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=d_train,
            eval_dataset=d_val,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
        )

        class_weights = d_train.get_class_weights()
        trainer.loss_fn = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(class_weights.astype("float32"))
        ).to(model.device)

        trainer.train()
        model.eval()

        ACC, TP, FP, FN, predictions = test_model(model, data_test[1].tolist())

        predictions = [re.sub("@", " ", p) for p in predictions]
        df = pd.DataFrame({"word": data_test[0].tolist(), "predictions": predictions})
        df.to_csv(f"/out/llm-segm/{lang}.pred", header=None, index=False, sep="\t")

        P = TP / (TP + FP)
        R = TP / (TP + FN)
        F1 = 2 * P * R / (P + R)

        with open(f"/out/llm-segm/{lang}.json", "w") as results_f:
            results_map = {
                "ACC": np.round(100 * ACC / len(data_test), 2),
                "Prec": np.round(100 * P, 2),
                "Rcl": np.round(100 * R, 2),
                "F1": np.round(100 * F1, 2),
                "train-len": len(data_train),
                "dev-len": len(data_dev),
                "test-len": len(data_test),
                "LANG": lang,
            }
            json.dump(results_map, results_f, indent=4)
