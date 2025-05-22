import re
import os
import csv
from filelock import FileLock


def extract_strings_between_quotes(input_string):
    return re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', input_string)


def batchify(lst, batch_size):
    return [lst[i:i+batch_size] for i in range(0, len(lst), batch_size)]


def mean_pooling(token_embeddings, mask):
    token_embeddings = token_embeddings.masked_fill(
        ~mask[..., None].bool(), 0.)
    sentence_embeddings = token_embeddings.sum(
        dim=1) / mask.sum(dim=1)[..., None]
    return sentence_embeddings


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]
    return preds, labels


def postprocess_output(output):
    if output.startswith("[Review]:"):
        return output[len("[Review]:"):].strip()
    return output


def write_to_csv(method, metric, value, file_path = "../result.csv"):
    # file_path = "../result.csv"
    lock_path = file_path + ".lock"
    with FileLock(lock_path):
        if os.path.exists(file_path):
            with open(file_path, mode='r', newline='', encoding='utf-8') as file:
                reader = list(csv.reader(file))
        else:
            reader = []
        if not reader:
            reader.append(["method"])
        headers = reader[0]
        methods = [row[0] for row in reader[1:]]
        if metric not in headers:
            headers.append(metric)
            for row in reader[1:]:
                row.append("")
        if method not in methods:
            reader.append([method] + ["" for _ in range(len(headers) - 1)])
        for row in reader:
            while len(row) < len(headers):
                row.append("")
        method_index = methods.index(method) + 1 if method in methods else len(reader) - 1
        metric_index = headers.index(metric)
        reader[method_index][metric_index] = value
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerows(reader)
