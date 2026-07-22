import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import ndcg_score

def recall(found_docs: list, test_list: list, value: int) -> float:
    counter = 0

    found_docs_first_value = []
    test_set = set(test_list)

    for mytuple in sorted(found_docs, reverse=True):
        found_docs_first_value.append(mytuple[1])

        if len(found_docs_first_value) == value:
            break

    for doc_id in found_docs_first_value:
        if doc_id in test_set:
            counter += 1

    return (counter / len(test_list)) * 100

def precision(found_docs: list, test_list: list, value: int) -> float:
        counter = 0

        found_docs_first_value = []
        test_set = set(test_list)

        for mytuple in sorted(found_docs, reverse=True):
            found_docs_first_value.append(mytuple[1])

            if len(found_docs_first_value) == value:
                break

        for doc_id in found_docs_first_value:
            if doc_id in test_set:
                counter += 1

        return (counter / value) * 100

def create_statistical_columns(df: pd.DataFrame, qrels_dict: dict, doc_id: dict, similarities: np.array) -> pd.DataFrame:
    recall_5_list, recall_10_list = [], []
    precision_5_list, precision_10_list = [], []

    for idx in range(len(df)):
        query_id = df.loc[idx, "Query_ID"]

        similarity = similarities[idx]

        index_of_found_docs = np.where(similarity > 0)[0]

        found_docs = []

        for index in index_of_found_docs:
            found_docs.append((similarity[index], doc_id[index]))

        test_list = qrels_dict[query_id]

        if found_docs:
            recall_5_value = recall(found_docs, test_list, 5)
            recall_10_value = recall(found_docs, test_list, 10)
            precision_5_value = precision(found_docs, test_list, 5)
            precision_10_value = precision(found_docs, test_list, 10)

            recall_5_list.append(recall_5_value)
            recall_10_list.append(recall_10_value)
            precision_5_list.append(precision_5_value)
            precision_10_list.append(precision_10_value)
        else:
            recall_5_list.append(0)
            recall_10_list.append(0)
            precision_5_list.append(0)
            precision_10_list.append(0)

    df["recall_5"] = recall_5_list
    df["recall_10"] = recall_10_list
    df["precision_5"] = precision_5_list
    df["precision_10"] = precision_10_list

    df["f_score_5"] = 2 * df["recall_5"] * df["precision_5"] / (df["recall_5"] + df["precision_5"])
    df["f_score_10"] = 2 * df["recall_10"] * df["precision_10"] / (df["recall_10"] + df["precision_10"])

    df["f_score_5"] = df["f_score_5"].fillna(0)
    df["f_score_10"] = df["f_score_10"].fillna(0)

    return df

def precision_AP(found_docs: list, test_list: list) -> float:
    counter = 0

    test_set = set(test_list)

    for doc in found_docs:
        if doc in test_set:
            counter += 1

    return counter / len(found_docs)

def AP(found_docs: list, test_list: list, app_value: int) -> float:
    found_docs_first_app_value_list = []

    for mytuple in sorted(found_docs, reverse=True):
        found_docs_first_app_value_list.append(mytuple[1])

        if len(found_docs_first_app_value_list) == app_value:
            break

    total = 0

    for i in range(1, app_value + 1):
        if i <= len(found_docs_first_app_value_list):
            precision_k = precision_AP(found_docs_first_app_value_list[:i], test_list)
            total += precision_k * (found_docs_first_app_value_list[i - 1] in set(test_list))

    return total / len(test_list)

def create_AP(df: pd.DataFrame, qrels_dict: dict, doc_id: dict, similarities: np.array) -> pd.DataFrame:
    ap_5_list, ap_10_list = [], []

    for idx in range(len(df)):
        query_id = df.loc[idx, "Query_ID"]

        similarity = similarities[idx]

        index_of_found_docs = np.where(similarity > 0)[0]

        found_docs = []

        for index in index_of_found_docs:
            found_docs.append((similarity[index], doc_id[index]))

        test_list = qrels_dict[query_id]

        if found_docs:
            ap_5_value = AP(found_docs, test_list, 5)
            ap_10_value = AP(found_docs, test_list, 10)

            ap_5_list.append(ap_5_value)
            ap_10_list.append(ap_10_value)
        else:
            ap_10_list.append(0)
            ap_5_list.append(0)

    df["AP_5"] = ap_5_list
    df["AP_10"] = ap_10_list

    df["AP_5"] = df["AP_5"].fillna(0)
    df["AP_10"] = df["AP_10"].fillna(0)

    return df

class Scoredoc:
    def __init__(self, doc_id, score):
        self.doc_id = doc_id
        self.score = score

def create_ndcg(df: pd.DataFrame, doc_id: dict, similarities: np.array, score_doc_dict: dict) -> pd.DataFrame:
    doc_id_score_dict = defaultdict(float)

    ndcg_5_list, ndcg_10_list = [], []

    for idx in range(len(df)):
        query_id = df.loc[idx, "Query_ID"]

        scoredoc_object_list = score_doc_dict[query_id]

        for scoredoc_object in scoredoc_object_list:
            doc_id_score_dict[scoredoc_object.doc_id] = scoredoc_object.score

        similarity = similarities[idx]
        index_of_found_docs = np.where(similarity > 0)[0]

        y_true, y_score = [], []

        for index in index_of_found_docs:
            y_score.append(similarity[index])
            y_true.append(doc_id_score_dict.get(doc_id[index], 0.0))

        if len(y_score) == 0:
            ndcg_5_list.append(0.0)
            ndcg_10_list.append(0.0)
            continue

        if len(y_score) == 1:
            y_true.append(0.0)
            y_score.append(0.0)

        if len(y_true) == len(y_score):
            ndcg_5_list.append(ndcg_score(np.asarray([y_true]), np.asarray([y_score]), k=5))
            ndcg_10_list.append(ndcg_score(np.asarray([y_true]), np.asarray([y_score]), k=10))
        else:
            print("There is a problem with query", query_id)

    df["NDCG_5"] = ndcg_5_list
    df["NDCG_10"] = ndcg_10_list

    df["NDCG_5"] = df["NDCG_5"].fillna(0)
    df["NDCG_10"] = df["NDCG_10"].fillna(0)

    return df

def print_columns(df: pd.DataFrame):
    print(f"recall_5_mean: {df['recall_5'].mean()}")
    print(f"recall_5_std: {df['recall_5'].std()}")
    print(f"recall_5_max: {df['recall_5'].max()}")
    print(f"recall_5_min: {df['recall_5'].min()}")
    print(f"recall_10_mean: {df['recall_10'].mean()}")
    print(f"recall_10_std: {df['recall_10'].std()}")
    print(f"recall_10_max: {df['recall_10'].max()}")
    print(f"recall_10_min: {df['recall_10'].min()}")
    print(f"precision_5_mean: {df['precision_5'].mean()}")
    print(f"precision_5_std: {df['precision_5'].std()}")
    print(f"precision_5_max: {df['precision_5'].max()}")
    print(f"precision_5_min: {df['precision_5'].min()}")
    print(f"precision_10_mean: {df['precision_10'].mean()}")
    print(f"precision_10_std: {df['precision_10'].std()}")
    print(f"precision_10_max: {df['precision_10'].max()}")
    print(f"precision_10_min: {df['precision_10'].min()}")
    print(f"f_score_5_mean: {df['f_score_5'].mean()}")
    print(f"f_score_5_std: {df['f_score_5'].std()}")
    print(f"f_score_5_max: {df['f_score_5'].max()}")
    print(f"f_score_5_min: {df['f_score_5'].min()}")
    print(f"f_score_10_mean: {df['f_score_10'].mean()}")
    print(f"f_score_10_std: {df['f_score_10'].std()}")
    print(f"f_score_10_max: {df['f_score_10'].max()}")
    print(f"f_score_10_min: {df['f_score_10'].min()}")
    print(f"MAP_5: {df['AP_5'].mean()}")
    print(f"MAP_10: {df['AP_10'].mean()}")
    print(f"NDCG_5_mean: {df['NDCG_5'].mean()}")
    print(f"NDCG_5_std: {df['NDCG_5'].std()}")
    print(f"NDCG_5_max: {df['NDCG_5'].max()}")
    print(f"NDCG_5_min: {df['NDCG_5'].min()}")
    print(f"NDCG_10_mean: {df['NDCG_10'].mean()}")
    print(f"NDCG_10_std: {df['NDCG_10'].std()}")
    print(f"NDCG_10_max: {df['NDCG_10'].max()}")
    print(f"NDCG_10_min: {df['NDCG_10'].min()}")