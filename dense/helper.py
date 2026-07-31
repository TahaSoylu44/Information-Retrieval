from collections import defaultdict
from sklearn.metrics import ndcg_score
import numpy as np
import pandas as pd

def recall(found_list: list, test_list: list) -> float:
    counter = 0

    test_set = set(test_list)

    for doc_id in found_list:
        if doc_id in test_set:
            counter += 1

    return (counter / len(test_list)) * 100

def precision(found_list: list, test_list: list) -> float:
    if len(found_list) == 0:
        return 0.0

    counter = 0

    test_set = set(test_list)

    for doc_id in found_list:
        if doc_id in test_set:
            counter += 1

    return (counter / len(found_list)) * 100

def precision_AP(found_docs: list, test_list: list) -> float:
    counter = 0

    test_set = set(test_list)

    for doc in found_docs:
        if doc in test_set:
            counter += 1

    return counter / len(found_docs)


def AP(found_docs: list, test_list: list, value: int) -> float:
    total = 0

    for i in range(1, value + 1):
        if i < len(found_docs):
            precision_k = precision_AP(found_docs[:i], test_list)
            total += precision_k * (found_docs[i - 1] in set(test_list))

    return total / len(test_list)

def get_recall_list(results: dict, query_ids: list, qrels_dict: dict) -> list:
    recall_list = []

    for idx in range(len(query_ids)):
        found_list = results['ids'][idx]
        test_list = qrels_dict[query_ids[idx]]

        recall_list.append(recall(found_list, test_list))

    return recall_list

def get_precision_list(results: dict, query_ids: list, qrels_dict: dict) -> list:
    precision_list = []

    for idx in range(len(query_ids)):
        found_list = results['ids'][idx]
        test_list = qrels_dict[query_ids[idx]]

        precision_list.append(precision(found_list, test_list))

    return precision_list

def get_ap_list(results: dict, value: int, query_ids: list, qrels_dict: dict) -> list:
    ap_list = []

    for idx in range(len(query_ids)):
        found_list = results['ids'][idx]
        test_list = qrels_dict[query_ids[idx]]
        ap_list.append(AP(found_list, test_list, value))

    return ap_list

def get_score_list(results: dict) -> list:
    scores = []

    for score_list in results['distances']:
        sub_scores = []

        for score in score_list:
            sub_scores.append(1 - score)

        scores.append(sub_scores)

    return scores

class Scoredoc:
    def __init__(self, doc_id, score):
        self.doc_id = doc_id
        self.score = score

def get_most_dict(results: dict, scores: list, value: int, query_ids: list) -> dict:
    most_dict_with_scores = defaultdict(list)

    for idx in range(len(query_ids)):
        for i in range(value):
            doc_id = results['ids'][idx][i]
            score = scores[idx][i]

            most_dict_with_scores[query_ids[idx]].append((doc_id, score))

    return most_dict_with_scores

def get_ndcg_list(most_dict: dict, value: int, query_ids: list, score_doc_dict: dict) -> list:
    doc_id_score_dict = defaultdict(float)
    ndcg_list = []

    for query_id in query_ids:
        scoredoc_object_list = score_doc_dict[query_id]

        for scoreddoc_object in scoredoc_object_list:
            doc_id_score_dict[scoreddoc_object.doc_id] = scoreddoc_object.score

        model_score_tuple_list = most_dict[query_id]
        y_score, y_true = [], []

        for mytuple in model_score_tuple_list:
            doc_id = mytuple[0]
            score = mytuple[1]

            y_score.append(score)
            y_true.append(doc_id_score_dict[doc_id])

        if len(y_score) == 0:
            ndcg_list.append(0.0)
            continue

        if len(y_score) == 1:
            y_true.append(0.0)
            y_score.append(0.0)

        if len(y_true) == len(y_score):
            ndcg_list.append(ndcg_score(np.asarray([y_true]), np.asarray([y_score]), k=value))
        else:
            print("There is a problem with query", query_id)

    return ndcg_list

def base_df(
        query_ids: list,
        recall_5: list,
        precision_5: list,
        AP_5: list,
        NDCG_5: list,
        recall_10: list,
        precision_10: list,
        AP_10: list,
        NDCG_10: list
) -> pd.DataFrame:

    df = pd.DataFrame({
        "Query_ID": query_ids,
        "recall_5": recall_5,
        "precision_5": precision_5,
        "AP_5": AP_5,
        "NDCG_5": NDCG_5,
        "recall_10": recall_10,
        "precision_10": precision_10,
        "AP_10": AP_10,
        "NDCG_10": NDCG_10
    })

    df["f_score_5"] = 2 * df["recall_5"] * df["precision_5"] / (df["recall_5"] + df["precision_5"])
    df["f_score_10"] = 2 * df["recall_10"] * df["precision_10"] / (df["recall_10"] + df["precision_10"])

    df["f_score_5"] = df["f_score_5"].fillna(0)
    df["f_score_10"] = df["f_score_10"].fillna(0)

    return df

def create_parquet_df(method_name: str, df: pd.DataFrame) -> pd.DataFrame:
    mydict = {
        "Method": method_name,
        "recall_5_mean": df["recall_5"].mean(),
        "recall_5_std": df["recall_5"].std(),
        "recall_5_max": df["recall_5"].max(),
        "recall_5_min": df["recall_5"].min(),
        "recall_10_mean": df["recall_10"].mean(),
        "recall_10_std": df["recall_10"].std(),
        "recall_10_max": df["recall_10"].max(),
        "recall_10_min": df["recall_10"].min(),
        "precision_5_mean": df["precision_5"].mean(),
        "precision_5_std": df["precision_5"].std(),
        "precision_5_max": df["precision_5"].max(),
        "precision_5_min": df["precision_5"].min(),
        "precision_10_mean": df["precision_10"].mean(),
        "precision_10_std": df["precision_10"].std(),
        "precision_10_max": df["precision_10"].max(),
        "precision_10_min": df["precision_10"].min(),
        "f_score_5_mean": df["f_score_5"].mean(),
        "f_score_5_std": df["f_score_5"].std(),
        "f_score_5_max": df["f_score_5"].max(),
        "f_score_5_min": df["f_score_5"].min(),
        "f_score_10_mean": df["f_score_10"].mean(),
        "f_score_10_std": df["f_score_10"].std(),
        "f_score_10_max": df["f_score_10"].max(),
        "f_score_10_min": df["f_score_10"].min(),
        "MAP_5": df["AP_5"].mean(),
        "MAP_10": df["AP_10"].mean(),
        "NDCG_5_mean": df["NDCG_5"].mean(),
        "NDCG_5_std": df["NDCG_5"].std(),
        "NDCG_5_max": df["NDCG_5"].max(),
        "NDCG_5_min": df["NDCG_5"].min(),
        "NDCG_10_mean": df["NDCG_10"].mean(),
        "NDCG_10_std": df["NDCG_10"].std(),
        "NDCG_10_max": df["NDCG_10"].max(),
        "NDCG_10_min": df["NDCG_10"].min()
    }

    df_parquet = pd.DataFrame(mydict, index=[0])

    return df_parquet

def pipeline(results_5: dict, results_10: dict, query_ids: list, method_name: str, qrels_dict: dict, score_doc_dict: dict) -> pd.DataFrame:
    recall_5_list = get_recall_list(results_5, query_ids, qrels_dict)
    precision_5_list = get_precision_list(results_5, query_ids, qrels_dict)
    AP_5_list = get_ap_list(results_5, 5, query_ids, qrels_dict)

    score_5_list = get_score_list(results_5)
    most_5_dict = get_most_dict(results_5, score_5_list, 5, query_ids)
    ndcg_5_list = get_ndcg_list(most_5_dict, 5, query_ids, score_doc_dict)

    recall_10_list = get_recall_list(results_10, query_ids, qrels_dict)
    precision_10_list = get_precision_list(results_10, query_ids, qrels_dict)
    AP_10_list = get_ap_list(results_10, 10, query_ids, qrels_dict)

    score_10_list = get_score_list(results_10)
    most_10_dict = get_most_dict(results_10, score_10_list, 10, query_ids)
    ndcg_10_list = get_ndcg_list(most_10_dict, 10, query_ids, score_doc_dict)

    df = base_df(
        query_ids,
        recall_5_list,
        precision_5_list,
        AP_5_list,
        ndcg_5_list,
        recall_10_list,
        precision_10_list,
        AP_10_list,
        ndcg_10_list
    )

    print(df)

    df_parquet = create_parquet_df(method_name, df)
    print(df_parquet)

    return df_parquet