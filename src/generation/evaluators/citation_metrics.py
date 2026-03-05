def citation_score(y_true: list[list[str]], y_pred: list[list[str]]) -> tuple[list[float], list[float], list[float]]:
  assert len(y_true) == len(y_pred), "y_true and y_pred must have the same length"
  precision = []
  recall = []
  f1 = []
  for i_true, i_pred in zip(y_true, y_pred):
    i_pred = set(i_pred)
    i_true = set(i_true)

    n_pred = len(i_pred)
    n_true = len(i_true)

    if n_pred == 0 and n_true == 0:
      precision.append(1.0); recall.append(1.0); f1.append(1.0)
      continue

    tp = len(i_pred & i_true)


    i_precision = tp/n_pred if n_pred != 0 else 0.0
    i_recall = tp/n_true if n_true != 0 else 0.0
    denom = i_precision + i_recall

    i_f1 = 2 * i_precision * i_recall / denom if denom != 0 else 0.0

    precision.append(i_precision)
    recall.append(i_recall)
    f1.append(i_f1)
  return precision, recall, f1
