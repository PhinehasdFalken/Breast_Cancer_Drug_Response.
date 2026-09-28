import pandas as pd

LABELS = {0: "Sensitive", 1: "Intermediate", 2: "Resistant"}


def rank_normalize(expr: pd.DataFrame) -> pd.DataFrame:
    """
    Turn each sample's gene values into positions between 0 and 1.

    Rows are samples, columns are genes. Within each row, the lowest gene 
    becomes 0 and the highest becomes 1. Because every sample is scaled on 
    its own, this works the same for one patient or a thousand cell lines, 
    and it hides differences in scale between lab machines
    """
    ranks = expr.rank(axis=1, method="average")
    return (ranks - 1) / (expr.shape[1] - 1)
