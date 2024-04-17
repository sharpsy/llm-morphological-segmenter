import pandas as pd


def _format_single_segmentation(s):
    res = []
    for ss in s.split():
        ss = ss.split(":")[0]
        if ss not in "~":
            res.append(ss)
    return "@".join(res)


def extract_single_segmentation(s):
    s = s.split(",")[0]
    return _format_single_segmentation(s)


def extract_all_segmentations(s):
    res = []
    for ss in s.split(","):
        res.append(_format_single_segmentation(ss))
    return ",".join(res)


df = pd.read_csv(
    "/data/goldstd_trainset.segmentation.fin",
    sep="\t",
    header=None,
    encoding="latin-1",
)

dfc = pd.read_csv(
    "/data/goldstd_combined.segmentation.fin",
    sep="\t",
    header=None,
    encoding="latin-1",
)
df_test = dfc[~dfc[0].isin(df[0])]
df_test[1] = df_test[1].apply(extract_all_segmentations)

with open("/data/fin_test.segmentation.csv", "w") as f:
    f.writelines((df_test[0] + "," + df_test[1] + "\n").tolist())

df = df.sample(frac=1.0)
df[1] = df[1].apply(extract_single_segmentation)
df.iloc[:100].to_csv("/data/fin_val.segmentation.csv", header=None, index=None)
df.iloc[100:].to_csv("/data/fin_train.segmentation.csv", header=None, index=None)

##

df = pd.read_csv("/data/goldstd_trainset.segmentation.tur", sep="\t", header=None)
dfc = pd.read_csv("/data/goldstd_combined.segmentation.tur", sep="\t", header=None)
dfc = dfc[~dfc[0].isin(df[0])]
dfc[1] = dfc[1].apply(extract_all_segmentations)

with open("/data/tur_test.segmentation.csv", "w") as f:
    f.writelines((dfc[0] + "," + dfc[1] + "\n").tolist())

df = df.sample(frac=1.0)
df[1] = df[1].apply(extract_single_segmentation)
df.iloc[:100].to_csv("/data/tur_val.segmentation.csv", header=None, index=None)
df.iloc[100:].to_csv("/data/tur_train.segmentation.csv", header=None, index=None)

##

df = pd.read_csv("/data/goldstd_trainset.segmentation.eng", sep="\t", header=None)
dfc = pd.read_csv("/data/goldstd_combined.segmentation.eng", sep="\t", header=None)
dfc = dfc[~dfc[0].isin(df[0])]
dfc[1] = dfc[1].apply(extract_all_segmentations)

with open("/data/eng_test.segmentation.csv", "w") as f:
    f.writelines((dfc[0] + "," + dfc[1] + "\n").tolist())
df = df.sample(frac=1.0)
df[1] = df[1].apply(extract_single_segmentation)
df.iloc[:100].to_csv("/data/eng_val.segmentation.csv", header=None, index=None)
df.iloc[100:].to_csv("/data/eng_train.segmentation.csv", header=None, index=None)
