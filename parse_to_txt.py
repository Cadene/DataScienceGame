import pandas as pd

train_df = pd.read_csv('./data/train_sample.csv', header=0, escapechar='\\',quotechar="\"",low_memory=False)
test_df = pd.read_csv('./data/test_sample.csv', header=0, escapechar='\\',quotechar="\"",low_memory=False)

cat = train_df["video_category_id"].tolist()
titles = train_df["title"].tolist()
desc = train_df["description"].tolist()


with open("data/train_only_text_simple","w",encoding="utf-8") as f:
    f.write("sentence;category")
    for c,t,d in zip(cat,titles,desc):
        f.write("\"{}\";{}\n".format(str(t)+str(d),c))


id = test_df["id"].tolist()
titles = test_df["title"].tolist()
desc = test_df["description"].tolist()


with open("data/test_only_text_simple","w",encoding="utf-8") as f:
    f.write("id;sentence")
    for c,t,d in zip(id,titles,desc):
        f.write("{},{}\n".format(c,str(t)+str(d)))



