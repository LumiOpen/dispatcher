import pandas as pd
import sys

MAX_SCORE = 5
VALID_SCORE = 4
TRAIN_SIZE = 30000
TEST_SIZE = 1000

def check_zero_score(row):
    has_zero_score = False
    for message in row['messages']:
        if 'score' in message and message['score'] == 0:
            has_zero_score = True
    return has_zero_score

def check_perfect_score(row):
    has_perfect_score = False
    for message in row['messages']:
        if 'score' in message and message['score'] == MAX_SCORE:
            has_perfect_score = True
        else:
            has_perfect_score = False
    return has_perfect_score

def check_valid_score(row):
    has_valid_score = False
    for message in row['messages']:
        if 'score' in message and message['score'] >= VALID_SCORE:
            has_valid_score = True
        else:
            has_valid_score = False
    return has_valid_score

def fiilter_data(filepath):
    print(f"Processing file: {filepath}")
    df = pd.read_json(filepath, lines=True)
    df = df[df['__ERROR__'].isnull()==True]
    df['perfect_score'] = df.apply(check_perfect_score, axis=1)
    df['valid_score'] = df.apply(check_valid_score, axis=1)

    df = df[df['valid_score'] == True]
    print(f"Rows with valid scores: {len(df)}")
    print(f"Turn distribution\n{df['turns'].value_counts()}")

    df = df[['messages']]
    df["messages"] = df["messages"].apply(
        lambda msg_list: [{k: v for k, v in msg.items() if k != "score" and k!= "category"} for msg in msg_list]
    )
    df_train = df[: TRAIN_SIZE]
    df_test = df[TRAIN_SIZE : TRAIN_SIZE+TEST_SIZE]
    df_train.to_json("train.jsonl", orient='records', lines=True, force_ascii=False)
    df_test.to_json("test.jsonl", orient='records', lines=True, force_ascii=False)
    print("Train and test datasets created successfully.")

if __name__ == "__main__":
    if  len(sys.argv) < 1:
        print("Usage: python prepare_final_data.py <path_to_file.jsonl>")
    else:
        filepath = sys.argv[1]
        fiilter_data(filepath)
