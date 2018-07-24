from collections import Counter

from models.answer_generation import read_file, DATA_DIR, DATA_TEST, MODEL_PATH, read_relations
from models.answer_generation.AnswerGenerator import AnswerGenerator
import re
import string

def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1

def main():
    X_test, Y_test = read_file(DATA_TEST)
    relations = read_relations(DATA_TEST)

    rel2f1_scores = {r:[] for r in relations}

    ag = AnswerGenerator.load(MODEL_PATH)
    f1_scores = []
    for x,y,r in zip(X_test, Y_test, relations):
        answer = ag.evaluate(x)
        answer = " ".join(answer[0][:-1])
        cur_f1_score = f1_score(answer, y)
        f1_scores.append(cur_f1_score)

        rel2f1_scores[r].append(cur_f1_score)
        print()
        print(x)
        print(answer)
        print(y)
        print(cur_f1_score)

    print("avg f1 score:", sum(f1_scores)/len(f1_scores))

    print("f1 for relation:")
    for r, f1_list in rel2f1_scores.items():
        print(r,": ", sum(f1_list)/len(f1_list))



if __name__ == '__main__':
    main()