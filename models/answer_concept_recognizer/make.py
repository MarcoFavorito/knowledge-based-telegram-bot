import models.answer_concept_recognizer.build
import models.answer_concept_recognizer.train
import models.answer_concept_recognizer.test


def main():
    models.answer_concept_recognizer.build.main()
    models.answer_concept_recognizer.train.main()
    models.answer_concept_recognizer.test.main()


if __name__ == '__main__':
    main()