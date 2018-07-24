import models.concept_recognizer.build
import models.concept_recognizer.train
import models.concept_recognizer.test

def main():
    models.concept_recognizer.build.main()
    models.concept_recognizer.train.main()
    models.concept_recognizer.test.main()


if __name__ == '__main__':
    main()