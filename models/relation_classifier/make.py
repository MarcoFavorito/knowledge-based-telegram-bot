import models.relation_classifier.build
import models.relation_classifier.train
import models.relation_classifier.test

def main():
    models.relation_classifier.build.main()
    models.relation_classifier.train.main()
    models.relation_classifier.test.main()
if __name__ == '__main__':
    main()
