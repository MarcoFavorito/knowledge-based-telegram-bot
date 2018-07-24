import models.answer_generation.build
import models.answer_generation.train
import models.answer_generation.test

def main():
    models.answer_generation.build.main()
    models.answer_generation.train.main()
    models.answer_generation.test.main()


if __name__ == '__main__':
    main()