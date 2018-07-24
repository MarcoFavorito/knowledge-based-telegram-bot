import models
import tensorflow as tf

def main():
    """
    build the data and train the model
    output paths:
        models.name_of_the_model.(model|data)
    """

    from models.relation_classifier import build,train
    from models.concept_recognizer import  build,train
    from models.answer_generation import  build,train
    from models.answer_concept_recognizer  import build,train

    # to limit the memory occupied by tensorflow
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    print("relation classifier")
    models.relation_classifier.build.main()
    models.relation_classifier.train.main()
    print("concept recognizer")
    models.concept_recognizer.build.main()
    models.concept_recognizer.train.main()
    print("answer generator")
    models.answer_generation.build.main()
    models.answer_generation.train.main()
    print("answer concept recognizer")
    models.answer_concept_recognizer.build.main()
    models.answer_concept_recognizer.train.main()


if __name__ == '__main__':
    main()
