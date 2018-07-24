cd kbs/data


rm -R chatbot_maps
rm -R patterns
rm db_chatbot.dump.txt
rm cleaned_patterns.tsv
rm relations

cd ../..

cd kbs

rm kb_mirror.db

cd ..

cd  models

rm -R answer_concept_recognizer/data/*
rm -R answer_concept_recognizer/model/*
rm -R answer_generation/data/*
rm -R answer_generation/model/*
rm -R concept_recognizer/data/*
rm -R concept_recognizer/model/*
rm -R relation_classifier/data/*
rm -R relation_classifier/model/*

cd ..