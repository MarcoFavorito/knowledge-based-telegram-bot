# Knowledge Based Telegram-Bot 

## Installation
Where not specified, I assume that the working directory of the terminal is the root of the project.
### Environment
Install all the needed packages with `requirements.txt` file:

    pip3 install -r requirements.txt

Then download the spaCy model:

    python -m spacy download en_core_web_md
    
And [install Torch](http://pytorch.org/). For me the following commands worked: 

    pip3 install http://download.pytorch.org/whl/cu80/torch-0.2.0.post3-cp35-cp35m-manylinux1_x86_64.whl
    pip3 install torchvision
    
You need to install also SQLite since the database uses that engine:
```
sudo apt-get install sqlite3 libsqlite3-dev
```

I do not ensure you will have all the dependencies covered, caused by some machine specific package, or something that I've missed. Please be patient, I did my best.

### Unpack data
First of all, get the following data and copy them into `kbs/data`:

- [db_chatbot.dump.txt](https://drive.google.com/file/d/0B9vexhdMNG0qdmVLeXo1X0htSHc/view?usp=sharing)
- [patterns.zip](https://drive.google.com/file/d/1oCEfT1Omps1d74Jy6ojZZdw10PmsByeA/view?usp=sharing)    
- [chatbot_maps.tar.gz](https://drive.google.com/file/d/1NuzL_KEOt6E4fsZFuX7Dnx9Fm7iMyMzB/view?usp=sharing)

Then execute the following commands (it needs 5-10 minutes):
    
    . setup.sh
    python make_database.py #answer "yes" to the question about dropping tables
    
Now you should be able to train all the models by yourself:
    
    python make_models.py
    
to clean up:

    . clean.sh

###Run
to run the project:

    python main.py
    
The id name of the telegram bot is: `MF_nlp_chatbot`

## Project file structure

### Important
- `kbs/`: the directory which contains the data provided by you, the mirror db and the modules needed for interact with it.
    - `ApiManager.py`: for interact with the KB
    - `DBManager.py`: a singleton class for allow easy interaction with db
    - `models.py`: ORM definitions (SQLAlchemy)
    - `kb_mirror.db`: the db, can be overwritten and rebuilt with `/make_database.py` 
- `logs/`: the most useful thing I did in my life. The software is highly auditable from that file. I suggest you to check it.
- `models/`: the package which contains all the machine learning models of the system and some related utils.
- `workflow/`: the core modules for manage the bot workflow. The main classes are `WorkflowManager` and the abstract class `Job`, whose implementations represent each workflow step.
- `main.py`: it is the main script; execute it to start the bot.
- `make_database.py`: the script that builds all the needed data and populates the database. 
- `config.py`: file for some global configuration (please, paths should not be changed too easily)
- `constant.py`: some global constant useful for the program

### Less important

- `exceptions/`: single-module package which contains some user-defined exceptions.
- `utils/`: misc utils (actually there are implemented very useful methods for the software)
- `handlers/handles.py`: the module which contains the main handler for messages. It is only a dispatcher for the right chat_id

If you're lucky, you will find README in package you're interested in with more details.

## Execution path:

- the `main.py` initializes the `handlers/handler.py` which runs as thread
- the handler dispatch by chat_id the message to the correct `WorkflowManager` object
- the Workflow manager, depending on the state in which it is:
    - dispatch the message to the scheduled `Job` object (defined in `workflow/jobs`);
    - waits the end of execution of the called `Job` (i.e. `Job.__call__()`), which returns the new state of the `WorkflowManager`.
- The abstract class overrides the method `__call__()` which make it callable. The execution consists of:
    - `validate_input(input_text)`, for validate the input of the Job (eventually overrided by the `Job` implementations)
    - `the_job(input_text)`, the actual job. Here the messages are processed, the models are called to predict something etc.
    - `get_new_state()` has to be overridden in every Job implementation
    - Sometimes there is the need to share parameters among Jobs: this is done with the `Context` class, instantiated in the Workflow manager.
    
## Workflow
The `Job` classes are defined in `workflow/jobs`.
Follows the workflow:
- StartJob: The entry point of the application. It expects a "/start" message to begin to chat.
- AskDomainJob: Ask the domain to the user
- ReceiveDomainJob: receive and store the Domain. Schedule the AskQueryEnrichSwitchJob
- AskQueryEnrichSwitchJob: ask to the user if he wants to aks a question or let the bot to ask a question.
- SwitcherQEJob: the job who dispatches the correct Job according to the chosen mode:
    - if in Querying mode, the workflow is:
        - QueryProcessingJob:
            - Predict the relation (`models.relation_classifier`)
            - Detect the concepts into the question (`models.concept_recognizer`)
        - QueryAnswerGenerationJob: 
            - Extract the correct concepts
            - Query the database to answer correctly `utils/workflow/ConceptSelector`
            - Call the AnswerGenerator (`models.answer_generation`)
    - if in Enriching mode, the workflow is:
        - EnrichingQueryGenerationJob:
            - Chose the concept about we want to know more;
            - Chose the relation which has less occurrences for the given concept;
            - Retrieve a random pattern from the database and replace the concept mentions into the query;
        - EnrichingAnswerProcessingJob:
            - Detect the concepts into the answer (`models.answer_concept_recognizer`)
            - Update the knowledge base.
- FinishWorkflowJob: reschedule the AskDomainJob

