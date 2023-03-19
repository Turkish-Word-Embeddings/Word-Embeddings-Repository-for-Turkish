This folder contains a package for evaluating nlp models. Currently, it is covering gensim models but the package can be extended to cover more models.

## Scripts

There are two scripts:

| Script | Use Case |
| - | - |
| `evaluate.py` | Loads and runs a model on the given tasks. Generates an artifact for every task (every folder) which contains the score of the model on each file in the folder |
| `populate_task_metadata.py` | Choose a folder and create a metadata file for it. Metadata file contains a lookup table of `file -> score`. Goal is to store reference statistics of tasks for generating latex tables |

## Usage of `evaluate.py`

See the `evaluate.py` for the available parameters. Here is an example of calling `evaluate.py`:

```
python evaluate.py -m "../tmp/model/turkish-word2vec-binary.model" -af ../tasks/analogy -af "../tasks/analogy/test file"
```

Note that multiple analogy folders can be passed at the same time.

If the function is being called for the first time with this particular model instance, user will be prompted to provide information about how the model loaded. In this case, the model is a word2vec binary, which needs to be loaded with the `Word2VecWrapper.from_keyed_vectors` method. Program prompts the user to enter which class and method will be used like below. Additionally, it asks for parameters apart from `model_path`:

```
 >>> python evaluate.py -m "../tmp/model/turkish-word2vec-binary.model" -af ../tasks/analogy/test1 -af "../tasks/analogy/test2"

WARNING: No metadata about the model `../tmp/model/turkish-word2vec-binary.model`was found. Please configure how the model is to be loaded below.
 -> Class of the model (see classes in `model.py`): Word2VecWrapper
 -> abstractmethod which loads the model: from_keyed_vectors
Will this method require parameters in addition to `dict_keys(['model_path'])`?
 -> yes/[no]: yes
    -> Parameter name: binary
    -> Parameter type (float/bool/str): bool
       Enter 1 for True, 0 for False
    -> Parameter value: 1
Will this method require parameters in addition to `dict_keys(['model_path', 'binary'])`?
 -> yes/[no]:
Here is the list of parameters you choose:
{'model_path': '../tmp/model/turkish-word2vec-binary.model', 'binary': True}
Are they correct?
 -> yes/[no]: yes
INFO: Saving the new metadata to `../tmp/model/turkish-word2vec-binary.model.nlp_metadata`
```

Once this step is done, program will never ask for the same information when it is run with the same model. Configuration is saved to `<model>.nlp_metadata`.

Model is evaluated with the given tasks and results of each task is saved to the directory where `evaluate.py` is called. This result will be used to generate latex comparison tables.

## Usage of `populate_task_metadata.py`

You can choose a folder with `-f` and create a metadata file for it. This will be used as reference when creating a latex table from the task.

```
 >>> python populate_task_metadata.py -f ../tasks
INFO: No metadata found in `../tasks`. Initialising with empty metadata
INFO: Found files: ['analogy.txt', 'similarity.txt']
 -> Reference score for `analogy.txt`: 12
 -> Reference score for `similarity.txt`: 32
{'analogy.txt': 12.0, 'similarity.txt': 32.0}
INFO: Metadata of `../tasks` is updated
```