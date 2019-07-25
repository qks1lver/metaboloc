# TonKnows
#### A simple network miner
Very much in ALPHA right now

Python 3

### What is this for?
I have a network with **nodes** connected through **links**. Some of the nodes I know have property **X** and some I
know have property **Y**. But I don't know them all, so can I predict the properties (X, Y, or both) of the unknown nodes?

**TonKnows** is a multilabel multiclass classifier designed to shed some light on the characteristics of your network
using very simple base classifiers to ask understandable questions.
1. Do I have enough data?
2. Do neighboring nodes determine the property of a target node?
3. How do the links contribute to the properties of the nodes?
4. If I'm friend with Merry and also friend with Jason, are Merry and Jason friends to each other?

### Install
* Need a few things:
    1. Python 3.5+
    2. Scikit Learn - which should install numpy, scipy, etc...
    3. Pandas
    4. (for plotting) Seaborn

* No wheels yet, so just download or clone this repository, and we'll go from there

### Usage
##### TL;DR
* To **train** and save model

    ```./know.py -v --td where/you/have/network.tsv --save```

* To **evaluate** the trained model with another network

    ```./know.py -v --mod models/model.pkl --ed where/you/have/network2.tsv```

* To use trained model to **predict** nodes of another network with unknown labels

    ```./know.py -v --mod models/model.pkl --pd where/you/have/network3.tsv```

##### Let's first familiarize ourselves with where things are.
* Open a **terminal** and navigate to the ``cd where/you/put/tonknows/`` folder. You should see a ``know.py`` file. That file runs everything.
Please keep it there. Now try this:

    ```python3 know.py --help```

    Now you should see an awful list of possible **options** to use. At this point, you probably also noticed a bunch
    of folders suddenly got created under ``tonknows/``: ``model/``, ``test/``, ``tmp/``.
    They will be used to store many things automatically generated by **TonKnows**.

##### Nows let's create, train, and evaluate some test networks.
* Enter this into the terminal while you are still in the ``tonknows/`` folder:

    ```./know.py --test```
    
    Doing this creates 3 networks:
    1. Pure network ``test/pure_network.tsv``
    2. Mixed network ``test/mix_network.tsv``
    3. Unknown network ``test/unknown_network.tsv``
    
* Lets try training on these networks. First, the pure network. Enter:
    
    ```./know.py -v --td test/pure_network.tsv```
    
    By default, TonKnows will iterates through the entire dataset 3 times with a 10-fold CV each time. This not only
    helps evaluate the base classifier performances but also is necessary to train the final ensemble classifier and
    a few other cutoff values. Basically use ``--td`` (**t**rain **d**ata) to train a network and, also, a bunch of 
    text will scroll through since the ``-v`` verbose flag is used.
    
    You should see from each othe the **performance blocks** that AUC-ROC, F1, precision, recall are pretty much
    perfect. You will never get this unless there are **leaks** in the data or if it is a **pure network** such as this
    one, where each label is only associated to a subnetwork that is not mixed with another label's network.
    
* Let's try something more interesting, and let's use the mixed network:
    
    ```./know.py -v --td test/mix_network.tsv --save --setcurrent```
    
    This network would not result in as high of a performance as the pure network, because about 40% of the labels are
    associated to random nodes. Thus it becomes very challenging to determine any real relationships.
    
    This time the model is saved with the ``--save`` flag. And this new model is set to the **current** one with the
    ``--setcurrent`` flag.
    
* Let's evaluate the trained model using the pure network with ``--ed`` (**e**valuate **d**ata):
    
    ```./know.py -v --ed test/pure_network.tsv --shell```
    
    Since there is a **-current** model, TonKnows just finds that and use it by default.
    
    The performance is should be pretty good still. The ``--shell`` option
    opens a Python interactive session. This can let you take a closer look at the evaluation results however you want.

* Let's see what we can do in the interactive session. Interactive sessions opened after different tasks can have
different items to play with. After running ``--ed``, you would have ``m`` (a Model object) and ``res`` (a Python
dictionary of results). Try enter:
    
    ```res.keys()``` To see all the **keys** available for this dictionary
    
    ```res['ytruth']``` To see the one-hot encoded labels of the **true** data
    
    ```res['yopt']``` To see the predicted probabilities of labels for each sample
    
    ```m.datas[m.train_idx].labels``` To see the labels corresponding to each columns of **y** (i.e. ytruth, yopt,
    yinf, ymatch, ynet)
    
    The performance displayed shows the weighted average scores, so what if we want to see how the model performs for
    each label. Enter:
    
    ```
    r = m.scores(res['ytruth'], res['yopt'])
    r['aucroc_labs']
    ```
    
    This will recompute the scores for ``res['yopt']`` against ``res['ytruth']``. The AUC-ROC for each label is an
    array under ``r['aucroc_labs']``
    
    Of course, you can also do the same for f1, precision, and recall by using ``r['f1_labs']``,
    ``r['precision_labs']``, ``r['recall_labs']``
    
    Finally there's ``r['lab_ratios']``, which shows the distribution of each label in the data.

* At last, let's try to predict the properties of nodes that we don't have labels for with ``--pd`` (**p**redict
**d**ata). Let's do:
    
    ```./know.py -v --pd test/unknown_network.tsv```
    
    The **unknown_network.tsv** dataset contains labeled and unlabeled data. The previous **mix_data** is mixed again,
    then written to the ``unknown_network`` file twice: once with labels and once without labels.
    
    In ``--pd`` mode, samples with labels will be used the estimate the prediction performance by evaluating model
    performance the same way as ``--ed``. Then predict labels for data without labels. Predictions are written to the
    file in the same folder as the input network file. The samples without labels in the original data will be written
    to file first, followed by the samples with labels. The known labels are in the **known_labels** column, final
    predictions from the optimized ensemble classifier is in the **predictions** column. The **merged** column contains
    predictions for samples with no true labels and original true labels for samples with them.
    