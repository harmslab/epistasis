Accepted data format
====================

Currently, the ``epistasis`` package only takes JSON formated data as input to
the epistasis model. We expect to get include more formats (i.e. csv/excel) moving
forward, just haven't gotten around to it.

The only keys recognized by the json reader are:

    1. `genotypes`
    2. `phenotypes`
    3. `stdeviations`
    4. `mutations`
    5. `n_replicates`
    6. `log_transform`

All other keys are ignored in the epistasis models. You can keep other metadata
stored in the JSON, but it won't be appended to the epistasis model object.

.. code-block:: javascript

    {
        "genotypes" : [
            '000',
            '001',
            '010',
            '011',
            '100',
            '101',
            '110',
            '111'
        ],
        "phenotypes" : [
            0.62344582,
            0.87943151,
            -0.11075798,
            -0.59754471,
            1.4314798,
            1.12551439,
            1.04859722,
            -0.27145593
        ],
        "stdeviations" : [
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
            0.01,
        ],
        "mutations" : {
            0 : ["0", "1"],
            1 : ["0", "1"],
            2 : ["0", "1"],
        }
        "n_replicates" : 12,
        "log_transform" : false,
        "title" : "my data",
        "description" : "a really hard experiment"
    }
