Reading/Writing
===============

All epistasis models/simulators store data in Pandas_ Series/DataFrames, so the data
can be easily written to various file formats. This page lists a few.

.. _Pandas: http://pandas.pydata.org/

Saving an epistasis model
-------------------------

The simplest way to save an epistasis model is to use Python's ``pickle`` library.

.. code-block:: python

  # Import pickle module
  import pickle
  from epistasis.models import EpistasisLinearRegression

  # Simple model object
  model = EpistasisLinearRegression(model=1)

  # Save to disk to open later.
  with open('model.pickle', 'wb') as f:
      pickle.dump(f, model)

To load the saved model,

.. code-block:: python

  # Import pickle module
  import pickle

  # Read from file.
  with open('model.pickle', 'rb') as f:
      model = pickle.load(f)


Excel Spreadsheet
-----------------

Epistatic coefficients can be written to an excel file using the ``to_excel`` method

.. code-block:: python

  model.to_excel('data.xlsx')

.. raw:: html

  <table border="1" class="dataframe">
    <thead>
      <tr style="text-align: center;">
        <th></th>
        <th>genotypes</th>
        <th>phenotypes</th>
        <th>stdeviations</th>
        <th>n_replicates</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <th>0</th>
        <td>PTEE</td>
        <td>0.243937</td>
        <td>0.013269</td>
        <td>1</td>
      </tr>
      <tr>
        <th>1</th>
        <td>PTEY</td>
        <td>0.657831</td>
        <td>0.055803</td>
        <td>1</td>
      </tr>
      <tr>
        <th>2</th>
        <td>PTFE</td>
        <td>0.104741</td>
        <td>0.013471</td>
        <td>1</td>
      </tr>
      <tr>
        <th>3</th>
        <td>PTFY</td>
        <td>0.683304</td>
        <td>0.081887</td>
        <td>1</td>
      </tr>
      <tr>
        <th>4</th>
        <td>PIEE</td>
        <td>0.774680</td>
        <td>0.069631</td>
        <td>1</td>
      </tr>
      <tr>
        <th>5</th>
        <td>PIEY</td>
        <td>0.975995</td>
        <td>0.059985</td>
        <td>1</td>
      </tr>
      <tr>
        <th>6</th>
        <td>PIFE</td>
        <td>0.500215</td>
        <td>0.098893</td>
        <td>1</td>
      </tr>
      <tr>
        <th>7</th>
        <td>PIFY</td>
        <td>0.501697</td>
        <td>0.025082</td>
        <td>1</td>
      </tr>
      <tr>
        <th>8</th>
        <td>RTEE</td>
        <td>0.233230</td>
        <td>0.052265</td>
        <td>1</td>
      </tr>
      <tr>
        <th>9</th>
        <td>RTEY</td>
        <td>0.057961</td>
        <td>0.036845</td>
        <td>1</td>
      </tr>
      <tr>
        <th>10</th>
        <td>RTFE</td>
        <td>0.365238</td>
        <td>0.050948</td>
        <td>1</td>
      </tr>
      <tr>
        <th>11</th>
        <td>RTFY</td>
        <td>0.891505</td>
        <td>0.033239</td>
        <td>1</td>
      </tr>
      <tr>
        <th>12</th>
        <td>RIEE</td>
        <td>0.156193</td>
        <td>0.085638</td>
        <td>1</td>
      </tr>
      <tr>
        <th>13</th>
        <td>RIEY</td>
        <td>0.837269</td>
        <td>0.070373</td>
        <td>1</td>
      </tr>
      <tr>
        <th>14</th>
        <td>RIFE</td>
        <td>0.599639</td>
        <td>0.050125</td>
        <td>1</td>
      </tr>
      <tr>
        <th>15</th>
        <td>RIFY</td>
        <td>0.277137</td>
        <td>0.072571</td>
        <td>1</td>
      </tr>
    </tbody>
  </table>
  <br>


CSV File
--------

Epistatic coefficients can be written to a csv file using the ``to_csv`` method

.. code-block:: python

  model.to_csv('data.csv')

JSON Format
-----------

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
