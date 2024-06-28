# AlzyFinder
#### AlzyFinder Platform: A web-based tool for ligand-based virtual screening and network pharmacology

[Pharmacoinformatics and Systems Pharmacology lab](https://ramirezlab.github.io/index) at [Facultad de Ciencias Biológicas](https://cienciasbiologicasudec.cl/) - [Universidad de Concepción](https://www.udec.cl/pexterno/) <br>

## Table of contents  

* [Objective](#objective)
* [Usage](#usage)
* [Contact](#contact)
* [License](#license)
* [Citation](#citation)

## Objective

(Back to [Table of contents](#table-of-contents))

Currently, there are few therapeutic alternatives for Alzhiemer's Disease (AD) (lecanemab, aducanumab, donepezil, rivastigmine, galantamine, and memantine)[1]. However, multiple clinical trials (Phase I – IV) of different drugs and bioactive compounds are being carried out [2]. Polypharamacological profiles of drug/targets could be used for drug repurposing, identifying, and validating new targets, and finding new bioactive ligands, among other applications.

Here we introduce the [AlzyFinder Platform](https://www.alzyfinder-platform.udec.cl), a web-based tool designed for virtual screening that uses an array of machine learning models built for over 80 key targets associated with Alzheimer’s disease. The platform’s user-friendly interface facilitates the execution of multiple virtual screening tasks (up to 100 molecules screened at the same time agaist 85 AD targets), utilizing ligand-based drug design approaches.

If users want to screen more than 100 molecules, in this AlzyFinder repository they will find all the ML models, their validation as well as a script to perform the screening locally. For more details on how the machine learning models were built and validated for each AD target, ***see the article***.
Additionally, since each protein was modeled using three different optimized classifications (one per selected metric balanced accuracy, precision and F1), a fourth integrative model was developed by using a soft-voting method [3] implemented by calculating the average of the classification probabilities provided by each of the three independent models' results. In this ensemble result, each of the three models contributes a vote weighted by its confidence in the classification, calculated from the probability of belonging to the assigned class.



1. [DOI: 10.1016/j.cellsig.2022.110539](https://www.sciencedirect.com/science/article/pii/S0898656822003011).
2. [DOI: 10.3233/JAD-190507](https://content.iospress.com/articles/journal-of-alzheimers-disease/jad190507).
3. [DOI: 10.1007/978-3-642-38067-9_27](https://link.springer.com/chapter/10.1007/978-3-642-38067-9_27).

## Usage

(Back to [Table of contents](#table-of-contents))

You can use AlzyFinder locally (download repository and install dependencies).

#### Linux - Install locally

-  Get your local copy of the AlzyFinder repository by:

    - Downloading it as zip archive and unzipping it.
    - Cloning it to your computer using the package `git`:

        ```bash
        git clone https://github.com/ramirezlab/AlzyFinder.git
        ```
        
-  Use the [Anaconda](https://docs.anaconda.com/anaconda/install/) for a clean package version management. 
   
-  Use the package management system conda to create an environment (called `AlzyFinder`) to perfom locally the ligand-based virtual screening (LBVS) using up to 255 machine learning (ML) models (85 AD targets, 3 ML models per target).
   
    We provide an environment file (yml file) containing all required packages.

    ```bash
    conda env create -f environment.yml
    ```

    **Note**: You can also create this environment manually. 
    Check ["Alternatively create conda environment manually"](#Alternatively-create-conda-environment-manually) for this.

-  Activate the conda environment.
    
    ```bash
    conda activate AlzyFinder
    ```
    
    Now you can work within the conda environment and get started with your LBVS campaigns. Have fun!!!
    
    
#### Ligand-Based Virtual Screening

- In the `input` folder you will find all ML models for the 85 selected AD targets (3 model per target). In the same `input` folder the file with the molecules to screen should be stored as a .CSV file. As an example, a file called `molecules-to-screen.csv` is given in this reposotory. Change the file with the molecules of interest.


- Execute the following python script to screen the selected molecues (in the `molecules-to-screen.csv` file) against all AD tagets. 
The probability that a molecule is active against each target is shown as a value from 0 to 1. The higher this value, the higher the probability that the molecule shows activity against a target as a result of a virtual screening with three different ML models. 

At the end of the <a href="https://github.com/ramirezlab/AlzyFinder/blob/main/AlzyFinder_ML-LBVS.py">`AlzyFinder_ML-LBVS.py`</a> python scrip you can include a `probability threshold` so that the results are filtered and only molecules with probabilities greater than the selected threshold are presented in a drug-protein interaction network (DPIn). In this case the probabilty is set to *0.7*. 

**Note**: Here we use the Morgan fingerprint using the RDKit. If you want to used another fingerprint, or for further information about available fingerprints check the [rdkit.Chem.rdMolDescriptors module](https://www.rdkit.org/docs/source/rdkit.Chem.rdMolDescriptors.html) for this.


```python 
import numpy as np
import pandas as pd
import xgboost as xgb
import os
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit import DataStructs
from rdkit.Chem import rdMolDescriptors


def load_models(models_folder):
    # Load XGBoost models from JSON files in the specified folder.
    models = {}
    for file in os.listdir(models_folder):
        if file.endswith(".json"):
            model_path = os.path.join(models_folder, file)
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            model_name = file.replace("_best_model.json", "")  # Get the model name
            models[model_name] = model
    return models


def read_data(csv_path):
    # Read a CSV file containing molecular data.
    df = pd.read_csv(csv_path, sep=" ", names=["smile", "name"], engine='python')
    # Assign default names to SMILES without a name
    counter = 1
    for i, row in df.iterrows():
        if pd.isnull(row['name']):
            df.at[i, 'name'] = f'Ligand {counter}'
            counter += 1
    return df


def calculate_fp(molecule, method='maccs'):
    # Calculate the fingerprint of a molecule.
    return rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, 2)


def to_fingerprint(df, fp_name, verbose=False):
    """
    Calculate the specified fingerprint for each molecule in the DataFrame.
    Args:
        df: Pandas DataFrame containing 'smiles' column.
        fp_name: Name of the fingerprint method to use.
        verbose: If True, print status messages during execution.
    Returns:
        Modified DataFrame with the calculated fingerprint column.
    """
    df_fp = df.copy()
    if verbose:
        print('> Constructing a molecule from SMILES and Creating fingerprints')
    df_fp['molecule'] = df_fp.smile.map(lambda smile: Chem.MolFromSmiles(smile))
    df_fp[fp_name] = df_fp.molecule.apply(calculate_fp, args=[fp_name])
    df_fp = df_fp.drop(['smile'], axis=1)
    return df_fp


def decode_fingerprints(df):
    # Decode fingerprints stored in DataFrame into a numerical format.
    decoded_fingerprints = []
    for fp in df['morgan2_c']:
        arr = np.zeros((0,))
        DataStructs.ConvertToNumpyArray(fp, arr)
        decoded_fingerprints.append(arr)
    return np.array(decoded_fingerprints)


def predictor(models_path, data_path):
    models = load_models(models_path)
    data_df = read_data(data_path)
    data_df.dropna(subset=['smile'], inplace=True)
    # Calculate fingerprints
    df_fingerprints = to_fingerprint(data_df, 'morgan2_c')
    df_fingerprints = decode_fingerprints(df_fingerprints)
    # Prepare DataFrame for results
    model_names = list(models.keys())
    result_columns = ['Ligand'] + [f'{model_name}' for model_name in model_names]
    df_results = pd.DataFrame(columns=result_columns)
    df_results['Ligand'] = data_df['name']
    # Evaluate with each model
    for model_name, model in models.items():
        pred_prob = model.predict_proba(df_fingerprints)[:, 1]
        df_results[f'{model_name}'] = pred_prob
    targets = set(col.split('_')[0] for col in df_results.columns if not col.startswith("Ligand"))
    targets = sorted(targets)
    # Save raw results
    df_results.to_csv('output/raw_results.csv', index=False)
    for target in targets:
        # Select columns corresponding to this target
        cols_target = [col for col in df_results.columns if target in col]
        # Calculate the average and add it as a new column
        df_results[f'{target}'] = df_results[cols_target].mean(axis=1)
        df_results.drop(columns=cols_target, inplace=True)
    # Save results
    df_results.to_csv('output/results.csv', index=False)
    return df_results


def to_graphos(df_results, threshold=0.5):
    # Create a DataFrame for ligands
    df_ligands = pd.DataFrame({'node': df_results['Ligand'], 'type': 'Ligand'})
    # Create a DataFrame for targets
    df_targets = pd.DataFrame({'node': df_results.columns[1:], 'type': 'Target'})
    # Combine both DataFrames
    df_nodes = pd.concat([df_ligands, df_targets])
    # Save to a CSV file
    df_nodes.to_csv('output/nodes.csv', index=False)
    # 'Melt' the DataFrame to obtain the desired structure
    df_melted = df_results.melt(id_vars='Ligand', var_name='target', value_name='probability')
    # Save to a CSV file
    df_melted.to_csv('output/ligand_target_probability.csv', index=False)
    # Filtering
    df_melted_filtered = df_melted[df_melted['probability'] >= threshold]
    # Save to a CSV file
    df_melted_filtered.to_csv('output/ligand_target_probability_filtered.csv', index=False)


if __name__ == "__main__":
    models_path = 'input/models'
    data_path = 'input/molecules-to-screen.csv'  # File name with the molecules to screen
    df_results = predictor(models_path, data_path)
    to_graphos(df_results, 0.7) # probablily threshold

```

- After performing the LBVS, five files will be created and stored in the `output` folder.

    - `raw_results.csv`: Complete screening results for each molecule against each ML model (3 models per protein).
    - `results.csv`: Complete screening results as a matrix, including all molecules against all targets (ensemble result). 
    - `ligand_target_probability.csv`: Complete screening (ensemble) results as a list, including all molecules against all targets. This file could be used to create a graph in a software to visualizing complex networks such as [Cytoscape](https://cytoscape.org/)
    - `ligand_target_probability_filtered.csv`: Screening (ensemble) results as a list, filtered using the probablily threshold, including all molecules against all targets. This file could be used to create a graph in a software to visualizing complex networks such as [Cytoscape](https://cytoscape.org/).
    - `nodes.csv`: Classification of the DPIn nodes. Nodes can be *ligands* or *targets*.


    
## Contact
(Back to [Table of contents](#table-of-contents))

Please contact us if you have questions or suggestions!

* If you have questions regarding our the ML models deposited here, please open an issue on our GitHub repository: https://github.com/ramirezlab/AlzyFinder/issues
* If you have questions regarding our [AlzyFinder Platform](https://www.alzyfinder-platform.udec.cl) please contact us via [this form](https://ramirezlab.github.io/7_contact)
* If you have ideas for new AD targets, or how to imporve our [AlzyFinder Platform](https://www.alzyfinder-platform.udec.cl), please send your ideas via [this form](https://ramirezlab.github.io/7_contact)
* For all other requests, please send us an email: dramirezs@udec.cl

We are looking forward to hearing from you!

## License
(Back to [Table of contents](#table-of-contents))

This work is licensed under the MIT license.
To view a copy of this license, visit https://opensource.org/license/mit.

## Citation
(Back to [Table of contents](#table-of-contents))

The authors of the [AlzyFinder Platform](https://www.alzyfinder-platform.udec.cl) received public funding from the following funders:
* Agencia Nacional de Investigación y Desarrollo (ANID) -  Fondo Nacional de Desarrollo Científico y Tecnológico (FONDECYT) – Chile (Grant No. 1220656) 

If you make use of the [AlzyFinder Platform](https://www.alzyfinder-platform.udec.cl) material in scientific publications, please cite our respective articles:
* Pending....

It will help measure the impact of the [AlzyFinder Platform](https://www.alzyfinder-platform.udec.cl) and future funding!!!

This repository was inspired and adapted from [TeachOpenCADD](https://github.com/dominiquesydow/TeachOpenCADD).
 

