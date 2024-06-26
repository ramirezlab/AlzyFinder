# AlzyFinder
#### Alzyfinder Platform: A web-based tool for ligand-based virtual screening and network pharmacology

[Ramirez Lab](https://ramirezlab.github.io/index)

Pharmacoinformatics and Systems Pharmacology <br>
[Facultad de Ciencias Biológicas](https://cienciasbiologicasudec.cl/) - [Universidad de Concepción](https://www.udec.cl/pexterno/) <br>

## Table of contents  

* [Objective](#objective)
* [Usage](#usage)
* [Contact](#contact)
* [License](#license)
* [Citation](#citation)

## Objective

(Back to [Table of contents](#table-of-contents).)
Currently, there are few therapeutic alternatives for Alzhiemer's Disease (AD) (lecanemab, aducanumab, donepezil, rivastigmine, galantamine, and memantine)[1]. However, multiple clinical trials (Phase I – IV) of different drugs and bioactive compounds are being carried out [2]. Polypharamacological profiles of drug/targets could be used for drug repurposing, identifying, and validating new targets, and finding new bioactive ligands, among other applications.
Here we introduce the [Alzyfinder Platform](https://www.alzyfinder-platform.udec.cl), a web-based tool designed for virtual screening that uses an array of machine learning models built for over 80 key targets associated with Alzheimer’s disease. The platform’s user-friendly interface facilitates the execution of multiple virtual screening tasks (up to 100 molecules screened at the same time agaist 85 AD targets), utilizing ligand-based drug design approaches.
If users want to screen more than 100 molecules, in this AlzyFinder repository they will find all the ML models, their validation as well as a script to perform the screening locally. For more details on how the machine learning models were built and validated for each AD target, see the article.

1. [DOI: 10.1016/j.cellsig.2022.110539](https://www.sciencedirect.com/science/article/pii/S0898656822003011)
2. [DOI: 10.3233/JAD-190507](https://content.iospress.com/articles/journal-of-alzheimers-disease/jad190507) 

## Usage

(Back to [Table of contents](#table-of-contents).)

You can use AlzyFinder locally (download repository and install dependencies).

#### Linux

1.  Get your local copy of the AlzyFinder repository by:

    - Downloading it as zip archive and unzipping it.
    - Cloning it to your computer using the package `git`:

        ```bash
        git clone https://github.com/ramirezlab/AlzyFinder.git
        ```
        
2.  Use the [Anaconda](https://docs.anaconda.com/anaconda/install/) for a clean package version management. 
   
3.  Use the package management system conda to create an environment (called `AlzyFinder`) to perfom locally the ligand-based virtual screening (LBVS) using up to 255 machine learning (ML) models (85 AD targets, 3 ML models per target)
   
    We provide an environment file (yml file) containing all required packages.

    ```bash
    conda env create -f environment.yml
    ```

    Note: You can also create this environment manually. 
    Check ["Alternatively create conda environment manually"](#Alternatively-create-conda-environment-manually) for this.

4.  Activate the conda environment.
    
    ```bash
    conda activate AlzyFinder
    ```
    
    Now you can work within the conda environment and get started with your LBVS campaigns. Have fun!!!
 

