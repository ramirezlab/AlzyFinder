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
