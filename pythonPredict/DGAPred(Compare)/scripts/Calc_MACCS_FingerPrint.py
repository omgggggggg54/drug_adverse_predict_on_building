import os
import numpy as np
import pandas as pd
import os
import rdkit
from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem import Draw
from rdkit.Chem import MACCSkeys
from alive_progress import alive_bar

def getMaccsFingerPrint(smiles):
    if smiles=="" or pd.isna(smiles):
        return ""
    mol = Chem.MolFromSmiles(smiles)
    fp = MACCSkeys.GenMACCSKeys(mol)
    return "".join(map(str,fp))

if __name__ == "__main__":

    df_smilesList = pd.read_csv('D:/web/mysql-8.0.28-winx64/data/exp_data/MeshID_SMILES6980.csv',header=None,delimiter='\t')
    df_smilesList.columns=["MeshID","SMILES"]
    print(df_smilesList.columns)
    dict_maccs=dict()
    with alive_bar(df_smilesList.shape[0]) as bar:
        for drugid, simle in zip(df_smilesList["MeshID"],df_smilesList["SMILES"]):
            bar()
            if simle=='-666' or simle=='restricted' or simle=='' or pd.isna(simle):
                continue
            if drugid=="MESH:C002385" or drugid=="MESH:C029217" or drugid=="MESH:C043438" or drugid=="MESH:C082874" or drugid=="MESH:C114535" or drugid=="MESH:C118603" or drugid=="MESH:C120851"\
                or drugid=="MESH:C453079":  #BRD-U57440914
                print("skip:",drugid, Chem.MolFromSmiles(simle))
                continue
            dict_maccs[drugid]=getMaccsFingerPrint(simle)
    df_maccs=pd.DataFrame(list(dict_maccs.items()))
    df_maccs.columns=[df_smilesList.columns[0],"MACCS"]
    df_maccs.to_csv('meta_SIMLES_Maccs_MESH6980.csv',header=True,index=False)