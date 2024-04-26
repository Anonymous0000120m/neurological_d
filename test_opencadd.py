import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors  # 导入Descriptors模块
import matplotlib.pyplot as plt

# 读取SDF文件并将其转换为RDKit的Mol对象列表
sdf_file = 'molecules.sdf'
suppl = Chem.SDMolSupplier(sdf_file)
molecules = [mol for mol in suppl if mol]

# 计算分子描述符
descriptors = []
for idx, mol in enumerate(molecules, 1):
    if mol is not None:
        descriptors.append([
            Descriptors.MolWt(mol),
            Descriptors.NumHDonors(mol),
            Descriptors.NumHAcceptors(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.NumAromaticRings(mol),
            Descriptors.NumAliphaticRings(mol),
            Descriptors.FractionCSP3(mol),
            Descriptors.MaxPartialCharge(mol),
            Descriptors.MinPartialCharge(mol)
        ])

        # 保存分子图片
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(f'molecule_{idx}.png')

        # 绘制分子描述符曲线图
        x_labels = ['MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA', 'NumRotatableBonds', 
                    'NumAromaticRings', 'NumAliphaticRings', 'FractionCSP3', 
                    'MaxPartialCharge', 'MinPartialCharge']
        plt.plot(x_labels, descriptors[-1], marker='o', label=f'Molecule {idx}')
        plt.xlabel('Descriptors')
        plt.ylabel('Values')
        plt.xticks(rotation=45)
        plt.legend()
        plt.savefig(f'molecule_{idx}_descriptors.png', bbox_inches='tight')
        plt.close()

# 将分子描述符保存到CSV文件
csv_file = 'molecule_descriptors.csv'
pd.DataFrame(descriptors, columns=x_labels).to_csv(csv_file, index=False)

print(f'Molecule descriptors saved to {csv_file}')
