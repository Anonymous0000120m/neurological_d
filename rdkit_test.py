from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import SDWriter

# 创建一个包含示例分子的RDKit Mol对象
mol1 = Chem.MolFromSmiles('CCO')
mol2 = Chem.MolFromSmiles('CCN')

# 添加分子的3D坐标
AllChem.EmbedMolecule(mol1)
AllChem.EmbedMolecule(mol2)

# 创建一个SDWriter对象，用于写入SDF文件
sdf_writer = SDWriter('molecules.sdf')

# 将分子写入SDF文件
sdf_writer.write(mol1)
sdf_writer.write(mol2)

# 关闭SDF文件写入器
sdf_writer.close()

print('SDF file generated successfully.')
