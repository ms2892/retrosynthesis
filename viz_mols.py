from GNN_utils import *

from paroutes import PaRoutesInventory, PaRoutesModel, get_target_smiles


model = torch.load('GNN_split_Model.pth')
# model_critic = GATCritic(9,1)
inventory = get_target_smiles(5)
index=2505

edges = model(convert_smiles_to_pyg_graph(inventory[index]))


from split_environment import *

env = SASEnvironment(inventory)

action = np.argmax(edges.detach().numpy())

action

bond = get_BRICS_bonds(Chem.MolFromSmiles(inventory[index]))

bond = [list(i[0]) for i in bond]
bond.sort()
bond = bond[action]

t = env.step([bond],inventory[index])

print(t)

print(reward_func([Chem.MolFromSmiles(inventory[index])],'sas'))
# display(Chem.MolFromSmiles(inventory[index]))

for i,mol in enumerate(t[0]):
    with open(f"mol_{i}_{reward_func([mol],'sas')}.png", "wb") as png:
        png.write(mol)
    