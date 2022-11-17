# https://molecularai.github.io/aizynthfinder/python_interface.html
import json

from aizynthfinder.aizynthfinder import AiZynthFinder



def predict(finder, idx, smiles):
    print(idx, smiles)
    finder.target_smiles = smiles
    finder.tree_search()

    finder.build_routes()
    stats = finder.extract_statistics()

    print(stats)
    routes = finder.routes

    with open(f'{idx}_routes.json', 'w') as outfile:
        json.dump(routes.dicts, outfile, indent=4)

    with open(f'{idx}_stats.json', 'w') as outfile:
        json.dump(stats, outfile, indent=4)


def main():
    filename = "config.yml"
    finder = AiZynthFinder(configfile=filename)
    print("doing")

    finder.stock.select("zinc")
    finder.expansion_policy.select("uspto")
    finder.filter_policy.select("uspto")

    mols = {
        'MOL001': 'CC(=O)NCC1CN(c2ccc(N3CCOCC3)cc2)C(=O)O1',
        'MOL002': 'Cc1[nH]c2ccccc2c1CCNCc1ccc(/C=C/C(=O)NO)cc1',
        'MOL003': 'N[C@@H](CC(=O)N1CCn2c(nnc2C(F)(F)F)C1)Cc1cc(F)c(F)cc1F',
        'MOL004': 'Cc1cccc(C)c1N(CC(=O)Nc1ccc(-c2ncon2)cc1)C(=O)C1CCS(=O)(=O)CC1',
        'MOL005': 'CN1CCC(c2c(O)cc(O)c3c(=O)cc(-c4ccccc4Cl)oc23)C(O)C1',
        'MOL006': 'COc1cc(F)ccc1-c1ncnc(Nc2cccc(CS(C)(=N)=O)c2)n1',
        'MOL007': 'CCc1cccc(CC)c1NC(=O)CCl',
    }

    for idx, smiles in mols.items():
        predict(finder, idx, smiles)


if __name__ == '__main__':
    main()
