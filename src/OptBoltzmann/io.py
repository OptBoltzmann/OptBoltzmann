import cobra
import simplesbml
from cobra.io.sbml import F_REPLACE as f_replace


def parse_stoichiometry(side):
    if len(specie.split()) > 1:
        coefficient, metab = specie.split()
        return f_replace["F_SPECIE_REV"](metab), int(coefficient)
    return f_replace["F_SPECIE_REV"](metab), 1


def dataframes2simplesbml(
    Stoich_matrix: pd.DataFrame,
    reactions: pd.DataFrame,
    metabolites: pd.DataFrame,
    Keq: pd.DataFrame,
    compartment_volume,
) -> simplesbml.SbmlModel:
    model = simplesbml.SbmlModel()
    compartments = set(met.split(":")[1] for met in metabolites.index)
    for compartment in compartments:
        model.addCompartment(compartment_volume[compartment], comp_id=compartment)
    for m in metabolites.index:
        if metabolites.loc[m, "Variable"]:
            prefix = ""
        else:
            prefix = "$"
        model.addSpecies(
            prefix + f_replace["F_SPECIE_REV"](m), metabolites.loc[m, "Conc"], comp=m.split(":")[1]
        )  # needs to be in moles
    for rxn in reactions.index:
        left = [parse_stoichiometry(specie) for specie in reactions.loc[rxn, "LEFT"].split(" + ")]
        right = [parse_stoichiometry(specie) for specie in reactions.loc[rxn, "RIGHT"].split(" + ")]
        f = f"Keq*{'*'.join(left)}/({'*'.join(right)})"  # does not include nonunit stoichiometry
        model.addReaction(
            reactants=left,
            products=right,
            expression=f"{f} - 1/{f}",
            local_params={"Keq": Keq.loc[rxn, "Keq"]},
            rxn_id=f_replace["F_REACTION_REV"](rxn),
        )
    return model
