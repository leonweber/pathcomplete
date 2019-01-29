from pathme.reactome import rdf_sparql
from pathme import utils as pm_utils
import tqdm
from rdflib.term import URIRef, Literal

GET_URI_ID_TO_UNIPROT = """
SELECT DISTINCT ?protein_reference ?uniprot_id 
WHERE
    {
    ?protein biopax3:entityReference ?protein_reference .
    ?protein_reference biopax3:xref ?xref .
    ?xref biopax3:db ?db .
    ?xref biopax3:id ?uniprot_id .
    }
"""

GET_REACTION_LEFTS = """
SELECT ?reaction_ID ?entity_ID ?ref_id WHERE {
  ?reaction_ID a biopax3:BiochemicalReaction ;
  biopax3:left ?entity_ID .
  OPTIONAL {
  ?entity_ID biopax3:entityReference ?ref_id
  }
}
"""

GET_REACTION_RIGHTS = """
SELECT ?reaction_ID ?entity_ID ?ref_id WHERE {
  ?reaction_ID a biopax3:BiochemicalReaction ;
  biopax3:right ?entity_ID .
  OPTIONAL {
  ?entity_ID biopax3:entityReference ?ref_id
  }
}
"""

GET_COMPLEX_CONSTITUENTS = """
SELECT ?complex_ID ?entity_ID ?ref_id WHERE {
  ?complex_ID a biopax3:Complex ;
  biopax3:component ?entity_ID .
  OPTIONAL {
  ?entity_ID biopax3:entityReference ?ref_id
  }
}
"""

GET_CONTROLS = """
SELECT ?reaction_ID ?controller_id ?ref_id ?control_type WHERE {
  ?control_ID a biopax3:Control ;
  biopax3:controller ?controller_id .
  ?control_ID  biopax3:controlled ?reaction_ID .
  OPTIONAL {
  ?controller_id biopax3:entityReference ?ref_id .
  ?control_ID biopax3:controlType ?control_type .
  }
}
"""


def main():
    graph = pm_utils.parse_rdf('data/PathwayCommons10.pid.BIOPAX.owl', format='xml')
    tuples = []
    lefts = graph.query(GET_REACTION_LEFTS, initNs=rdf_sparql.PREFIXES)
    rights = graph.query(GET_REACTION_RIGHTS, initNs=rdf_sparql.PREFIXES)
    complexes = graph.query(GET_COMPLEX_CONSTITUENTS, initNs=rdf_sparql.PREFIXES)
    controls = graph.query(GET_CONTROLS, initNs=rdf_sparql.PREFIXES)



    for reaction_id, entity_id, ref_id in lefts:
        if 'SmallMolecule' in entity_id or 'Rna' in entity_id:
            continue
        if ref_id and 'uniprot' in ref_id:
            entity_id = ref_id.split('/')[-1]
        tuples.append((str(reaction_id), str(entity_id), "has_left"))

    for reaction_id, entity_id, ref_id in rights:
        if 'SmallMolecule' in entity_id or 'Rna' in entity_id:
            continue
        if ref_id and 'uniprot' in ref_id:
            entity_id = ref_id.split('/')[-1]
        tuples.append((str(reaction_id), str(entity_id), "has_right"))

    for complex_id, entity_id, ref_id in complexes:
        if 'SmallMolecule' in entity_id or 'Rna' in entity_id:
            continue
        if ref_id and 'uniprot' in ref_id:
            entity_id = ref_id.split('/')[-1]
        tuples.append((str(reaction_id), str(entity_id), "has_component"))

    for reaction_id, entity_id, ref_id, control_type in controls:
        if "BiochemicalReaction" not in reaction_id:
            continue

        if 'SmallMolecule' in entity_id or 'Rna' in entity_id or 'Pathway' in entity_id:
            continue

        if ref_id and 'uniprot' in ref_id:
            entity_id = ref_id.split('/')[-1]

        if not control_type:
            control_type = "regulation"
        else:
            control_type = control_type.lower()
        tuples.append((str(reaction_id), str(entity_id), control_type))

    
    with open('ncbi.cif', 'w') as f:
        for t in tuples:
            f.write("\t".join(t) + '\n')


if __name__ == '__main__':
    main()


