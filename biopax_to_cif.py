from pathme.reactome import rdf_sparql
from pathme import utils as pm_utils
import tqdm
import itertools
from rdflib.term import URIRef, Literal


GET_REACTION_LEFTS = """
SELECT ?reaction_id ?entity_id ?ref_id ?location_name ?mod_type WHERE {
  ?reaction_id a biopax3:BiochemicalReaction ;
  biopax3:left ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?entity_id biopax3:cellularLocation ?location_id .
  ?location_id biopax3:term ?location_name .
  }
  OPTIONAL {
  ?entity_id biopax3:feature ?mod_id .
  ?mod_id biopax3:modificationType ?vocab_id .
  ?vocab_id biopax3:term ?mod_type .
  } 

  
}
"""

GET_REACTION_RIGHTS = """
SELECT ?reaction_id ?entity_id ?ref_id ?location_name ?mod_type WHERE {
  ?reaction_id a biopax3:BiochemicalReaction ;
  biopax3:right ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?entity_id biopax3:cellularLocation ?location_id .
  ?location_id biopax3:term ?location_name .
  }
  OPTIONAL {
  ?entity_id biopax3:feature ?mod_id .
  ?mod_id biopax3:modificationType ?vocab_id .
  ?vocab_id biopax3:term ?mod_type .
  } 
  
}
"""

GET_COMPLEX_CONSTITUENTS = """
SELECT ?complex_id ?entity_id ?ref_id WHERE {
  ?complex_id a biopax3:Complex ;
  biopax3:component ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id
  }
}
"""

GET_CONTROLS = """
SELECT ?reaction_id ?controller_id ?ref_id ?control_type WHERE {
  ?control_id a biopax3:Control ;
  biopax3:controller ?controller_id .
  ?control_id  biopax3:controlled ?reaction_id .
  OPTIONAL {
  ?controller_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?control_id biopax3:controlType ?control_type .
  }
}
"""
GET_CATALYSIS = """
SELECT ?reaction_id ?controller_id ?ref_id ?control_type WHERE {
  ?control_id a biopax3:Catalysis ;
  biopax3:controller ?controller_id .
  ?control_id  biopax3:controlled ?reaction_id .
  OPTIONAL {
  ?controller_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?control_id biopax3:controlType ?control_type .
  }
}
"""


def main():
    graph = pm_utils.parse_rdf('data/PathwayCommons10.pid.BIOPAX.owl', format='xml')
    tuples = set()
    lefts = graph.query(GET_REACTION_LEFTS, initNs=rdf_sparql.PREFIXES)
    rights = graph.query(GET_REACTION_RIGHTS, initNs=rdf_sparql.PREFIXES)
    complexes = graph.query(GET_COMPLEX_CONSTITUENTS, initNs=rdf_sparql.PREFIXES)
    controls = graph.query(GET_CONTROLS, initNs=rdf_sparql.PREFIXES)  
    catalyses = graph.query(GET_CATALYSIS, initNs=rdf_sparql.PREFIXES)



    for reaction_id, entity_id, ref_id, location_name, modification in lefts:
        if 'SmallMolecule' in entity_id or 'Rna' in entity_id:
            continue

        if ref_id and 'uniprot' in ref_id:
            tuples.add((str(entity_id), str(ref_id), "has_id"))

        if location_name:
            tuples.add((str(entity_id), str(location_name), "has_location"))

        if modification:
            tuples.add((str(entity_id), str(modification), "has_modification"))
            
        tuples.add((str(reaction_id), str(entity_id), "has_left"))

    for reaction_id, entity_id, ref_id, location_name, modification in rights:
        if 'SmallMolecule' in entity_id or 'Rna' in entity_id:
            continue
        if ref_id and 'uniprot' in ref_id:
            tuples.add((str(entity_id), str(ref_id), "has_id"))

        if location_name:
            tuples.add((str(entity_id), str(location_name), "has_location"))

        if modification:
            tuples.add((str(entity_id), str(modification), "has_modification"))

        tuples.add((str(reaction_id), str(entity_id), "has_right"))

    for complex_id, entity_id, ref_id in complexes:
        if 'SmallMolecule' in entity_id or 'Rna' in entity_id:
            continue
        if ref_id and 'uniprot' in ref_id:
            tuples.add((str(entity_id), str(ref_id), "has_id"))
        tuples.add((str(complex_id), str(entity_id), "has_component"))

    for reaction_id, entity_id, ref_id, control_type in itertools.chain(controls, catalyses):
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
        tuples.add((str(reaction_id), str(entity_id), control_type))

    
    with open('ncbi.cif', 'w') as f:
        for t in tuples:
            f.write("\t".join(t) + '\n')


if __name__ == '__main__':
    main()


