import json
import sys
from collections import defaultdict

from pathme.reactome import rdf_sparql
from pathme import utils as pm_utils
import tqdm
import itertools
from rdflib.term import URIRef, Literal


GET_REACTION_LEFTS = """
SELECT ?reaction_id ?entity_id ?ref_id ?name ?location_name ?mod_type WHERE {
  ?reaction_id a biopax3:BiochemicalReaction ;
  biopax3:left ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id . }
  OPTIONAL {
  ?entity_id biopax3:displayName ?name . }
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

GET_TRANSPORT_LEFTS = """
SELECT ?reaction_id ?entity_id ?ref_id ?name ?location_name ?mod_type WHERE {
  ?reaction_id a biopax3:Transport ;
  biopax3:left ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?entity_id biopax3:displayName ?name . }
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

GET_TRANSPORT_WITH_REACTION_LEFTS = """
SELECT ?reaction_id ?entity_id ?ref_id ?name ?location_name ?mod_type WHERE {
  ?reaction_id a biopax3:TransportWithReaction ;
  biopax3:left ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?entity_id biopax3:displayName ?name . }
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

GET_COMPLEX_ASSEMBLY_LEFTS = """
SELECT ?reaction_id ?entity_id ?ref_id ?name ?location_name ?mod_type WHERE {
  ?reaction_id a biopax3:ComplexAssembly ;
  biopax3:left ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?entity_id biopax3:displayName ?name . }
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
SELECT ?reaction_id ?entity_id ?ref_id ?name ?location_name ?mod_type WHERE {
  ?reaction_id a biopax3:BiochemicalReaction ;
  biopax3:right ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?entity_id biopax3:displayName ?name . }
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

GET_TRANSPORT_RIGHTS = """
SELECT ?reaction_id ?entity_id ?ref_id ?name ?location_name ?mod_type WHERE {
  ?reaction_id a biopax3:Transport ;
  biopax3:right ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?entity_id biopax3:displayName ?name . }
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

GET_TRANSPORT_WITH_REACTION_RIGHTS = """
SELECT ?reaction_id ?entity_id ?ref_id ?name ?location_name ?mod_type WHERE {
  ?reaction_id a biopax3:TransportWithReaction ;
  biopax3:right ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?entity_id biopax3:displayName ?name . }
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

GET_COMPLEX_ASSEMBLY_RIGHTS = """
SELECT ?reaction_id ?entity_id ?ref_id ?name ?location_name ?mod_type WHERE {
  ?reaction_id a biopax3:ComplexAssembly ;
  biopax3:right ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?entity_id biopax3:displayName ?name . }
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
SELECT ?complex_id ?entity_id ?ref_id ?name WHERE {
  ?complex_id a biopax3:Complex ;
  biopax3:component ?entity_id .
  OPTIONAL {
  ?entity_id biopax3:entityReference ?ref_id
  }
  OPTIONAL {
  ?entity_id biopax3:displayName ?name . }
}
"""

GET_CONTROLS = """
SELECT ?reaction_id ?controller_id ?ref_id ?name ?control_type WHERE {
  ?control_id a biopax3:Control ;
  biopax3:controller ?controller_id .
  ?control_id  biopax3:controlled ?reaction_id .
  OPTIONAL {
  ?controller_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?controller_id biopax3:displayName ?name . }
  OPTIONAL {
  ?control_id biopax3:controlType ?control_type .
  }
}
"""

GET_CATALYSIS = """
SELECT ?reaction_id ?controller_id ?ref_id ?name ?control_type WHERE {
  ?control_id a biopax3:Catalysis ;
  biopax3:controller ?controller_id .
  ?control_id  biopax3:controlled ?reaction_id .
  OPTIONAL {
  ?controller_id biopax3:entityReference ?ref_id .
  }
  OPTIONAL {
  ?controller_id biopax3:displayName ?name . }
  OPTIONAL {
  ?control_id biopax3:controlType ?control_type .
  }
}
"""

GET_XREFS = """
SELECT ?id ?xref WHERE {
    ?id biopax3:xref ?xref .
}
"""


def main(input, output):
    graph = pm_utils.parse_rdf(input, format='xml')
    tuples = set()
    lefts1 = graph.query(GET_REACTION_LEFTS, initNs=rdf_sparql.PREFIXES)
    lefts2 = graph.query(GET_TRANSPORT_LEFTS, initNs=rdf_sparql.PREFIXES)
    lefts3 = graph.query(GET_TRANSPORT_WITH_REACTION_LEFTS, initNs=rdf_sparql.PREFIXES)
    lefts4 = graph.query(GET_COMPLEX_ASSEMBLY_LEFTS, initNs=rdf_sparql.PREFIXES)
    lefts = itertools.chain(lefts1, lefts2, lefts3, lefts4)

    rights1 = graph.query(GET_REACTION_RIGHTS, initNs=rdf_sparql.PREFIXES)
    rights2 = graph.query(GET_TRANSPORT_RIGHTS, initNs=rdf_sparql.PREFIXES)
    rights3 = graph.query(GET_TRANSPORT_WITH_REACTION_RIGHTS, initNs=rdf_sparql.PREFIXES)
    rights4 = graph.query(GET_COMPLEX_ASSEMBLY_RIGHTS, initNs=rdf_sparql.PREFIXES)
    rights = itertools.chain(rights1, rights2, rights3, rights4)

    complexes = graph.query(GET_COMPLEX_CONSTITUENTS, initNs=rdf_sparql.PREFIXES)
    controls = graph.query(GET_CONTROLS, initNs=rdf_sparql.PREFIXES)  
    catalyses = graph.query(GET_CATALYSIS, initNs=rdf_sparql.PREFIXES)

    xrefs = graph.query(GET_XREFS, initNs=rdf_sparql.PREFIXES)
    pm_ids = defaultdict(list)
    for id_, xref in xrefs:
        if 'pubmed' in xref:
            pm_ids[id_].append(xref)

    for reaction_id, entity_id, ref_id, name, location_name, modification in lefts:
        # if ref_id and 'uniprot' in ref_id:
        if ref_id:
            tuples.add((str(entity_id), str(ref_id), "has_id"))
        elif name:
            tuples.add((str(entity_id), str(name), "has_id"))


        if location_name:
            tuples.add((str(entity_id), str(location_name), "has_location"))

        if modification:
            tuples.add((str(entity_id), str(modification), "has_modification"))
            
        tuples.add((str(reaction_id), str(entity_id), "has_left"))

    for reaction_id, entity_id, ref_id, name, location_name, modification in rights:
        # if ref_id and 'uniprot' in ref_id:
        if ref_id:
            tuples.add((str(entity_id), str(ref_id), "has_id"))
        elif name:
            tuples.add((str(entity_id), str(name), "has_id"))

        if location_name:
            tuples.add((str(entity_id), str(location_name), "has_location"))

        if modification:
            tuples.add((str(entity_id), str(modification), "has_modification"))

        tuples.add((str(reaction_id), str(entity_id), "has_right"))

    for complex_id, entity_id, ref_id, name in complexes:
        # if ref_id and 'uniprot' in ref_id:
        if ref_id:
            tuples.add((str(entity_id), str(ref_id), "has_id"))
        elif name:
            tuples.add((str(entity_id), str(name), "has_id"))
        tuples.add((str(complex_id), str(entity_id), "has_component"))

    for reaction_id, entity_id, ref_id, name, control_type in itertools.chain(controls, catalyses):
        # if "BiochemicalReaction" not in reaction_id:
            # continue

        # if 'SmallMolecule' in entity_id or 'Rna' in entity_id or 'Pathway' in entity_id:
        #     continue
        if 'Pathway' in entity_id:
            continue

        if ref_id:
            entity_id = ref_id.split('/')[-1]
        elif name:
            tuples.add((str(entity_id), str(name), "has_id"))

        if not control_type:
            control_type = "regulation"
        else:
            control_type = control_type.lower()
        tuples.add((str(reaction_id), str(entity_id), control_type))

    with open(output + '.cif', 'w') as f:
        for t in tuples:
            f.write("\t".join(t) + '\n')

    with open(f'{output}_references.json', 'w') as f:
        json.dump(pm_ids, f)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])


