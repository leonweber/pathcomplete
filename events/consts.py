import re

PC13_EVENT_TYPES = {"Conversion", "Phosphorylation", "Dephosphorylation",
                    "Acetylation", "Deacetylation", "Methylation",
                    "Demethylation", "Ubiquitination", "Deubiquitination",
                    "Localization", "Transport", "Gene_expression",
                    "Transcription", "Translation", "Degradation",
                    "Activation", "Inactivation", "Binding",
                    "Dissociation", "Regulation", "Positive_regulation",
                    "Negative_regulation", "Pathway", "Hydroxylation", "Dehydroxylation",
                    "None","Protein_modification", "Protein_catabolism"
                    # "Fail"
                    }

PC13_ENTITY_TYPES = {"Simple_chemical", "Gene_or_gene_product", "Complex",
                     "Cellular_component", "None", "Protein", "Entity",
                     }
PC13_EDGE_TYPES = {"None", "Theme", "Product", "Cause", "Site", "AtLoc", "FromLoc", "ToLoc",
                    "Participant", "Trigger", "InText", "CSite"}
PC13_EDGE_TYPES_TO_MOD = {
    "Product": "resulting",
    "Site": "at",
    "AtLoc": "in",
    "FromLoc": "from",
    "ToLoc": "to",
    "Participant": "with",
}
PC13_RESULT_RE = r'.*==\[(?:ALL-)?TOTAL\]==.*?(\d[\d]?[\d]?\.\d\d)\s+(\d[\d]?[\d]?\.\d\d)\s+(\d[\d]?[\d]?\.\d\d)$'
PC13_EVAL_SCRIPT = 'evaluation-PC.py'
PC13_DUPLICATES_ALLOWED = {("Binding", "Theme"), ("Dissociation", "Product"),
                           ("Pathway", "Participant"), ("Conversion", "Theme")}
PC13_NO_THEME_ALLOWED = {"Conversion", "Pathway", "Binding", "Dissociation"}
PC13_MOLECULE = {"Simple_chemical", "Gene_or_gene_product", "Complex"}
PC13_EVENT_MODS = {"Speculation", "Negation"}


PC13_EVENT_TYPE_TO_ORDER = {}
for event in PC13_EVENT_TYPES:
    if "regulation" in event.lower():
        PC13_EVENT_TYPE_TO_ORDER[event] = 1
    else:
        PC13_EVENT_TYPE_TO_ORDER[event] = 0

GE_EVENT_TYPES = {
    'Gene_expression', 'Transcription', 'Protein_catabolism', 'Phosphorylation', 'Localization',
    'Regulation', 'Positive_regulation', 'Negative_regulation', 'Protein_modification', 'Binding',
    'Ubiquitination', 'Acetylation', 'Deacetylation'
                  }

GE_EDGE_TYPES_TO_MOD = {
    "Product": "resulting",
    "Site": "at",
    "AtLoc": "in",
    "FromLoc": "from",
    "ToLoc": "to",
    "Participant": "with",
}

GE_ENTITY_TYPES = {"Protein", "None", "Entity"}
GE_EDGE_TYPES = {"Theme", "None", "Cause", "ToLoc"}
GE_RESULT_RE = r'.*===\[TOTAL\]===.*?(\d[\d]?[\d]?\.\d\d)\s+(\d[\d]?[\d]?\.\d\d)\s+(\d[\d]?[\d]?\.\d\d)$'
GE_EVAL_SCRIPT = 'a2-evaluate.pl'
GE_DUPLICATES_ALLOWED = {("Binding", "Theme")}
GE_NO_THEME_FORBIDDEN = {}
GE_MOLECULE = {"Protein"}


PC13_EVENT_TYPE_TO_ORDER = {}
for event in PC13_EVENT_TYPES:
    if "regulation" in event.lower():
        PC13_EVENT_TYPE_TO_ORDER[event] = 1
    else:
        PC13_EVENT_TYPE_TO_ORDER[event] = 0


NODE_TYPES = {"token": 0, "word_type": 1}
EDGE_TYPES = {"entity_to_trigger": 0, "trigger_to_entity": 1}

TOKEN_LABELS_ENT = {"O-Entity": 0, "B-Entity": 1, "I-Entity": 2, "E-Entity": 3,
                    "S-Entity": 4}
TOKEN_LABELS_TRIG = {"O-Trigger": 0, "B-Trigger": 1, "I-Trigger": 2, "E-Trigger": 3,
                     "S-Trigger": 4}


