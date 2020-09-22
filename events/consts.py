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
PC13_NO_ARGUMENT_ALLOWED = {"Pathway"}
PC13_MOLECULE = {"Simple_chemical", "Gene_or_gene_product", "Complex"}
PC13_EVENT_MODS = {"Speculation", "Negation"}


PC13_EVENT_TYPE_TO_ORDER = {}
for event in PC13_EVENT_TYPES:
    if "regulation" in event.lower():
        PC13_EVENT_TYPE_TO_ORDER[event] = 1
    else:
        PC13_EVENT_TYPE_TO_ORDER[event] = 0

