PC13_EVENT_TYPES = {"Conversion", "Phosphorylation", "Dephosphorylation",
                    "Acetylation", "Deacetylation", "Methylation",
                    "Demethylation", "Ubiquitination", "Deubiquitination",
                    "Localization", "Transport", "Gene_expression",
                    "Transcription", "Translation", "Degradation",
                    "Activation", "Inactivation", "Binding",
                    "Dissociation", "Regulation", "Positive_regulation",
                    "Negative_regulation", "Pathway", "Hydroxylation", "Dehydroxylation",
                    "None"}

PC13_ENTITY_TYPES = {"Simple_chemical", "Gene_or_gene_product", "Complex",
                     "Cellular_component", "None"}
PC13_EDGE_TYPES = {"None", "Theme", "Product", "Cause", "Site", "AtLoc", "FromLoc", "ToLoc",
                    "Participant", "Trigger", "InText"}
PC13_RESULT_RE = r'.*===\[TOTAL\]===.*?(\d[\d]?[\d]?\.\d\d)\s+(\d[\d]?[\d]?\.\d\d)\s+(\d[\d]?[\d]?\.\d\d)$'
PC13_EVAL_SCRIPT = 'evaluation-PC.py'
PC13_DUPLICATES_ALLOWED = {("Binding", "Theme"), ("Dissociation", "Product")}
PC13_NO_THEME_FORBIDDEN = {"Site"}
PC13_EDGES_FORBIDDEN = {("ToLoc", "Gene_or_gene_product")}

NODE_TYPES = {"token": 0, "word_type": 1}
EDGE_TYPES = {"entity_to_trigger": 0, "trigger_to_entity": 1}

TOKEN_LABELS_ENT = {"O-Entity": 0, "B-Entity": 1, "I-Entity": 2, "E-Entity": 3,
                    "S-Entity": 4}
TOKEN_LABELS_TRIG = {"O-Trigger": 0, "B-Trigger": 1, "I-Trigger": 2, "E-Trigger": 3,
                     "S-Trigger": 4}


