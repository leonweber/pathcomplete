# Helper file for parsing a standoff file
import re
import copy
import networkx as nx

from util.utils import overlaps

N_SELF_LOOPS = 0

ENTITY_TRIGGER_TYPES = ['complex',
 'dna',
 'drug',
 'entity',
 'ion',
 'protein',
 'receptor',
 'rna',
 'simple_molecule',
 'simple_chemical',
 'tag',
 'gene',
 'gene_or_gene_product',
 'cellular_component']

EVENT_TRIGGER_TYPES = ['conversion',
 'catabolism',
 'dissociation',
 'inactivation',
 'regulation',
 'gene_expression',
 'catalysis',
 'positive_regulation',
 'pathway',
 'demethylation',
 'localization',
 'activation',
 'degradation',
 'transcription',
 'translation',
 'association',
 'ubiquitination',
 'protein_catabolism',
 'acetylation',
 'phosphorylation',
 'dephosphorylation',
 'methylation',
 'deubiquitination',
 'binding',
 'deacetylation',
 'negative_regulation',
 'transport',
 'hydroxylation',
 'dehydroxylation',
 'protein_modification',
 'none'
                       ]


ALL_TRIGGER_TYPES = ENTITY_TRIGGER_TYPES + EVENT_TRIGGER_TYPES


class TextAnnotation:
    id = ""
    type = ""
    type_lower = ""
    start = ""
    end = ""
    text = ""

    def __str__(self):
        return "<trigger {0}, {1}, {2}, {3}, '{4}'>".format( self.id, self.type, self.start, self.end, self.text)

    def to_a_star(self):
        return f"{self.id}\t{self.type} {self.start} {self.end}\t{self.text}"


class EntityTrigger(TextAnnotation):
    def get_url(self):
        return

class EventTrigger(TextAnnotation):
    pass


class Event():
    def __init__(self, ):
        self.id = None
        self.type = None
        self.type_lower = None
        self.trigger = None
        self.roles = []
        
    def get_roles( self, role, ignore_case = True):
        if ignore_case:
            return [r[1] for r in self.roles if r[0].lower() == role.lower()]
        else:
            return [r[1] for r in self.roles if r[0] == role]

    def __str__(self):
        return "<event {0}, {1}, trigger: {2}, roles: [{3}]>".format( self.id, self.type, self.trigger.id, ", ".join([ "{0}: {1}".format( r[0], r[1].id) for r in self.roles]))

    def __repr__(self):
        return str(self)

class MappingError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr( self.value)

def parse_line( line):
    """Transforms a standoff line into EntityTrigger, EventTrigger, or Event
       Throws a MappingError with information in case of error"""
    entity_trigger = None
    event_trigger = None
    event = None
    equivalence = None

    # handling trigger T[0-1]*
    if line.startswith( "T"):
        # format id \t type start end \t annotation
        split = line.strip().split('\t')
        trigger_type = split[1].split(' ')[0].lower()

         # entity trigger
        if trigger_type in ENTITY_TRIGGER_TYPES:
            # format id \t type start end \t annotation
            split = line.strip().split('\t')
            entity_trigger = EntityTrigger()
            entity_trigger.id = split[0]
            type_start_end = split[1].split(' ')
            entity_trigger.type = type_start_end[0]
            entity_trigger.type_lower = trigger_type
            entity_trigger.start = type_start_end[1]
            entity_trigger.end = type_start_end[2]
            entity_trigger.text = split[2]
            #TRIGGER[entity_trigger.id] =  entity_trigger
            #ENTITY_TRIGGER.append( entity_trigger)
        # event trigger
        elif trigger_type in EVENT_TRIGGER_TYPES:
            event_trigger = EventTrigger()
            event_trigger.id = split[0]
            type_start_end = split[1].split(' ')
            event_trigger.type = type_start_end[0]
            event_trigger.type_lower = trigger_type
            event_trigger.start = type_start_end[1]
            event_trigger.end = type_start_end[2]
            event_trigger.text = split[2]
            #TRIGGER[event_trigger.id] = event_trigger
        else:
            raise MappingError( "unknown trigger in " + line)
    # handling event E[0-1]*
    elif line.startswith("E"):
        # format id \t type role1 role2
        split = line.strip().split('\t')
        event = Event()
        event.id = split[0]

        roles = split[1].split(' ')

        # type
        type_trigger = roles[0].split(':')
        event.type = type_trigger[0]
        event.type_lower = event.type.lower()
        # try if we get the trigger
        try:
            event.trigger = type_trigger[1]
        except: 
            pass

        # roles
        for r in range( len(roles) - 1):
            role = roles[r+1]
            role_split = role.split(':')
            role_name = re.sub('[^a-zA-Z]*$','', role_split[0]) # remove trailing numbers from the role
            event.roles.append( (role_name, role_split[1]))
    elif line.startswith("*"):
        split = line.strip().split('\t')
        if len(split) > 3:
            if split[1] == "Equiv":
                equivalence = ( split[2], split[3])

        # EVENTS[event.id] = event
    return ( entity_trigger, event_trigger, event, equivalence)

def parse_lines( lines):
    triggers = {} # entities and event triggers
    entity_triggers = [] # entity triggers only
    events = {} # events annotations
    equivalences = []
    for i,line in enumerate(lines):
        try:
            entity_trigger, event_trigger, event, equivalence = parse_line( line)
            if entity_trigger != None:
                triggers[entity_trigger.id] =  entity_trigger
                entity_triggers.append( entity_trigger)
            if event_trigger != None:
                triggers[event_trigger.id] = event_trigger
            if event != None:
                events[event.id] = event
            if equivalence != None:
                equivalences.append( equivalence)
        except MappingError as e:
            print("ERROR " + ":" + str(i) + ": mapping error:" + e.value)

    # handle equivalences (add missing triggers and events)
    for equivalence in equivalences:
        t1 = equivalence[0]
        t2 = equivalence[1]
        old = None
        new = None
        
        if t1 in triggers and not t2 in triggers:
            old = triggers[t1]
        elif t2 in triggers and not t1 in triggers:
            old = triggers[t2]
        if old != None:
            # create a new trigger (copy the id of the old)
            new = copy.copy( old)
            if old.id == t1:
                new.id = t2
            else:
                new.id = t1
            triggers[new.id] = new
            if not old in entity_triggers:
                print("ERROR " +  " equivalence between non-entities cannot be handled (equivalence: " + equivalence + ")")
            else:
                entity_triggers.append( new)
        
    # replace all event roles with objects, replace trigger id with actual trigger
    for event in events.values():
        try:
            trigger = triggers[event.trigger]
            event.trigger = trigger
        except:
           print("ERROR " + ": entity trigger '" + event.trigger + "' not found")
            
        roles = []
        for role_tuple in event.roles:
            role_filler = None
            try:
                role_filler = events[role_tuple[1]]
            except:
                pass
            try:
                role_filler = triggers[role_tuple[1]]
            except:
                pass
            if role_filler == None:
                print("ERROR " + " entity/event '" + role_tuple[1] + "' not found")
            else:
                roles.append((role_tuple[0], role_filler))
        event.roles = roles

    return triggers, entity_triggers, events

def parse_a1_a2( a1_file_path, a2_file_path):
    with open(a1_file_path) as f:
        a1_lines = f.readlines()
    with open(a2_file_path) as f:
        a2_lines = f.readlines()
    return parse_lines(a1_lines + a2_lines)
    

def events_to_nx(events, triggers):
    G = nx.DiGraph()
    for trigger in triggers.values():
        G.add_node(trigger.id, type=trigger.type, text=trigger.text, span=(trigger.start, trigger.end))
    for event in events.values():
        G.add_node(event.id, type=event.type)
        G.add_edge(event.id, event.trigger.id, type="Trigger")
        for role, dst in event.roles:
            G.add_edge(event.id, dst.id, type=role)
    return G


def events_to_text_graph(events, triggers):
    G = nx.MultiDiGraph()
    global N_SELF_LOOPS
    added_signatures = set()
    for trigger in triggers.values():
        G.add_node(trigger.id, type=trigger.type, text=trigger.text, span=(trigger.start, trigger.end))
    for event in events.values():
        event_signature = [(event.trigger.id, "Trigger")]
        for role, dst in event.roles:
            if dst.id in triggers:
                dst_trigger_id = dst.id
            else:
                dst_trigger_id = events[dst.id].trigger.id
            event_signature.append((dst_trigger_id, role))
        event_signature = tuple(sorted(event_signature))

        if event_signature not in added_signatures:
            added_signatures.add(event_signature)
            G.add_node(event.id, type=event.type)
            G.add_edge(event.id, event.trigger.id, type="Trigger")
            for role, dst in event.roles:
                if dst.id in triggers:
                    dst_trigger_id = dst.id
                else:
                    dst_trigger_id = events[dst.id].trigger.id
                # trigger_span = (event.trigger.start, event.trigger.end)
                # dst_span = (triggers[dst_trigger_id].start, triggers[dst_trigger_id].end)
                # if not overlaps(trigger_span, dst_span): # No self-loops
                G.add_edge(event.id, dst_trigger_id, type=role)
    return G


class StandoffAnnotation:
    def __init__(self, a1_lines, a2_lines):
        self.triggers, self.entity_triggers, self.events = parse_lines(a1_lines + a2_lines)
        self.event_graph = events_to_nx(self.events, self.triggers)
        self.text_graph = events_to_text_graph(self.events, self.triggers)
        self.a1_lines = a1_lines
        self.a2_lines = a2_lines

    def contains_event(self, event):
        trigger_span = (event.trigger.start, event.trigger.end)
        for event_cand in self.events.values():
            if event_cand.type != event_cand.type:
                continue

            if not overlaps((event_cand.trigger.start, event_cand.trigger.end),
                            trigger_span):
                continue

            matched_roles = []
            for _, role in event.roles:
                try:
                    role_pos = (role.start, role.end)
                except AttributeError:
                    role_pos = (role.trigger.start, role.trigger.end)

                for _, role_cand in event_cand.roles:
                    try:
                        role_cand_pos = (role_cand.start, role_cand.end)
                    except AttributeError:
                        role_cand_pos = (role_cand.trigger.start, role_cand.trigger.end)

                    if overlaps(role_pos, role_cand_pos):
                        matched_roles.append(role)
                        break

            if len(matched_roles) == len(event.roles):
                return True

        return False





