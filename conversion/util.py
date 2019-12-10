import requests

from conversion.make_dataset import HiddenPrints


def geneid_to_uniprot(symbol, mg):
    try:
        res = mg.query('%s' % symbol, size=1, fields='uniprot')['hits']
    except requests.exceptions.HTTPError:
        print("Couldn't find %s" % symbol)
        return None
    if res and 'uniprot' in res[0]:
        if 'Swiss-Prot' in res[0]['uniprot']:
            uniprot = res[0]['uniprot']['Swiss-Prot']
            if isinstance(uniprot, list):
                return uniprot
            else:
                return [uniprot]

    print("Couldn't find %s" % symbol)
    return None


def hgnc_to_uniprot(symbol, mapping, mg):
    try:
        symbol = mapping[symbol]
        return symbol
    except KeyError as ke:
        with HiddenPrints():
            res = mg.query('symbol:%s' % symbol, size=1, fields='uniprot')['hits']
        if res and 'uniprot' in res[0]:
            if 'Swiss-Prot' in res[0]['uniprot']:
                uniprot = res[0]['uniprot']['Swiss-Prot']
                return [uniprot]

        print("Couldn't find %s" % symbol)
        return None


def natural_language_to_uniprot(string, mg):
    string = string.replace("/", " ").replace("+", "").replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    with HiddenPrints():
        res = mg.query(string, size=1, fields='uniprot')['hits']
    if res and 'uniprot' in res[0]:
        if 'Swiss-Prot' in res[0]['uniprot']:
            uniprot = res[0]['uniprot']['Swiss-Prot']
            return uniprot

    return None


def get_pfam(uniprot, mg):
    with HiddenPrints():
        res = mg.query('uniprot:'+uniprot, size=1, fields='pfam')['hits']
