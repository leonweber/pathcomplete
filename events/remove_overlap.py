from lxml import etree

if __name__ == '__main__':

    test_texts = set()
    with open("events/data/BioCreative_BEL_Track/SampleSet.tab") as f:
        lines = f.readlines()
        for line in lines:
            test_texts.add(line.split("\t")[2])

    train_texts = set()
    tree = etree.parse("events/data/BioCreative_BEL_Track/train.bioc.xml")
    for document in tree.xpath('//document'):
        for passage in document.xpath("./passage"):
            text = passage.xpath("./text")[0].text
            if text in test_texts:
                document.remove(passage)
    with open("events/data/BioCreative_BEL_Track/train_no_overlap.bioc.xml", "wb") as f:
        tree.write(f)


