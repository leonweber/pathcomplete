from lxml import etree
from sklearn.model_selection import train_test_split

if __name__ == '__main__':
    tree = etree.parse("events/data/BioCreative_BEL_Track/TrainingBioC_mentions.xml")
    documents = [i for i in tree.xpath("//document") if i.xpath("./passage")]
    train_documents, dev_documents = train_test_split(documents, train_size=0.9)
    with open("events/data/BioCreative_BEL_Track/train.bioc.xml", "wb") as f:
        root = etree.Element("collection")
        for doc in train_documents:
            root.append(doc)
        etree.ElementTree(root).write(f)
    with open("events/data/BioCreative_BEL_Track/dev.bioc.xml", "wb") as f:
        root = etree.Element("collection")
        for doc in dev_documents:
            root.append(doc)
        etree.ElementTree(root).write(f)

