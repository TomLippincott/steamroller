from concrete import Communication, AnnotationMetadata, CommunicationTagging, Section, Sentence, TextSpan, CommunicationTagging
import gzip
import codecs
from concrete.util.file_io import CommunicationWriterTGZ
from concrete.util.concrete_uuid import AnalyticUUIDGeneratorFactory
from time import time

writer = codecs.getwriter("utf-8")
reader = codecs.getreader("utf-8")

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest="input")
    parser.add_argument("-o", "--output", dest="output")
    parser.add_argument("-t", "--tag_type", dest="tag_type", default="attribute")
    options = parser.parse_args()

    ugf = AnalyticUUIDGeneratorFactory()
    ofd = CommunicationWriterTGZ(options.output)
    with reader(gzip.open(options.input)) as ifd:
        for i, line in enumerate(ifd):
            cid, label, text = line.strip().split("\t")
            g = ugf.create()
            t = int(time())
            comm = Communication(id=cid,
                                 uuid=g.next(),
                                 type="Text document",
                                 text=text,
                                 communicationTaggingList=[CommunicationTagging(uuid=g.next(),
                                                                                metadata=AnnotationMetadata(tool="Gold labeling",
                                                                                                            timestamp=t,
                                                                                                            kBest=1,
                                                                                ),
                                                                                taggingType=options.tag_type,
                                                                                tagList=[label],
                                                                                confidenceList=[1.0],
                                 )],
                                 metadata=AnnotationMetadata(tool="text_to_concrete.py ingester", timestamp=t, kBest=1),
                                 sectionList=[Section(uuid=g.next(),
                                                      textSpan=TextSpan(start=0, ending=len(text)),
                                                      kind="content",
                                                      )
                                 ])
            ofd.write(comm)
    ofd.close()
