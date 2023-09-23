import json
import spacy

from spacy.tokens import DocBin
from tqdm import tqdm

with open("./annotations_v1.0.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# print(data['classes'])
# print(data['annotations'][0][0])
# print(data['annotations'][0][1])
# print(data['annotations'][0][1]['entities'])
# print(data['annotations'][0][0][0:7])

nlp = spacy.blank("si")  # load a new spacy model
doc_bin = DocBin()

from tqdm.gui import tqdm
from spacy.util import filter_spans

for training_example in tqdm(data["annotations"]):
    text = training_example[0]
    labels = training_example[1]["entities"]
    doc = nlp.make_doc(text)
    ents = []
    for start, end, label in labels:
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entry")
        else:
            ents.append(span)
    filtered_ents = filter_spans(ents)
    doc.ents = filtered_ents
    doc_bin.add(doc)

doc_bin.to_disk("train.spacy")

# from spacy import displacy

# nlp_ner = spacy.load("model-best")

# doc1 = nlp_ner(
#     "කර්ණාටක ප්‍රාන්තයේ අනපේක්ෂිත මැතිවරණ ප්‍රතිපලයෙන් පසු ඉන්දීය ජාතික කොන්ග්‍රසය නැවතත් ජයග්‍රාහි මාවතකට අවතීර්ණ වෙමින් සිටිනවා."
# )
# doc2 = nlp_ner(
#     "පසුගිය සතියේ නිමාවට පත් වුණු ශ්‍රී ලංකා-අයර්ලන්ත දෙවන ටෙස්ට් තරගයේ දී වාර්තා කිහිපයක් අලුත් වුණා. ඒ අතුරින් කාගේත් අවධානය දිනා ගත්තේ ප්‍රභාත් ජයසූරිය බිහි කළ ලෝක වාර්තාව යි. එහි දී ඔහු අඩුම ටෙස්ට් තරග ප්‍රමාණයකින් කඩුලු 50 කඩඉම වෙත පැමිණි දඟපන්දු යවන්නා ලෙස වාර්තා අතරට එකතු වුණා."
# )

# colors = {"LOCATION": "#F67DE3", "PERSON": "#7DF609", "ORGANIZATION": "#A6E22D", "DATE": "#FFFF00", "TIME": "#800000"}
# options = {"colors": colors}

# displacy.serve(doc1, style="ent", options=options)