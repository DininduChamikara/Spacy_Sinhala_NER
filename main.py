import spacy
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

from spacy import displacy

nlp_ner = spacy.load("model-best")

doc1 = nlp_ner(
    "කර්ණාටක ප්‍රාන්තයේ අනපේක්ෂිත මැතිවරණ ප්‍රතිපලයෙන් පසු ඉන්දීය ජාතික කොන්ග්‍රසය නැවතත් ජයග්‍රාහි මාවතකට අවතීර්ණ වෙමින් සිටිනවා."
)
doc2 = nlp_ner(
    "පසුගිය සතියේ නිමාවට පත් වුණු ශ්‍රී ලංකා-අයර්ලන්ත දෙවන ටෙස්ට් තරගයේ දී වාර්තා කිහිපයක් අලුත් වුණා. ඒ අතුරින් කාගේත් අවධානය දිනා ගත්තේ ප්‍රභාත් ජයසූරිය බිහි කළ ලෝක වාර්තාව යි. එහි දී ඔහු අඩුම ටෙස්ට් තරග ප්‍රමාණයකින් කඩුලු 50 කඩඉම වෙත පැමිණි දඟපන්දු යවන්නා ලෙස වාර්තා අතරට එකතු වුණා."
)

colors = {
    "LOCATION": "#F67DE3",
    "PERSON": "#7DF609",
    "ORGANIZATION": "#A6E22D",
    "DATE": "#FFFF00",
    "TIME": "#800000",
}
options = {"colors": colors}

# server can only handle one visualization at a time on the same host and port combination
# displacy.serve(doc1, style="ent", options=options, page=True, host='localhost', port=5000)
# displacy.serve(doc2, style="ent", options=options, page=True, host='localhost', port=5000)

# print(doc2.ents)


# Extract named entities from documents
def extract_named_entities(doc):
    entities = [ent.text for ent in doc.ents]
    return " ".join(entities)


documents = [
    "කර්ණාටක ප්‍රාන්තයේ අනපේක්ෂිත මැතිවරණ ප්‍රතිපලයෙන් පසු ඉන්දීය ජාතික කොන්ග්‍රසය නැවතත් ජයග්‍රාහි මාවතකට අවතීර්ණ වෙමින් සිටිනවා.",
    "පසුගිය සතියේ නිමාවට පත් වුණු ශ්‍රී ලංකා-අයර්ලන්ත දෙවන ටෙස්ට් තරගයේ දී වාර්තා කිහිපයක් අලුත් වුණා. ඒ අතුරින් කාගේත් අවධානය දිනා ගත්තේ ප්‍රභාත් ජයසූරිය බිහි කළ ලෝක වාර්තාව යි. එහි දී ඔහු අඩුම ටෙස්ට් තරග ප්‍රමාණයකින් කඩුලු 50 කඩඉම වෙත පැමිණි දඟපන්දු යවන්නා ලෙස වාර්තා අතරට එකතු වුණා.",
    "වර්ෂ ගණනාවක් පුරා ශ්‍රී ලංකාවේ ජනප්‍රිය ක්‍රීඩා මෙන් ම ලොව පවතින ජනප්‍රියත ම ක්‍රීඩා ද්විත්වයක් වන පාපන්දු සහ රග්බි ක්‍රීඩාවලින් තවදුරටත් ජාත්‍යන්තරය නියෝජනය කිරීමට ශ්‍රී ලංකාවට අවස්ථාව ලැබෙන්නේ නෑ. එයට මෙම තහනම හේතු වී තිබෙනවා. පාපන්දු ක්‍රීඩාවෙන් මෑතකාලීන ව දැඩි පසුබෑමකට ලක් ව සිටින ශ්‍රී ලංකාව ශ්‍රේණිගත කිරීම් හි 207 වන ස්ථානය දක්වා පසුබැස තිබුණා. එය ඉතිහාසයේ ශ්‍රී ලංකා පාපන්දු කණ්ඩායමක් ලැබූ පහළ ම ස්ථානය යි.",
    "මේ වසරේ ජනවාරි මස 23 වැනි දා ජාත්‍යන්තර පාපන්දු සම්මේලනයේ (FIFA) වෙබ් අඩවිය වාර්තා කොට තිබෙන ලිපියකට අනුව ජනවාරි මස 14 වනදා ජාත්‍යන්තර පාපන්දු කාර්යංශය ගත් තීරණයක් අනුව ශ්‍රී ලංකා පාපන්දු තහනම් කිරීමට කටයුතු යෙදූ බව සඳහන් කොට තිබෙනවා.",
    "පාකිස්තානයේ හිටපු අගමැති ඉම්රාන් ඛාන්, දූෂණ චෝදනා මත පසුගිය සතියේ අඟහරුවා දා අත් අඩංගුවට ගැනීමත් සමඟ පාකිස්තානයේ දේශපාලන අර්බුදය තවදුරටත් තීව්‍ර වුණා.",
]


with open("./annotations_v1.0.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for annotation in data["annotations"]:
    documents.append(annotation[0])

# print(documents[0])

named_entities = [extract_named_entities(nlp_ner(doc)) for doc in documents]

# print(named_entities)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer()
tfidf_matrix = tfidf_vectorizer.fit_transform(named_entities)

# print(tfidf_matrix)

# Perform K-Means clustering
num_clusters = 10  # You can adjust the number of clusters
kmeans = KMeans(n_clusters=num_clusters, random_state=42)
kmeans.fit(tfidf_matrix)

cluster_labels = kmeans.labels_

# print(cluster_labels)

# Print the documents and their cluster assignments
for i, doc in enumerate(documents):
    if(cluster_labels[i] == 3):
        print(f"Document: {doc}\nCluster: {cluster_labels[i]}\n")
