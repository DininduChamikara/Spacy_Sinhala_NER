import spacy

from spacy import displacy

nlp_ner = spacy.load("model-best")

doc1 = nlp_ner(
    "කර්ණාටක ප්‍රාන්තයේ අනපේක්ෂිත මැතිවරණ ප්‍රතිපලයෙන් පසු ඉන්දීය ජාතික කොන්ග්‍රසය නැවතත් ජයග්‍රාහි මාවතකට අවතීර්ණ වෙමින් සිටිනවා."
)
doc2 = nlp_ner(
    "පසුගිය සතියේ නිමාවට පත් වුණු ශ්‍රී ලංකා-අයර්ලන්ත දෙවන ටෙස්ට් තරගයේ දී වාර්තා කිහිපයක් අලුත් වුණා. ඒ අතුරින් කාගේත් අවධානය දිනා ගත්තේ ප්‍රභාත් ජයසූරිය බිහි කළ ලෝක වාර්තාව යි. එහි දී ඔහු අඩුම ටෙස්ට් තරග ප්‍රමාණයකින් කඩුලු 50 කඩඉම වෙත පැමිණි දඟපන්දු යවන්නා ලෙස වාර්තා අතරට එකතු වුණා."
)

colors = {"LOCATION": "#F67DE3", "PERSON": "#7DF609", "ORGANIZATION": "#A6E22D", "DATE": "#FFFF00", "TIME": "#800000"}
options = {"colors": colors}

# server can only handle one visualization at a time on the same host and port combination
# displacy.serve(doc1, style="ent", options=options, page=True, host='localhost', port=5000)
displacy.serve(doc2, style="ent", options=options, page=True, host='localhost', port=5000)