import fasttext
from cs336_data.text_extractor import extract_text
from cs336_data.text_extractor import extract_text_from_warc

hatespeech_classifier_path = "/home/user/data/classifiers/dolma_fasttext_hatespeech_jigsaw_model.bin"
nsfw_classifier_path = "/home/user/data/classifiers/dolma_fasttext_nsfw_jigsaw_model.bin"

def find_nsfw(unicode_text:str):
    classifier = fasttext.load_model(nsfw_classifier_path)
    unicode_text = unicode_text.replace("\n", " ")
    label, score = classifier.predict(unicode_text)
    label = label[0]
    if label == "__label__nsfw": label = "nsfw"
    if label == "__label__non-nsfw": label = "non-nsfw"
    return label, score[0]

def find_toxic(unicode_text:str):
    classifier = fasttext.load_model(hatespeech_classifier_path)
    unicode_text = unicode_text.replace("\n", " ")
    label, score = classifier.predict(unicode_text)
    label = label[0]
    if label == "__label__toxic": label = "toxic"
    if label == "__label__non-toxic": label = "non-toxic"
    return label, score[0]

def find_nsfw_with_model(unicode_text: str, classifier):
    unicode_text = unicode_text.replace("\n", " ")
    label, score = classifier.predict(unicode_text)
    label = label[0]
    if label == "__label__nsfw": label = "nsfw"
    if label == "__label__non-nsfw": label = "non-nsfw"
    return label, score[0]

def find_toxic_with_model(unicode_text: str, classifier):
    unicode_text = unicode_text.replace("\n", " ")
    label, score = classifier.predict(unicode_text)
    label = label[0]
    if label == "__label__toxic": label = "toxic"
    if label == "__label__non-toxic": label = "non-toxic"
    return label, score[0]

def main():
    warc_path = "/home/user/data/CC_example/example.warc.wet.gz"
    docs = extract_text_from_warc(warc_path, max_records=2000)
    nsfw_classifier = fasttext.load_model(nsfw_classifier_path)
    hatespeech_classifier = fasttext.load_model(hatespeech_classifier_path)
    nsfw_examples = []
    toxic_examples = []
    for doc in docs:
        nsfw_label, nsfw_score = find_nsfw_with_model(doc, nsfw_classifier)
        toxic_label, toxic_score = find_toxic_with_model(doc, hatespeech_classifier)
        if nsfw_label == "nsfw" and len(nsfw_examples) < 10:
            nsfw_examples.append((doc, nsfw_label, nsfw_score, toxic_label, toxic_score))
        if toxic_label == "toxic" and len(toxic_examples) < 10:
            toxic_examples.append((doc, nsfw_label, nsfw_score, toxic_label, toxic_score))
        if len(nsfw_examples) >= 10 and len(toxic_examples) >= 10:
            break
    print("\n--- NSFW Examples (up to 10) ---\n")
    for i, (doc, nsfw_label, nsfw_score, toxic_label, toxic_score) in enumerate(nsfw_examples, 1):
        print(f"--- NSFW Document {i} ---\n{doc}\n[NSFW: {nsfw_label}, NSFW Score: {nsfw_score}, Toxic: {toxic_label}, Toxic Score: {toxic_score}]\n")
    print("\n--- Toxic Examples (up to 10) ---\n")
    for i, (doc, nsfw_label, nsfw_score, toxic_label, toxic_score) in enumerate(toxic_examples, 1):
        print(f"--- Toxic Document {i} ---\n{doc}\n[NSFW: {nsfw_label}, NSFW Score: {nsfw_score}, Toxic: {toxic_label}, Toxic Score: {toxic_score}]\n")

if __name__ == "__main__":
    main()