import fasttext
from cs336_data.text_extractor import extract_text, extract_text_from_warc

classifier_path = "/home/user/data/classifiers/lid.176.bin"


def find_language_scores(unicode_text:str):
    classifier = fasttext.load_model(classifier_path)
    unicode_text = unicode_text.strip()
    unicode_text = unicode_text.replace("\n", " ")
    languages, scores = classifier.predict(unicode_text)
    language = languages[0]
    if language == '__label__en':
        language = 'en'
    elif language == '__label__zh':
        language = 'zh'
    return language, scores[0]


def main():
    warc_path = "/home/user/data/CC_example/example.warc.wet.gz"
    docs = extract_text_from_warc(warc_path, max_records=20)
    for i, doc in enumerate(docs, 1):
        lang, score = find_language_scores(doc)
        print(f"--- Document {i} ---\n{doc}\n[Detected language: {lang}, Score: {score}]\n")

if __name__ == "__main__":
    main()

