from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding, bytes_to_str
from resiliparse.extract.html2text import extract_plain_text

def extract_text(byte_string:bytes) -> str:
    encoding = detect_encoding(byte_string)
    text = bytes_to_str(byte_string, encoding)
    return extract_plain_text(text)


MAX_CONTENT_LENGTH = 1000

def extract_text_from_warc(warc_file_path: str, max_records: int = 5, max_content_length: int = MAX_CONTENT_LENGTH) -> list:
    texts = []
    with open(warc_file_path, "rb") as f:
        for record in ArchiveIterator(f):
            try:
                content_bytes = record.reader.read()
                text = extract_text(content_bytes)
                if len(text) > max_content_length:
                    text = text[:max_content_length] + "..."
                texts.append(text)
            except Exception as e:
                print(f"Skipping record due to error: {e}")
            if len(texts) >= max_records:
                break
    return texts

if __name__ == "__main__":
    records = extract_text_from_warc("/home/user/data/CC_example/example.warc.gz", max_records=5)
    for i, rec in enumerate(records, 1):
        print(f"--- WARC Record {i} ---\n{rec}\n")
    records = extract_text_from_warc("/home/user/data/CC_example/example.warc.wet.gz", max_records=5)
    for i, rec in enumerate(records, 1):
        print(f"--- WET Record {i} ---\n{rec}\n")