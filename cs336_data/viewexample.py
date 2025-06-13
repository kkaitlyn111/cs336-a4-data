from fastwarc.stream_io import GZipStream
from fastwarc.warc import ArchiveIterator, WarcRecordType

wet_path = "/home/user/data/CC_example/CC-MAIN-20250430220529-20250501010529-00892.warc.wet.gz"  # <-- change this to your file

with GZipStream(open(wet_path, "rb")) as gz_stream:
    for idx, record in enumerate(ArchiveIterator(gz_stream, record_types=WarcRecordType.conversion), 1):
        if idx > 60:
            break
        raw_bytes = record.reader.read()
        text = raw_bytes.decode("utf-8", errors="replace")
        print(f"\n--- Document {idx} ---\n")
        print(text[:1000])  # print first 1000 chars for brevity