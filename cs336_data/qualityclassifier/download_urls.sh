wget --timeout=5 \
     --tries=3 \
     -i "/home/user/cs336-a4-data/datagen/positive_urls.txt" \
     --warc-file="/home/user/cs336-a4-data/datagen/positive_urls" \
     -O /dev/null