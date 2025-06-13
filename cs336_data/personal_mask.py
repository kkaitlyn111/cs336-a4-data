import re

email_char = "|||EMAIL_ADDRESS|||"
phone_char = "|||PHONE_NUMBER|||"
ip_char = "|||IP_ADDRESS|||"

email_replacement = email_char

def mask_emails(text):
    initial_count = text.count(email_char)
    pattern = r"\b[\w.-]+@[\w.-]+\.\w+\b"
    masked_text = re.sub(pattern, email_char, text)
    num_masked = masked_text.count(email_char) - initial_count
    return masked_text, num_masked

def mask_phone_numbers(text):
    initial_count = text.count(phone_char)
    pattern = r"""
        (?:\(\d{3}\)[ -]?\d{3}[ -]?\d{4})|   # (xxx) xxx xxxx or (xxx)-xxx-xxxx
        (?:\d{3}-\d{3}-\d{4})|                # xxx-xxx-xxxx
        (?:\d{10})                             # xxxxxxxxxx
    """
    masked_text = re.sub(pattern, phone_char, text, flags=re.VERBOSE)
    num_masked = masked_text.count(phone_char) - initial_count
    return masked_text, num_masked

def mask_ips(text):
    initial_count = text.count(ip_char)
    ip_regex = r'\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b'
    ip_masked = re.sub(ip_regex, ip_char, text)
    return ip_masked, ip_masked.count(ip_char) - initial_count

if __name__ == "__main__":
    import random
    from cs336_data.text_extractor import extract_text_from_warc
    warc_path = "/home/user/data/CC_example/example.warc.wet.gz"  # Change path as needed
    docs = extract_text_from_warc(warc_path, max_records=100)
    results = []
    for doc in docs:
        masked, n_email = mask_emails(doc)
        masked, n_phone = mask_phone_numbers(masked)
        masked, n_ip = mask_ips(masked)
        total_replacements = n_email + n_phone + n_ip
        if total_replacements > 0:
            results.append((doc, masked, n_email, n_phone, n_ip))
    print(f"Found {len(results)} docs with at least one replacement.")
    if len(results) > 20:
        results = random.sample(results, 20)
    for i, (orig, masked, n_email, n_phone, n_ip) in enumerate(results, 1):
        print(f"--- Example {i} ---")
        print(f"Original: {orig[:300]}\nMasked: {masked[:300]}")
        print(f"Emails masked: {n_email}, Phones masked: {n_phone}, IPs masked: {n_ip}\n")
    print("\nLook for false positives (non-PII masked) and false negatives (PII not masked) in the above examples.")