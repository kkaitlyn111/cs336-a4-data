blacklist_immediate = {
    "{", "CAPTCHA", "have been blocked","privacy policy",
    "use cookies",
    "use of cookies", "outdated browser"}


def filter_blacklist(text: str):
    # Immediate blacklist: if any phrase is found, return immediately
    for word in blacklist_immediate:
        if word in text:
            print(f"immediate blacklist: {word}")
            return False

    return True




