import re
import math
from urllib.parse import urlparse, parse_qs
import tldextract as _tldextract_module

# Use the bundled TLD list — avoids network calls inside containers.
_extractor = _tldextract_module.TLDExtract(suffix_list_urls=())

SUSPICIOUS_KEYWORDS = [
    'login', 'signin', 'sign-in', 'sign_in', 'account', 'secure',
    'security', 'update', 'verify', 'verification', 'confirm', 'confirmation',
    'bank', 'banking', 'payment', 'pay', 'wallet', 'password', 'passwd',
    'credential', 'paypal', 'amazon', 'google', 'facebook', 'apple',
    'microsoft', 'netflix', 'instagram', 'twitter', 'webscr', 'ebayisapi',
    'free', 'prize', 'winner', 'click', 'urgent', 'limited', 'offer',
]

URL_SHORTENERS = [
    'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'buff.ly',
    'short.io', 'tiny.cc', 'is.gd', 'cli.gs', 'tr.im', 'snipurl.com',
]

# Ordered list of feature names — MUST match extract_url_features output.
FEATURE_COLS_ORDER = [
    # URL-level
    'url_length', 'url_entropy', 'has_https', 'has_http', 'has_no_scheme',
    # Domain
    'domain_length', 'domain_entropy', 'num_dots_in_domain',
    'num_hyphens_in_domain', 'has_hyphen_in_domain',
    'tld_length', 'registered_domain_length', 'num_subdomains',
    'has_ip_address', 'has_port', 'port_value',
    # Path
    'path_length', 'path_depth',
    # Query / Fragment
    'query_length', 'num_query_params', 'has_fragment',
    # Character counts
    'num_dots', 'num_hyphens', 'num_underscores', 'num_slashes',
    'num_question_marks', 'num_equal_signs', 'num_ampersands',
    'num_percent_signs', 'num_at_symbols',
    'num_digits', 'num_letters', 'digit_ratio', 'letter_ratio',
    # Suspicious patterns
    'has_at_symbol', 'has_double_slash_path', 'has_hex_encoding',
    'num_suspicious_keywords', 'has_suspicious_keyword', 'is_url_shortened',
    # Word-level
    'num_words', 'longest_word_length', 'avg_word_length',
]  # 43 features


def _shannon_entropy(s: str) -> float:
    if not s:
        return 0.0
    prob = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in prob)


def extract_url_features(url: str) -> dict:
    url     = str(url).strip()
    url_low = url.lower()
    f       = {}

    f['url_length']  = len(url)
    f['url_entropy'] = _shannon_entropy(url)

    has_https        = url_low.startswith('https://')
    has_http         = url_low.startswith('http://')
    f['has_https']    = int(has_https)
    f['has_http']     = int(has_http)
    f['has_no_scheme']= int(not has_https and not has_http)

    url_for_parse = url if (has_https or has_http) else 'http://' + url
    try:
        parsed   = urlparse(url_for_parse)
        netloc   = parsed.netloc   or ''
        path     = parsed.path     or ''
        query    = parsed.query    or ''
        fragment = parsed.fragment or ''
    except Exception:
        netloc = path = query = fragment = ''

    domain = netloc.split(':')[0] if ':' in netloc else netloc

    f['domain_length']         = len(domain)
    f['domain_entropy']        = _shannon_entropy(domain)
    f['num_dots_in_domain']    = domain.count('.')
    f['num_hyphens_in_domain'] = domain.count('-')
    f['has_hyphen_in_domain']  = int('-' in domain)

    try:
        ext                         = _extractor(url_for_parse)
        tld                         = ext.suffix or ''
        subdomain                   = ext.subdomain or ''
        f['tld_length']              = len(tld)
        f['registered_domain_length']= len(ext.registered_domain or '')
        f['num_subdomains']          = len([s for s in subdomain.split('.') if s]) if subdomain else 0
    except Exception:
        parts                       = domain.split('.')
        f['tld_length']              = len(parts[-1]) if parts else 0
        f['registered_domain_length']= len(domain)
        f['num_subdomains']          = max(0, domain.count('.') - 1)

    # IP address as host → strong phishing indicator
    f['has_ip_address'] = int(
        bool(re.search(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b', domain))
    )

    try:
        port           = parsed.port
        f['has_port']   = int(port is not None)
        f['port_value'] = port if port is not None else 0
    except Exception:
        f['has_port']   = 0
        f['port_value'] = 0

    f['path_length'] = len(path)
    f['path_depth']  = len([p for p in path.split('/') if p])

    f['query_length']    = len(query)
    f['num_query_params']= len(parse_qs(query))
    f['has_fragment']    = int(bool(fragment))

    f['num_dots']          = url.count('.')
    f['num_hyphens']       = url.count('-')
    f['num_underscores']   = url.count('_')
    f['num_slashes']       = url.count('/')
    f['num_question_marks']= url.count('?')
    f['num_equal_signs']   = url.count('=')
    f['num_ampersands']    = url.count('&')
    f['num_percent_signs'] = url.count('%')
    f['num_at_symbols']    = url.count('@')

    num_digits  = sum(c.isdigit() for c in url)
    num_letters = sum(c.isalpha() for c in url)
    f['num_digits']   = num_digits
    f['num_letters']  = num_letters
    f['digit_ratio']  = num_digits  / max(len(url), 1)
    f['letter_ratio'] = num_letters / max(len(url), 1)

    f['has_at_symbol'] = int('@' in url)

    after_scheme               = url[url.find('//') + 2:] if '//' in url else url
    f['has_double_slash_path']  = int('//' in after_scheme)
    f['has_hex_encoding']       = int(bool(re.search(r'%[0-9a-fA-F]{2}', url)))

    num_kw                     = sum(kw in url_low for kw in SUSPICIOUS_KEYWORDS)
    f['num_suspicious_keywords']= num_kw
    f['has_suspicious_keyword'] = int(num_kw > 0)
    f['is_url_shortened']       = int(any(s in url_low for s in URL_SHORTENERS))

    words                   = [w for w in re.split(r'[^a-zA-Z0-9]', url) if w]
    f['num_words']           = len(words)
    f['longest_word_length'] = max((len(w) for w in words), default=0)
    f['avg_word_length']     = sum(len(w) for w in words) / len(words) if words else 0.0

    assert set(f.keys()) == set(FEATURE_COLS_ORDER), (
        f"Feature mismatch: {set(f.keys()).symmetric_difference(set(FEATURE_COLS_ORDER))}"
    )

    return f
