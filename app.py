import json, boto3, base64, re, time, logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np
import pandas as pd
import streamlit as st
from bs4 import BeautifulSoup
from transformers import AutoTokenizer

from feature_utils import extract_url_features, FEATURE_COLS_ORDER

logging.basicConfig(level=logging.INFO)

HIGH_CONF   = 0.85
LOW_CONF    = 0.25
VLM_TRIGGER = 0.40
DOM_MAX_LEN = 512
CLASS_ORDER = ['benign', 'phishing', 'defacement', 'malware']

class Decision(str, Enum):
    AUTO_BLOCK   = 'AUTO_BLOCK'
    HUMAN_REVIEW = 'HUMAN_REVIEW'
    ALLOW        = 'ALLOW'

@dataclass
class PipelineResult:
    url:            str
    decision:       Decision
    combined_score: float
    url_score:      Optional[float] = None
    dom_score:      Optional[float] = None
    vlm_score:      Optional[float] = None
    vlm_triggered:  bool = False
    vlm_reasoning:  Optional[str] = None
    url_model_raw:  dict = field(default_factory=dict)
    dom_model_raw:  dict = field(default_factory=dict)
    errors:         list = field(default_factory=list)
    latency_ms:     dict = field(default_factory=dict)

# ── Cached AWS clients (init once per session) ────────────────────────────────
# Amazon SageMaker AI runtime for endpoint invocation; Amazon Bedrock runtime for VLM
@st.cache_resource
def get_clients(region: str):
    return (
        boto3.client('sagemaker-runtime', region_name=region),
        boto3.client('bedrock-runtime',   region_name=region),
        AutoTokenizer.from_pretrained('distilbert-base-uncased'),
    )

def run_url_model(sm, url, url_ep):
    try:
        feats    = extract_url_features(url)
        csv_body = pd.DataFrame([[feats[c] for c in FEATURE_COLS_ORDER]],
                                columns=FEATURE_COLS_ORDER).to_csv(header=False, index=False).encode()
        t0       = time.time()
        raw      = json.loads(sm.invoke_endpoint(
            EndpointName=url_ep, ContentType='text/csv', Accept='application/json', Body=csv_body,
        )['Body'].read())
        latency  = round((time.time() - t0) * 1000, 1)
        probs    = raw['probabilities'][0]
        p_mal    = round(1.0 - float(probs[0]), 4)
        return p_mal, {'p_malicious': p_mal, 'multiclass': CLASS_ORDER[int(np.argmax(probs))],
                       'probabilities': {c: round(float(p), 4) for c, p in zip(CLASS_ORDER, probs)}}, latency
    except Exception as e:
        return None, {'error': str(e)}, 0

def extract_dom_text(html, url=''):
    try:    soup = BeautifulSoup(html, 'lxml')
    except: soup = BeautifulSoup(html, 'html.parser')
    for tag in soup(['script', 'style', 'noscript']): tag.decompose()
    host  = url.split('/')[2] if '//' in url else ''
    parts = [
        f"[TITLE] {(soup.find('title') or soup.new_tag('x')).get_text(strip=True)[:400]}",
        '[META] ' + ' | '.join(m['content'][:100] for m in soup.find_all('meta', attrs={'name': True, 'content': True})
                               if m.get('name','').lower() in ('description','keywords','author')),
        '[FORMS] ' + ' | '.join(f"{f.get('method','get').upper()}:{f.get('action','')[:80]}" for f in soup.find_all('form')[:10]),
        '[INPUTS] ' + ' '.join(i.get('type','text').lower() for i in soup.find_all('input')[:20]),
        '[EXT_RESOURCES] ' + ' | '.join(s[:80] for tag in soup.find_all(['script','iframe','img','link'])
                                        for s in [tag.get('src') or tag.get('href') or '']
                                        if s.startswith('http') and host and host not in s)[:10],
        '[LINKS] ' + ' | '.join(a['href'][:80] for a in soup.find_all('a', href=True)[:20]),
        '[TEXT] ' + (soup.find('body') or soup).get_text(separator=' ', strip=True)[:1000],
    ]
    return ' '.join(parts)

def run_dom_model(sm, tokenizer, html, url, dom_ep):
    try:
        tokens  = tokenizer(extract_dom_text(html, url), max_length=DOM_MAX_LEN, truncation=True, return_tensors=None)
        payload = json.dumps({"inputs": tokenizer.decode(tokens['input_ids'], skip_special_tokens=True)})
        t0      = time.time()
        raw     = json.loads(sm.invoke_endpoint(
            EndpointName=dom_ep, ContentType='application/json', Accept='application/json', Body=payload,
        )['Body'].read())
        latency = round((time.time() - t0) * 1000, 1)
        if isinstance(raw, list): raw = raw[0]
        label   = str(raw.get('label', 'LABEL_0')).strip()
        score   = float(raw.get('score', 0.5))
        prob    = score if label in ('LABEL_1', 'M', '1', 1) else 1.0 - score
        return round(prob, 4), raw, latency
    except Exception as e:
        return None, {'error': str(e)}, 0

def run_vlm(bedrock, url, html, vlm_model, screenshot_b64=None):
    try:
        soup = BeautifulSoup(html, 'lxml')
        for t in soup(['script','style','noscript']): t.decompose()
        visible = soup.get_text(separator=' ', strip=True)[:800]
        prompt  = (
            f"You are a phishing detection expert. Analyze this webpage.\n\n"
            f"URL: {url}\nVisible text: {visible}\n\n"
            f"Assess phishing likelihood. Consider brand impersonation, urgency tactics, "
            f"credential harvesting forms, suspicious links, domain/brand mismatches.\n\n"
            f'Respond in JSON only: {{"phishing_probability": <0.0-1.0>, "confidence": "high|medium|low", '
            f'"reasoning": "<1-2 sentences>", "indicators": ["<indicator1>", "<indicator2>"]}}'
        )
        content = (
            [{"image": {"format": "png", "source": {"bytes": base64.b64decode(screenshot_b64)}}}, {"text": prompt}]
            if screenshot_b64 else [{"text": prompt}]
        )
        t0  = time.time()
        out = bedrock.converse(
            modelId=vlm_model,
            system=[{"text": "You are a cybersecurity expert. Respond with valid JSON only."}],
            messages=[{"role": "user", "content": content}],
            inferenceConfig={"maxTokens": 512, "temperature": 0.0},
        )
        latency = round((time.time() - t0) * 1000, 1)
        text    = out["output"]["message"]["content"][0]["text"]
        m       = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            parsed     = json.loads(m.group())
            prob       = float(parsed.get('phishing_probability', 0.5))
            reasoning  = parsed.get('reasoning', '')
            indicators = parsed.get('indicators', [])
            return round(prob, 4), f"{reasoning} Indicators: {', '.join(indicators)}", latency
        return 0.5, text[:200], latency
    except Exception as e:
        return 0.5, f"VLM error: {e}", 0

def combined_score(url_s, dom_s, vlm_s=None):
    pairs = [(s, w) for s, w in [(url_s, 0.35), (dom_s, 0.45), (vlm_s, 0.20)] if s is not None]
    if not pairs: return 0.5
    total = sum(w for _, w in pairs)
    return round(sum(s * w for s, w in pairs) / total, 4)

def run_pipeline(url, html, url_ep, dom_ep, vlm_model, region, enable_vlm, screenshot_b64=None):
    sm, bedrock, tokenizer = get_clients(region)
    r = PipelineResult(url=url, decision=Decision.ALLOW, combined_score=0.0)

    r.url_score, r.url_model_raw, r.latency_ms['url_model'] = run_url_model(sm, url, url_ep)
    if r.url_score is None: r.errors.append('url_model_failed')

    r.dom_score, r.dom_model_raw, r.latency_ms['dom_model'] = run_dom_model(sm, tokenizer, html, url, dom_ep)
    if r.dom_score is None: r.errors.append('dom_model_failed')

    prelim = combined_score(r.url_score, r.dom_score)
    if enable_vlm and LOW_CONF < prelim < HIGH_CONF and prelim > VLM_TRIGGER:
        r.vlm_score, r.vlm_reasoning, r.latency_ms['vlm'] = run_vlm(bedrock, url, html, vlm_model, screenshot_b64)
        r.vlm_triggered = True

    r.combined_score = combined_score(r.url_score, r.dom_score, r.vlm_score)
    r.decision = (
        Decision.AUTO_BLOCK   if r.combined_score >= HIGH_CONF else
        Decision.ALLOW        if r.combined_score <= LOW_CONF  else
        Decision.HUMAN_REVIEW
    )
    r.latency_ms['total'] = sum(v for v in r.latency_ms.values())
    return r

st.set_page_config(page_title="Phishing Detection Demo", layout="wide")
st.title("Multi-Model Phishing Detection")

with st.sidebar:
    st.header("Configuration")
    url_ep    = st.text_input("URL Endpoint (LightGBM)",  placeholder="phishing-lgbm-ep-...")
    dom_ep    = st.text_input("DOM Endpoint (DistilBERT)", placeholder="html-distilbert-endpoint...")
    vlm_model = st.text_input("VLM Model ID", value="us.anthropic.claude-sonnet-4-6")
    region    = st.text_input("AWS Region",   value="us-east-1")
    enable_vlm = st.toggle("Enable VLM (Claude)", value=True)
    st.divider()
    st.caption("Thresholds")
    st.caption(f"Auto-block ≥ {HIGH_CONF}  |  Allow ≤ {LOW_CONF}  |  VLM trigger > {VLM_TRIGGER}")

url  = st.text_input("URL to analyze", placeholder="https://example.com/page")
html = st.text_area("HTML content", height=200, placeholder="Paste raw HTML here...")

PRESETS = {
    "Chase phishing (credential harvesting)": {
        "url": "https://chase-bank-phish.invalid/verify",
        "html": """<html><head><title>Chase Bank - Urgent Account Verification Required</title></head>
<body><div>SECURITY ALERT: Suspicious login activity detected.</div>
<form method="POST" action="https://collect-creds.invalid/chase/submit">
  <input type="email" name="email" placeholder="Email Address">
  <input type="password" name="password" placeholder="Password">
  <button type="submit">Verify Account Now</button>
</form>
<script src="https://cdn.secure-banking-resources.invalid/track.js"></script></body></html>""",
    },
    "IRS tax refund lure (no form)": {
        "url": "https://irs-refund-verify.invalid/notice",
        "html": """<html><head><title>IRS - Important Tax Refund Notice</title></head>
<body><div>IMPORTANT: Your $2,847 tax refund requires immediate verification.</div>
<p>Verify your identity within 48 hours or your refund will be forfeited.</p>
<a href="https://irs-refund-verify.invalid/claim?token=abc123">Claim Your Refund Now</a>
</body></html>""",
    },
    "Benign tech blog": {
        "url": "https://techblog.example.com/home",
        "html": """<html><head><title>Welcome to Our Blog</title></head>
<body><h1>Latest Articles</h1>
<p>Read our latest posts about web development, design trends, and more.</p>
<a href="/articles/intro-to-css">Introduction to CSS</a>
<a href="/articles/js-tips">JavaScript Tips</a></body></html>""",
    },
}

with st.expander("Load a preset test case"):
    preset = st.selectbox("Preset", list(PRESETS.keys()))
    if st.button("Load preset"):
        st.session_state['preset_url']  = PRESETS[preset]['url']
        st.session_state['preset_html'] = PRESETS[preset]['html']
        st.rerun()

if 'preset_url' in st.session_state:
    url  = st.session_state.pop('preset_url')
    html = st.session_state.pop('preset_html')

run = st.button("Analyze", type="primary", disabled=not (url and html and url_ep and dom_ep))

if not url_ep or not dom_ep:
    st.info("Enter both endpoint names in the sidebar to enable analysis.")

if run and url and html:
    with st.spinner("Running pipeline..."):
        try:
            r = run_pipeline(url, html, url_ep, dom_ep, vlm_model, region, enable_vlm)
        except Exception as e:
            st.error(f"Pipeline error: {e}")
            st.stop()

    badge = {"AUTO_BLOCK": ("🔴 AUTO BLOCK", "red"), "HUMAN_REVIEW": ("🟡 HUMAN REVIEW", "orange"), "ALLOW": ("🟢 ALLOW", "#28a745")}
    label, color = badge[r.decision.value]
    st.markdown(f"<h2 style='color:{color}'>{label}</h2>", unsafe_allow_html=True)

    st.subheader(f"Combined score: {r.combined_score:.4f}")
    st.progress(r.combined_score)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("URL Score (LightGBM)", f"{r.url_score:.4f}" if r.url_score is not None else "failed")
        if r.url_score is not None: st.progress(r.url_score)
    with c2:
        st.metric("DOM Score (DistilBERT)", f"{r.dom_score:.4f}" if r.dom_score is not None else "failed")
        if r.dom_score is not None: st.progress(r.dom_score)
    with c3:
        vlm_label = f"{r.vlm_score:.4f}" if r.vlm_triggered else ("skipped" if enable_vlm else "disabled")
        st.metric("VLM Score (Claude)", vlm_label)
        if r.vlm_triggered and r.vlm_score is not None: st.progress(r.vlm_score)

    if r.vlm_triggered and r.vlm_reasoning:
        st.subheader("Claude VLM Reasoning")
        st.info(r.vlm_reasoning)

    lat = r.latency_ms
    st.caption(
        f"Latency — URL: {lat.get('url_model','?')}ms  |  "
        f"DOM: {lat.get('dom_model','?')}ms  |  "
        f"VLM: {lat.get('vlm', 'not run')}{'ms' if 'vlm' in lat else ''}  |  "
        f"Total: {lat.get('total','?')}ms"
    )

    if r.errors:
        st.warning(f"Errors: {r.errors}")

    with st.expander("Raw model outputs"):
        st.json({"url_model": r.url_model_raw, "dom_model": r.dom_model_raw})
