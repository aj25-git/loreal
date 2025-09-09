#!/usr/bin/env python3
"""
CommentSense: AI-powered comment analysis prototype

Features:
- Text cleaning (emoji/HTML/URLs)
- Language detection (basic)
- Sentiment analysis (VADER baseline)
- Spam detection (heuristics)
- Relevance scoring (brand keywords)
- Category tagging (skincare, makeup, fragrance, haircare)
- Quality scoring (weighted relevance + richness + sentiment + engagement)
- KPIs: quality ratio, spam ratio, sentiment distribution, quality by category
- Outputs: enriched CSV + console KPIs
- Optional: run with Streamlit to launch an interactive dashboard

Usage:
  CLI (analysis only):
      python run_commentsense.py --in data/comments.csv --out results.csv

  Dashboard:
      streamlit run run_commentsense.py
"""

import argparse, re, html, unicodedata
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from emoji import replace_emoji
from ftfy import fix_text
from text_unidecode import unidecode
from langdetect import detect, DetectorFactory
from nltk.sentiment import SentimentIntensityAnalyzer

# reproducible langdetect
DetectorFactory.seed = 42

# -----------------------
# CONFIG (edit to match your dataset)
# -----------------------
CONFIG = {
    "COL_TEXT": "comment_text",    # column in CSV with raw comment text
    "COL_LIKES": "like_count",     # optional, set None if not available
    "COL_LANG": None,              # if dataset already has a language column
    "BRAND_KEYWORDS": [
        "loreal","l'oréal","maybelline","nyx","garnier","lancome","kiehl","kiehl's",
        "revitalift","elseve","age perfect","true match","telescopic"
    ],
    "CATEGORY_KEYWORDS": {
        "skincare":["serum","moisturizer","cleanser","toner","retinol","hyaluronic","spf","sunscreen"],
        "makeup":["foundation","concealer","lipstick","mascara","eyeliner","blush","highlighter"],
        "fragrance":["perfume","eau de parfum","fragrance","scent","cologne"],
        "haircare":["shampoo","conditioner","hair mask","hair oil","dandruff","hairfall","styling"]
    },
    "QUALITY_WEIGHTS": {
        "relevance":0.45, "richness":0.25, "sentiment":0.20, "engagement":0.10
    },
    "QUALITY_THRESHOLD":0.55
}

# -----------------------
# Text cleaning
# -----------------------
def clean_text(raw: str) -> str:
    if not isinstance(raw, str):
        return ""
    txt = fix_text(str(raw))
    txt = html.unescape(txt)
    txt = unicodedata.normalize("NFKC", txt)
    txt = BeautifulSoup(txt,"html.parser").get_text(" ")
    txt = replace_emoji(txt, replace=lambda e,d: f" :{d['en']}: ")
    txt = txt.lower()
    txt = re.sub(r"https?://\S+|www\.\S+"," ",txt)
    txt = re.sub(r"[^\w\s:']", " ", txt)
    txt = re.sub(r"\s+"," ",txt).strip()
    return unidecode(txt)

def detect_language_safe(text:str)->str:
    try:
        if not text or len(text.split())<2: return "und"
        return detect(text)
    except Exception: return "und"

# -----------------------
# Sentiment (VADER)
# -----------------------
_sia = SentimentIntensityAnalyzer()
def sentiment_score(text:str)->float:
    if not text: return 0.0
    return float(_sia.polarity_scores(text)["compound"])
def sentiment_label(score:float)->str:
    if score>=0.05: return "positive"
    if score<=-0.05: return "negative"
    return "neutral"

# -----------------------
# Relevance & categories
# -----------------------
def brand_relevance(text:str, terms:List[str])->float:
    hits = sum(1 for t in terms if t in text)
    return hits/max(1,len(terms))
def category_match(text:str, taxonomy:Dict[str,List[str]])->Tuple[str,int]:
    best,score="other",0
    for cat,words in taxonomy.items():
        hits=sum(1 for w in words if w in text)
        if hits>score: best,score=cat,hits
    return best,score

# -----------------------
# Spam detection heuristics
# -----------------------
SPAM_URL_RE=re.compile(r"(https?://|www\.)")
SPAM_INVITE_RE=re.compile(r"(dm|inbox|follow\s+me|check\s+my\s+profile)")
SPAM_MENTION_RE=re.compile(r"@\w+")
def spam_score(text:str)->float:
    if not text: return 0.0
    score=0.0
    if SPAM_URL_RE.search(text): score+=0.35
    if SPAM_INVITE_RE.search(text): score+=0.25
    if len(SPAM_MENTION_RE.findall(text))>=3: score+=0.2
    if re.search(r"(.)\1{4,}",text): score+=0.15
    toks=text.split()
    short_ratio=sum(1 for t in toks if len(t)<=2)/max(1,len(toks))
    score+=min(0.25,0.5*short_ratio)
    return min(1.0,score)
def is_spam(score:float,threshold:float=0.6)->bool:
    return score>=threshold

# -----------------------
# Quality score
# -----------------------
def richness_score(text:str)->float:
    toks=[t for t in text.split() if t.isalpha()]
    if not toks: return 0.0
    uniq=len(set(toks))/len(toks)
    longw=sum(1 for t in toks if len(t)>=6)/len(toks)
    return float(np.clip(0.6*uniq+0.4*longw,0,1))
def engagement_score(likes:Optional[float])->float:
    if likes is None or (likes!=likes): return 0.0
    return float(np.clip(np.log1p(likes)/np.log(1+1000),0,1))
def quality_score(rel:float,rich:float,sent:float,eng:float)->float:
    w=CONFIG["QUALITY_WEIGHTS"]
    sent01=(sent+1)/2.0
    return float(np.clip(w["relevance"]*rel+w["richness"]*rich+w["sentiment"]*sent01+w["engagement"]*eng,0,1))

# -----------------------
# Pipeline
# -----------------------
def process_df(df:pd.DataFrame)->pd.DataFrame:
    C=CONFIG; ct=C["COL_TEXT"]; cl=C["COL_LIKES"]; clang=C["COL_LANG"]

    if ct not in df.columns: raise ValueError(f"Missing column {ct}")

    df["_cleaned"]=df[ct].map(clean_text)
    df["_lang"]=df[clang] if (clang and clang in df.columns) else df["_cleaned"].map(detect_language_safe)
    df["_sent"]=df["_cleaned"].map(sentiment_score)
    df["_sent_label"]=df["_sent"].map(sentiment_label)
    df["_rel"]=df["_cleaned"].map(lambda t: brand_relevance(t,[k.lower() for k in C["BRAND_KEYWORDS"]]))
    cats=df["_cleaned"].map(lambda t: category_match(t,{k:[w.lower() for w in v] for k,v in C["CATEGORY_KEYWORDS"].items()}))
    df["_category"]=cats.map(lambda x:x[0]); df["_cat_hits"]=cats.map(lambda x:x[1])
    df["_spam"]=df["_cleaned"].map(spam_score); df["_is_spam"]=df["_spam"].map(lambda s:is_spam(s))
    df["_rich"]=df["_cleaned"].map(richness_score)
    likes=df[cl] if (cl and cl in df.columns) else pd.Series([np.nan]*len(df))
    df["_eng"]=likes.map(engagement_score)
    df["_quality"]=df.apply(lambda r: quality_score(r["_rel"],r["_rich"],r["_sent"],r["_eng"]),axis=1)
    df["_is_quality"]=df["_quality"]>=C["QUALITY_THRESHOLD"]
    return df

def compute_kpis(df:pd.DataFrame)->Dict[str,Any]:
    return {
        "total_comments":int(len(df)),
        "quality_ratio":float(df["_is_quality"].mean()),
        "spam_ratio":float(df["_is_spam"].mean()),
        "sentiment_dist":df["_sent_label"].value_counts(normalize=True).to_dict(),
        "by_category":df.groupby("_category")["_is_quality"].mean().to_dict(),
    }

def cli():
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="inp",default="data/comments.csv")
    ap.add_argument("--out",dest="out",default="results.csv")
    args=ap.parse_args()

    df=pd.read_csv(args.inp)
    out=process_df(df)
    keep=[c for c in df.columns]+["_cleaned","_lang","_sent","_sent_label","_rel","_category","_spam","_is_spam","_rich","_eng","_quality","_is_quality"]
    out[keep].to_csv(args.out,index=False)
    kpis=compute_kpis(out)
    print("=== KPIs ===")
    print("Total:",kpis["total_comments"])
    print("Quality ratio:",round(kpis["quality_ratio"],3))
    print("Spam ratio:",round(kpis["spam_ratio"],3))
    print("Sentiment dist:",kpis["sentiment_dist"])
    print("Quality by category:",{k:round(v,3) for k,v in kpis["by_category"].items()})

if __name__=="__main__":
    import sys
    if "streamlit" in sys.argv[0]:
        import streamlit as st
        st.title("CommentSense Dashboard")

        inp=st.sidebar.text_input("CSV file","results.csv")
        try:
            df=pd.read_csv(inp)
            st.success(f"Loaded {len(df)} comments")
            st.metric("Total",df.shape[0])
            st.metric("Quality %",f"{df['_is_quality'].mean()*100:.1f}")
            st.metric("Spam %",f"{df['_is_spam'].mean()*100:.1f}")
            st.bar_chart(df["_sent_label"].value_counts())
            st.bar_chart(df.groupby("_category")["_is_quality"].mean())
            st.dataframe(df[["_cleaned","_sent_label","_quality","_is_quality","_spam","_category"]].head(15))
        except Exception as e:
            st.error(str(e))
    else:
        cli()
