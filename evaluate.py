#!/usr/bin/env python3
"""
evaluate.py — Evaluation harness (17 questions, toutes catégories).
Voir README.md pour les détails.
"""
from __future__ import annotations
import json, os, sys, time, argparse
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openai
from src.application.index_usecase import BuildIndexUseCase
from src.application.query_usecase import QueryUseCase
from src.infrastructure.chunker import WikipediaChunker
from src.infrastructure.embedder import OpenAIEmbedder
from src.infrastructure.generator import OpenAIGenerator, OpenAITranslator
from src.infrastructure.hybrid_retriever import HybridRetriever
from src.infrastructure.loader import WikipediaLoader
from src.infrastructure.reranker import CrossEncoderReranker

EVAL_QUESTIONS = [
    {"id":1,"category":"Fait simple","question":"What is the capital of Madagascar?","expected":["Antananarivo"],"in_scope":True},
    {"id":2,"category":"Fait simple","question":"Quelle est la capitale de Madagascar ?","expected":["Antananarivo"],"in_scope":True},
    {"id":3,"category":"Fait simple","question":"What are the official languages of Madagascar?","expected":["Malagasy","French"],"in_scope":True},
    {"id":4,"category":"Chiffre précis","question":"What is the total area of Madagascar in square kilometres?","expected":["592"],"in_scope":True},
    {"id":5,"category":"Chiffre précis","question":"Quelle est la superficie de Madagascar en km² ?","expected":["592"],"in_scope":True},
    {"id":6,"category":"Lecture de tableau","question":"How many administrative regions does Madagascar have?","expected":["22","23"],"in_scope":True},
    {"id":7,"category":"Lecture de tableau","question":"What are the main ethnic groups of Madagascar?","expected":["Merina","Betsileo"],"in_scope":True},
    {"id":8,"category":"Raisonnement multi-passages","question":"How did political power change in Madagascar between 2009 and 2014?","expected":["coup","Rajoelina"],"in_scope":True},
    {"id":9,"category":"Raisonnement multi-passages","question":"What are the main economic sectors of Madagascar?","expected":["agriculture"],"in_scope":True},
    {"id":10,"category":"Ambiguïté temporelle","question":"Who is the current president of Madagascar?","expected":[],"in_scope":True},
    {"id":11,"category":"Ambiguïté temporelle","question":"What is the poverty rate in Madagascar?","expected":[],"in_scope":True},
    {"id":12,"category":"Ambiguïté temporelle","question":"What was the population of Madagascar in the most recent census in the article?","expected":[],"in_scope":True},
    {"id":13,"category":"Hors périmètre (piège)","question":"What is the national dish of Madagascar?","expected":[],"in_scope":False},
    {"id":14,"category":"Hors périmètre (piège)","question":"What is the current USD to Malagasy Ariary exchange rate?","expected":[],"in_scope":False},
    {"id":15,"category":"Hors périmètre (piège)","question":"Who directed the DreamWorks animated film Madagascar?","expected":[],"in_scope":False},
    {"id":16,"category":"Partiellement couverte","question":"What environmental challenges does Madagascar face?","expected":["forest","climate"],"in_scope":True},
    {"id":17,"category":"Partiellement couverte","question":"What recent political events occurred in Madagascar around 2023-2025?","expected":[],"in_scope":True},
]

REFUSAL_PHRASES = ["cannot answer","i cannot","not available","not found in","not mentioned",
                   "not present","no information","the page does not","based on the available",
                   "not covered","not provided","does not contain"]

def is_refusal(answer: str) -> bool:
    a = answer.lower()
    return any(p in a for p in REFUSAL_PHRASES)

def check_keywords(answer: str, keywords: list) -> bool:
    if not keywords: return True
    a = answer.lower()
    return any(kw.lower() in a for kw in keywords)

def run_evaluation(output_path: str = "eval_results.json"):
    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    retriever = HybridRetriever()
    embedder = OpenAIEmbedder(client)

    build_uc = BuildIndexUseCase(WikipediaLoader(), WikipediaChunker(), embedder, retriever)
    build_uc.execute()

    query_uc = QueryUseCase(
        retriever, embedder, OpenAIGenerator(client), OpenAITranslator(client),
        reranker=CrossEncoderReranker(),
    )

    results, n_correct, n_fp, n_fn, times = [], 0, 0, 0, []
    oos_total = sum(1 for q in EVAL_QUESTIONS if not q["in_scope"])

    print(f"\n{'='*70}\n  EVALUATION ({len(EVAL_QUESTIONS)} questions)\n{'='*70}\n")

    for q in EVAL_QUESTIONS:
        t0 = time.time()
        result = query_uc.execute(q["question"])
        elapsed = time.time() - t0
        times.append(elapsed)
        answer = result.answer
        refused = is_refusal(answer)
        kw_ok = check_keywords(answer, q["expected"])

        if q["in_scope"] and not refused and kw_ok:   verdict = "✅  CORRECT";             n_correct += 1
        elif q["in_scope"] and not refused:            verdict = "⚠️  ANSWERED (kw missing)"
        elif not q["in_scope"] and refused:            verdict = "✅  CORRECTLY REFUSED";   n_correct += 1
        elif q["in_scope"] and refused:                verdict = "❌  FALSE NEGATIVE";      n_fn += 1
        else:                                          verdict = "🚨  FALSE POSITIVE";      n_fp += 1

        print(f"[{q['id']:02d}] {q['category']}")
        print(f"  Q: {q['question']}")
        print(f"  A: {answer[:200]}{'…' if len(answer)>200 else ''}")
        print(f"  {verdict}  ({elapsed:.1f}s)\n")
        results.append({**{k:q[k] for k in("id","category","question","in_scope")},
                        "answer":answer,"refused":refused,"verdict":verdict,
                        "response_time_s":round(elapsed,2),"sources":result.dict()["sources"]})

    total = len(EVAL_QUESTIONS)
    fp_rate = (n_fp/oos_total*100) if oos_total else 0
    avg_t = sum(times)/len(times)
    print(f"\n{'='*70}\n  RÉSULTATS\n{'='*70}")
    print(f"  Total        : {total}")
    print(f"  Corrects     : {n_correct}  ({n_correct/total*100:.1f}%)")
    print(f"  Faux négatifs: {n_fn}")
    print(f"  Faux positifs: {n_fp}  ({fp_rate:.1f}% des hors-périmètre)")
    print(f"  Temps moyen  : {avg_t:.1f}s\n")

    payload = {"summary":{"total":total,"correct":n_correct,"accuracy_pct":round(n_correct/total*100,1),
                           "false_negatives":n_fn,"false_positives":n_fp,
                           "fp_rate_pct":round(fp_rate,1),"avg_response_time_s":round(avg_t,2)},
               "results":results}
    with open(output_path,"w",encoding="utf-8") as f:
        json.dump(payload,f,ensure_ascii=False,indent=2)
    print(f"Résultats → {output_path}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="eval_results.json")
    run_evaluation(p.parse_args().output)
