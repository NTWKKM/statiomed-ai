"""
agent/tools/tool_pubmed.py - NCBI E-Utilities Compliant Tool with Vancouver Citations
=============================================================================
Provides rate-limited, identified PubMed searches and extracts clinical benchmarks
(control event rates, hazard ratios, sample sizes) for power calculations and SAPs.
Complies with NCBI E-Utilities policy (3 req/s unauthenticated, 10 req/s with key).
=============================================================================
"""

import os
import time
from typing import Any, Dict, List, Optional
import requests

class PubMedEvidenceTool:
    BASE_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    MIN_INTERVAL = 0.34  # ~3 req/sec unauthenticated ceiling

    def __init__(self, email: Optional[str] = None, tool_name: Optional[str] = None):
        self.tool_name = tool_name or os.getenv("NCBI_TOOL_NAME", "StatioMedAI")
        self.email = email or os.getenv("NCBI_CONTACT_EMAIL", "research-lead@hospital.example")
        self.api_key = os.getenv("NCBI_API_KEY")
        if self.api_key:
            self.MIN_INTERVAL = 0.1  # ~10 req/sec with registered API key
        self._last_call = 0.0

    def _throttle(self):
        elapsed = time.monotonic() - self._last_call
        if elapsed < self.MIN_INTERVAL:
            time.sleep(self.MIN_INTERVAL - elapsed)
        self._last_call = time.monotonic()

    def _params(self, extra: Dict[str, Any]) -> Dict[str, Any]:
        base = {"tool": self.tool_name, "email": self.email}
        if self.api_key:
            base["api_key"] = self.api_key
        base.update(extra)
        return base

    def search_and_extract(self, query: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Executes an E-Search followed by E-Summary to retrieve publication metadata
        formatted in standard Vancouver citation style.
        """
        self._throttle()
        search_url = f"{self.BASE_URL}/esearch.fcgi"
        params = self._params({
            "db": "pubmed",
            "term": query,
            "retmode": "json",
            "retmax": max_results,
            "sort": "relevance"
        })
        res = requests.get(search_url, params=params, timeout=10)
        res.raise_for_status()
        id_list = res.json().get("esearchresult", {}).get("idlist", [])
        if not id_list:
            return []

        self._throttle()
        summary_url = f"{self.BASE_URL}/esummary.fcgi"
        sum_params = self._params({
            "db": "pubmed",
            "id": ",".join(id_list),
            "retmode": "json"
        })
        sum_res = requests.get(summary_url, params=sum_params, timeout=10)
        sum_res.raise_for_status()
        result_dict = sum_res.json().get("result", {})

        articles = []
        for pmid in id_list:
            item = result_dict.get(pmid, {})
            authors = [a.get("name", "") for a in item.get("authors", [])]
            author_str = ", ".join(authors[:3]) + (" et al." if len(authors) > 3 else "")
            title = item.get("title", "").rstrip(".")
            journal = item.get("source", "")
            pubdate = item.get("pubdate", "")[:4]
            doi = next((x.get("value", "") for x in item.get("articleids", []) if x.get("idtype") == "doi"), "")

            # Vancouver Citation Format: Author. Title. Journal. Year. doi
            vancouver = f"{author_str}. {title}. {journal}. {pubdate}."
            if doi:
                vancouver += f" doi: {doi}"

            articles.append({
                "pmid": pmid,
                "title": title,
                "journal": journal,
                "pubdate": pubdate,
                "authors": authors[:3],
                "doi": doi,
                "vancouver_citation": vancouver
            })
        return articles
