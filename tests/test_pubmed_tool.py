"""
tests/test_pubmed_tool.py - Unit Tests for Rate-Limited PubMed Evidence Tool
"""

from unittest.mock import patch, MagicMock
from agent.tools.tool_pubmed import PubMedEvidenceTool

def test_pubmed_tool_vancouver_formatting():
    tool = PubMedEvidenceTool()

    mock_search_resp = MagicMock()
    mock_search_resp.json.return_value = {
        "esearchresult": {"idlist": ["12345678"]}
    }

    mock_summary_resp = MagicMock()
    mock_summary_resp.json.return_value = {
        "result": {
            "12345678": {
                "title": "Clinical Efficacy of SGLT2 Inhibitors in Heart Failure.",
                "authors": [{"name": "Smith J"}, {"name": "Taylor R"}, {"name": "Brown A"}, {"name": "Johnson D"}],
                "source": "N Engl J Med",
                "pubdate": "2023 Sep 15",
                "articleids": [{"idtype": "doi", "value": "10.1056/NEJMoa2300000"}]
            }
        }
    }

    with patch("requests.get", side_effect=[mock_search_resp, mock_summary_resp]):
        results = tool.search_and_extract(query="SGLT2 heart failure", max_results=1)

        assert len(results) == 1
        art = results[0]
        assert art["pmid"] == "12345678"
        assert "Smith J, Taylor R, Brown A et al." in art["vancouver_citation"]
        assert "N Engl J Med. 2023." in art["vancouver_citation"]
        assert "doi: 10.1056/NEJMoa2300000" in art["vancouver_citation"]
