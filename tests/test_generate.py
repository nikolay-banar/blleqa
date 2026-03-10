from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

OPENROUTER_API_KEY_SET = bool(os.getenv("OPENROUTER_API_KEY"))


@pytest.mark.skipif(
    not OPENROUTER_API_KEY_SET,
    reason="set OPENROUTER_API_KEY for real generation test",
)
def test_agenerate_with_real_requests() -> None:
    from generation.pipeline.open_router import agenerate

    model_name = os.getenv("GENERATE_TEST_MODEL", "mistralai/ministral-8b-2512")
    provider_name = os.getenv("GENERATE_TEST_PROVIDER")

    model_config = {"name": model_name}
    if provider_name:
        model_config["providers"] = [provider_name]

    results, errors = asyncio.run(
        agenerate(
            inputs=[
                {
                    "id": "case-1",
                    "system_prompt": (
                        "You are a legal QA generator. Return only valid JSON matching the provided schema."
                    ),
                    "user_prompt": (
                        "Query: Can a consumer who bought running shoes online cancel after delivery, and are there any "
                        "exceptions or practical limits?\n\n"
                        "Articles: [\n"
                        "  {\"id\": \"A1\", \"text\": \"Distance sales generally give consumers a withdrawal period after receiving the goods.\"},\n"
                        "  {\"id\": \"A2\", \"text\": \"The withdrawal right usually starts the day after the consumer or a designated third party receives the product.\"},\n"
                        "  {\"id\": \"A3\", \"text\": \"Some legal systems require the trader to inform the consumer about the withdrawal process, deadline, and model form.\"},\n"
                        "  {\"id\": \"A4\", \"text\": \"Personalized or clearly custom-made goods are often excluded from the withdrawal right.\"},\n"
                        "  {\"id\": \"A5\", \"text\": \"Sealed hygiene products may lose their withdrawal eligibility once unsealed after delivery.\"},\n"
                        "  {\"id\": \"A6\", \"text\": \"Consumers may need to send an unambiguous statement before the deadline to exercise the right.\"},\n"
                        "  {\"id\": \"A7\", \"text\": \"The trader may withhold reimbursement until the goods are returned or proof of return is provided.\"},\n"
                        "  {\"id\": \"A8\", \"text\": \"The consumer can be liable for diminished value caused by handling beyond what is necessary to inspect the shoes.\"},\n"
                        "  {\"id\": \"A9\", \"text\": \"Standard return shipping costs may sometimes be borne by the consumer if properly disclosed in advance.\"},\n"
                        "  {\"id\": \"A10\", \"text\": \"Promotional discounts do not normally remove the underlying withdrawal right for standard goods.\"}\n"
                        "]\n\n"
                        "Create two short answer chunks. One should explain the basic withdrawal right, and the other "
                        "should mention an exception or practical limitation. Use at least one supported source label "
                        "per chunk."
                    ),
                },
                {
                    "id": "case-2",
                    "system_prompt": (
                        "You are a legal QA generator. Return only valid JSON matching the provided schema."
                    ),
                    "user_prompt": (
                        "Query: What happens if a tenant wants to leave a one-year apartment lease after only three "
                        "months because of a new job in another city?\n\n"
                        "Articles: [\n"
                        "  {\"id\": \"A1\", \"text\": \"Fixed-term residential leases often cannot be ended early without consequences unless the contract or statute allows it.\"},\n"
                        "  {\"id\": \"A2\", \"text\": \"Some jurisdictions allow early termination of a residential lease with prior notice and a penalty payment.\"},\n"
                        "  {\"id\": \"A3\", \"text\": \"Notice periods for tenants are commonly expressed in full calendar months.\"},\n"
                        "  {\"id\": \"A4\", \"text\": \"A relocation for work does not automatically cancel lease obligations unless a specific legal ground applies.\"},\n"
                        "  {\"id\": \"A5\", \"text\": \"The landlord may be entitled to compensation if the tenant leaves before the agreed end date.\"},\n"
                        "  {\"id\": \"A6\", \"text\": \"Certain contracts contain a break clause that defines when and how the tenant may terminate early.\"},\n"
                        "  {\"id\": \"A7\", \"text\": \"Security deposits are usually handled separately from any early termination fee.\"},\n"
                        "  {\"id\": \"A8\", \"text\": \"The tenant should give notice in a durable form and keep proof of delivery.\"},\n"
                        "  {\"id\": \"A9\", \"text\": \"If the landlord quickly re-lets the property, the actual financial impact may differ from the initial claim.\"},\n"
                        "  {\"id\": \"A10\", \"text\": \"Mandatory local tenancy rules can override clauses in the written lease.\"}\n"
                        "]\n\n"
                        "Create two short answer chunks. One should explain the general rule, and the other should "
                        "mention what costs or notice requirements may apply. Use at least one supported source label "
                        "per chunk."
                    ),
                },
            ],
            model_config=model_config,
            max_concurrency=2,
        )
    )

    assert not errors
    assert len(results) == 2
    results_by_id = {result["id"]: result for result in results}
    assert set(results_by_id) == {"case-1", "case-2"}

    for expected_id in ("case-1", "case-2"):
        result = results_by_id[expected_id]
        assert "error" not in result, result.get("error")
        assert isinstance(result["raw"], dict)
        assert result["raw"]
        assert isinstance(result["cleaned"], list)
        assert result["cleaned"]
        assert len(result["cleaned"]) >= 2
        print('======================')
        print(result["id"], result["cleaned"])
        for chunk in result["cleaned"]:
            assert isinstance(chunk["text"], str)
            assert chunk["text"]
            assert isinstance(chunk["supported_sources"], list)
            assert chunk["supported_sources"]
