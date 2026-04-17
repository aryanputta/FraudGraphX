"""
LLM Auto-Investigator using Claude claude-sonnet-4-6 with prompt caching.
Generates human-readable investigation notes for fraud alerts.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

import anthropic

from src.common.config import get_settings
from src.common.logging import get_logger
from src.common.schemas import FraudAlert, FraudCase

logger = get_logger(__name__)
settings = get_settings()


SYSTEM_PROMPT = """You are a senior fraud analyst at a financial institution.
You have deep expertise in fraud ring detection, money laundering, synthetic identity fraud,
and mule account patterns. Your role is to analyze fraud alerts and produce concise,
actionable investigation notes for the case management team.

When analyzing an alert, you will:
1. Summarize the risk indicators clearly
2. Identify the most likely fraud pattern
3. Recommend specific investigation steps
4. Estimate the potential fraud exposure
5. Suggest whether to BLOCK, REVIEW, or PASS the transaction

Be precise, factual, and professional. Limit responses to 300 words."""


class LLMInvestigator:
    """Generate AI-powered investigation notes for fraud alerts."""

    def __init__(self) -> None:
        if not settings.anthropic_api_key:
            logger.warning("ANTHROPIC_API_KEY not set – LLM investigator disabled")
            self._client = None
        else:
            self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def _is_enabled(self) -> bool:
        return self._client is not None and settings.llm_investigator_enabled

    def generate_notes(
        self,
        alert: FraudAlert,
        shap_top_features: List[Dict[str, Any]],
        graph_path: List[str],
        fraud_ring_members: Optional[List[str]] = None,
    ) -> str:
        if not self._is_enabled():
            return self._rule_based_notes(alert)

        alert_summary = {
            "transaction_id": alert.transaction_id,
            "customer_id": alert.customer_id,
            "ensemble_fraud_score": alert.ensemble_score,
            "risk_level": alert.risk_level.value,
            "detected_fraud_type": alert.fraud_type.value,
            "recommended_action": alert.action,
            "model_scores": [{"model": s.model_name, "score": s.score} for s in alert.model_scores],
            "top_shap_features": shap_top_features,
            "suspicious_graph_path": graph_path,
            "potential_fraud_ring_members": fraud_ring_members or [],
        }

        user_message = f"""Please analyze this fraud alert and produce investigation notes:

```json
{json.dumps(alert_summary, indent=2)}
```

Focus on:
- Why the ensemble model flagged this transaction
- What the SHAP features reveal about the behaviour
- Whether the graph connections suggest an organized fraud ring
- Recommended next steps for the investigation team"""

        try:
            response = self._client.messages.create(
                model=settings.llm_model,
                max_tokens=512,
                system=[
                    {
                        "type": "text",
                        "text": SYSTEM_PROMPT,
                        "cache_control": {"type": "ephemeral"},  # Prompt caching
                    }
                ],
                messages=[{"role": "user", "content": user_message}],
            )
            notes = response.content[0].text
            logger.info(
                "LLM investigation notes generated",
                alert_id=alert.alert_id,
                input_tokens=response.usage.input_tokens,
                cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0),
            )
            return notes
        except Exception as exc:
            logger.error("LLM investigator failed", error=str(exc))
            return self._rule_based_notes(alert)

    @staticmethod
    def _rule_based_notes(alert: FraudAlert) -> str:
        """Deterministic fallback when LLM is unavailable."""
        lines = [
            f"FRAUD ALERT – Risk Level: {alert.risk_level.value.upper()}",
            f"Ensemble Score: {alert.ensemble_score:.3f}",
            f"Recommended Action: {alert.action.upper()}",
            "",
            "Model Scores:",
        ]
        for s in alert.model_scores:
            lines.append(f"  • {s.model_name}: {s.score:.3f}")
        if alert.graph_path:
            lines += ["", "Suspicious Graph Path:", "  " + " → ".join(alert.graph_path)]
        lines += [
            "",
            "Next Steps:",
            "  1. Review transaction history for the past 30 days",
            "  2. Verify customer identity documents",
            "  3. Check shared device/IP connections",
            "  4. Escalate if linked to known fraud ring",
        ]
        return "\n".join(lines)
