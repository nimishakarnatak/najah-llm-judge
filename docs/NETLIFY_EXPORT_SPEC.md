# Episode export for Najah Review Studio

Add an authenticated admin-only download in the existing source project. The visible admin page currently exports rating rows but lacks a conversation/participant export. Do not scrape evaluator identity as conversation-participant identity.

Return `application/json` with a filename such as `najah-episodes.json`, using the same access checks as the existing authenticated admin ratings exports. Emit exactly one record per module episode and retain IDs used in the ratings CSV.

```json
{
  "schema_version": "najah-stage2-episodes-v1",
  "episodes": [
    {
      "episode_id": "N2E000001",
      "participant_id": "stable-deidentified-participant-id",
      "module": "job_search_strategy",
      "language": "fr",
      "sampling_weight": 3.61,
      "treatment": "Gender-sensitive",
      "turns": [
        {"turn_id": "002", "role": "assistant", "text": "Original message", "segment": "episode"}
      ],
      "delivery_evidence": []
    }
  ]
}
```

Use the original participant linkage in the dataset, transformed into a stable study pseudonym on the server. Keep that pseudonym consistent across all conversations and episodes. Preserve original turn numbering, order, roles, languages and the prior/focal/subsequent boundaries displayed in the queue. The existing `und` language value should be verified, not inferred from treatment or gender.

If real-world artifact delivery is verifiable, add de-identified evidence objects with a stable evidence ID, evidence type, verification status and permissible excerpt. Do not include expiring credentials, personal download links or raw identifiers. For future return analysis, add ISO timestamps, dataset extraction time, follow-up end, eligibility and censoring fields. They are optional for judge validation but required for valid follow-up analyses.

Acceptance checks: 300 distinct episode IDs matching the sampled rating records; no evaluator account IDs used as participant IDs; complete ordered focal text; no personal emails or phone numbers; weights agree with the sampling manifest; duplicates/repeated participants documented. Never substitute English translations for original-language evidence.

Implement the endpoint inside the existing project only after obtaining its source and database schema. No database table names, API routes, storage credentials or participant linkage are assumed by this standalone package.
