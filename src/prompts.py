from __future__ import annotations

from ontology_io import ConceptInfo
from ranking import GroupRank

SYSTEM_PROMPT = "You are an expert ontology engineer. Validate OWL direct superclass axioms of the form A subClassOf B. Return valid JSON only."


def build_prompt(group: GroupRank, concepts: dict[str, ConceptInfo], candidate_k: int = 10) -> list[dict]:
    child = concepts.get(group.child_id)
    label = child.display_label() if child else group.child_id
    definition = child.comment if child and child.comment else '(no definition available)'
    lines = [
        'CONCEPT UNDER REVIEW',
        f'ID: {group.child_id}',
        f'Label: {label}',
        f'Definition: {definition}',
        '',
        'CURRENT DIRECT SUPERCLASS AXIOMS',
    ]
    for parent_id in group.parent_ids:
        parent = concepts.get(parent_id)
        plabel = parent.display_label() if parent else parent_id
        pdef = parent.comment if parent and parent.comment else '(no definition available)'
        lines += [
            f'- {group.child_id} subClassOf {parent_id}',
            f'  Parent label: {plabel}',
            f'  Parent definition: {pdef}',
        ]

    lines += ['', f'TOP {candidate_k} ALTERNATIVE PARENT CANDIDATES FROM OnT']
    for idx, (candidate_id, _score) in enumerate(group.alternatives[:candidate_k], 1):
        concept = concepts.get(candidate_id)
        clabel = concept.display_label() if concept else candidate_id
        cdef = concept.comment if concept and concept.comment else '(no definition available)'
        lines += [f'{idx}. {candidate_id} - {clabel}', f'   {cdef}']

    lines += [
        '',
        'TASK',
        'For each current parent, decide whether it is a correct direct superclass.',
        'Suggest replacements only from known ontology class IDs when possible.',
        'Return only this JSON object:',
        '{',
        '  "concept": "<class_id>",',
        '  "verdict_per_parent": [',
        '    {"parent": "<parent_id>", "verdict": "correct|incorrect|partial", "confidence": "high|medium|low", "reason": "<brief reason>"}',
        '  ],',
        '  "suggested_replacements": [',
        '    {"remove": "<parent_id_or_null>", "add": "<new_parent_id>", "reason": "<brief reason>"}',
        '  ],',
        '  "missing_parents": [',
        '    {"parent": "<candidate_class_id>", "reason": "<brief reason>"}',
        '  ],',
        '  "overall_verdict": "correct|needs_review",',
        '  "summary": "<one sentence>"',
        '}',
    ]
    return [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': '\n'.join(lines)},
    ]
