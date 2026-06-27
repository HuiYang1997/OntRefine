from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ConceptInfo:
    class_id: str
    iri: str
    label: Optional[str] = None
    comment: Optional[str] = None
    scope_note: Optional[str] = None
    superclasses: list[str] = field(default_factory=list)
    restrictions: list[dict] = field(default_factory=list)

    def display_label(self) -> str:
        return self.label or self.class_id.replace('_', ' ')

    def text_for_embedding(self) -> str:
        parts = [self.display_label()]
        if self.comment:
            parts.append(self.comment)
        return '. '.join(parts)


@dataclass
class SubsumptionGroup:
    child_id: str
    parent_ids: list[str]


def _local_name(uri) -> str:
    text = str(uri)
    if '#' in text:
        text = text.rsplit('#', 1)[1]
    else:
        text = text.rstrip('/').rsplit('/', 1)[-1]
    return text.replace('-', '_').replace('.', '_') or 'Thing'


def _first_literal(graph, subject, predicates) -> Optional[str]:
    for predicate in predicates:
        for obj in graph.objects(subject, predicate):
            return str(obj)
    return None


def _make_uri_id_map(uris: list) -> dict:
    uri_to_id = {}
    used = {}
    for uri in sorted(uris, key=str):
        base = _local_name(uri)
        count = used.get(base, 0)
        used[base] = count + 1
        uri_to_id[uri] = base if count == 0 else f'{base}_{count + 1}'
    return uri_to_id


def load_owl(path: str | Path) -> tuple[dict[str, ConceptInfo], list[SubsumptionGroup]]:
    try:
        from rdflib import BNode, Graph, RDF, RDFS, URIRef
        from rdflib.namespace import OWL, SKOS
    except ImportError as exc:
        raise ImportError('OWL input requires rdflib. Install dependencies with run.sh or pip install -r requirements.txt') from exc

    path = Path(path)
    graph = Graph()
    graph.parse(str(path))

    class_uris = set(graph.subjects(RDF.type, OWL.Class))
    for child, _, parent in graph.triples((None, RDFS.subClassOf, None)):
        if isinstance(child, URIRef):
            class_uris.add(child)
        if isinstance(parent, URIRef) and parent != OWL.Thing:
            class_uris.add(parent)

    uri_to_id = _make_uri_id_map(list(class_uris))
    concepts: dict[str, ConceptInfo] = {}
    for uri, class_id in uri_to_id.items():
        concepts[class_id] = ConceptInfo(
            class_id=class_id,
            iri=str(uri),
            label=_first_literal(graph, uri, [RDFS.label]),
            comment=_first_literal(graph, uri, [RDFS.comment]),
            scope_note=_first_literal(graph, uri, [SKOS.scopeNote]),
        )

    for child_uri, _, parent in graph.triples((None, RDFS.subClassOf, None)):
        if not isinstance(child_uri, URIRef) or child_uri not in uri_to_id:
            continue
        child = concepts[uri_to_id[child_uri]]
        if isinstance(parent, URIRef):
            parent_id = 'owl:Thing' if parent == OWL.Thing else uri_to_id.get(parent)
            if parent_id and parent_id not in child.superclasses:
                child.superclasses.append(parent_id)
            continue
        if isinstance(parent, BNode) and (parent, RDF.type, OWL.Restriction) in graph:
            prop = next(graph.objects(parent, OWL.onProperty), None)
            filler = next(graph.objects(parent, OWL.someValuesFrom), None)
            rtype = 'someValuesFrom'
            if filler is None:
                filler = next(graph.objects(parent, OWL.allValuesFrom), None)
                rtype = 'allValuesFrom'
            child.restrictions.append({
                'type': rtype,
                'property': _local_name(prop) if prop is not None else '?',
                'filler': uri_to_id.get(filler, _local_name(filler)) if filler is not None else '?',
            })

    groups = [SubsumptionGroup(cid, list(c.superclasses)) for cid, c in concepts.items() if c.superclasses]
    return concepts, groups
