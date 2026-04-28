"""Phase 5: Call tracing for Axon.

Takes FileParseData from the parser phase and resolves call expressions to
target symbol nodes, creating CALLS relationships with confidence scores.

Resolution priority:
1. Same-file exact match (confidence 1.0)
2. Import-resolved match (confidence 1.0)
3. Global fuzzy match (confidence 0.5)
4. Receiver method resolution (confidence 0.8, or 0.75 for type-inferred)
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ThreadPoolExecutor

from axon.core.graph.graph import KnowledgeGraph
from axon.core.graph.model import (
    GraphRelationship,
    NodeLabel,
    RelType,
    generate_id,
)
from axon.core.ingestion.parser_phase import FileParseData
from axon.core.ingestion.resolved import ResolvedEdge
from axon.core.ingestion.symbol_lookup import (
    FileSymbolIndex,
    build_file_symbol_index,
    build_name_index,
    find_containing_symbol,
)
from axon.core.parsers.base import CallInfo

logger = logging.getLogger(__name__)

_CALLABLE_LABELS: tuple[NodeLabel, ...] = (
    NodeLabel.FUNCTION,
    NodeLabel.METHOD,
    NodeLabel.CLASS,
)

_KIND_TO_LABEL: dict[str, NodeLabel] = {
    "function": NodeLabel.FUNCTION,
    "method": NodeLabel.METHOD,
    "class": NodeLabel.CLASS,
}

# Names that should never produce CALLS edges.  These are language builtins,
# stdlib utilities, framework hooks, and common JS/TS globals whose definitions
# do not exist in the user's codebase.  Filtering them before resolution
# prevents low-confidence global-fuzzy matches against short, common names.
_CALL_BLOCKLIST: frozenset[str] = frozenset(
    {
        # Python builtins
        "print",
        "len",
        "range",
        "map",
        "filter",
        "sorted",
        "list",
        "dict",
        "set",
        "str",
        "int",
        "float",
        "bool",
        "type",
        "super",
        "isinstance",
        "issubclass",
        "hasattr",
        "getattr",
        "setattr",
        "open",
        "iter",
        "next",
        "zip",
        "enumerate",
        "any",
        "all",
        "min",
        "max",
        "sum",
        "abs",
        "round",
        "repr",
        "id",
        "hash",
        "dir",
        "vars",
        "input",
        "format",
        "tuple",
        "frozenset",
        "bytes",
        "bytearray",
        "memoryview",
        "object",
        "property",
        "classmethod",
        "staticmethod",
        "delattr",
        "callable",
        "compile",
        "eval",
        "exec",
        "globals",
        "locals",
        "breakpoint",
        "exit",
        "quit",
        # Python stdlib — common method names that collide with user-defined symbols
        "append",
        "extend",
        "update",
        "pop",
        "get",
        "items",
        "keys",
        "values",
        "split",
        "join",
        "strip",
        "replace",
        "startswith",
        "endswith",
        "lower",
        "upper",
        "encode",
        "decode",
        "read",
        "write",
        "close",
        # JS/TS built-in globals
        "console",
        "setTimeout",
        "setInterval",
        "clearTimeout",
        "clearInterval",
        "JSON",
        "Array",
        "Object",
        "Promise",
        "Math",
        "Date",
        "Error",
        "Symbol",
        "parseInt",
        "parseFloat",
        "isNaN",
        "isFinite",
        "encodeURIComponent",
        "decodeURIComponent",
        "fetch",
        "require",
        "exports",
        "module",
        "document",
        "window",
        "process",
        "Buffer",
        "URL",
        # JS/TS dotted method names extracted as bare call names
        "log",
        "error",
        "warn",
        "info",
        "debug",
        "parse",
        "stringify",
        "assign",
        "freeze",
        "isArray",
        "from",
        "of",
        "resolve",
        "reject",
        "race",
        "floor",
        "ceil",
        "random",
        # React hooks
        "useState",
        "useEffect",
        "useRef",
        "useCallback",
        "useMemo",
        "useContext",
        "useReducer",
        "useLayoutEffect",
        "useImperativeHandle",
        "useDebugValue",
        "useId",
        "useTransition",
        "useDeferredValue",
    }
)


def resolve_call(
    call: CallInfo,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
    caller_class_name: str | None = None,
    import_cache: dict[str, set[str]] | None = None,
) -> tuple[str | None, float]:
    """Resolve a call expression to a target node ID and confidence score.

    Resolution strategy (tried in order):

    1. **Same-file exact match** (confidence 1.0) -- the called symbol is
       defined in the same file as the caller.
    2. **Import-resolved match** (confidence 1.0) -- the called name was
       imported into this file; find the symbol in the imported file.
    3. **Global fuzzy match** (confidence 0.5) -- any symbol with this name
       anywhere in the codebase.  If multiple matches exist, the one sharing
       the longest directory prefix with the caller is preferred.

    For method calls (``call.receiver`` is non-empty):
    - If the receiver is ``"self"`` or ``"this"``, look for a method with
      that name in the same class (same file, matching class_name).
    - Otherwise, try to resolve the method name globally.

    Args:
        call: The parsed call information.
        file_path: Path to the file containing the call.
        call_index: Mapping from symbol names to node IDs built by
            :func:`build_call_index`.
        graph: The knowledge graph.
        caller_class_name: Optional class name of the calling symbol,
            used to scope ``self``/``this`` method resolution.

    Returns:
        A tuple of ``(node_id, confidence)`` or ``(None, 0.0)`` if the
        call cannot be resolved.
    """
    name = call.name
    receiver = call.receiver

    if receiver in ("self", "this"):
        result = _resolve_self_method(name, file_path, call_index, graph, caller_class_name)
        if result is not None:
            return result, 1.0

    candidate_ids = call_index.get(name, [])
    if not candidate_ids:
        return None, 0.0

    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is not None and node.file_path == file_path:
            return nid, 1.0

    effective_cache = (
        import_cache if import_cache is not None else _build_import_cache(file_path, graph)
    )
    imported_target = _resolve_via_imports(name, candidate_ids, graph, effective_cache)
    if imported_target is not None:
        return imported_target, 1.0

    if len(candidate_ids) > 5:
        return None, 0.0
    return _pick_closest(candidate_ids, graph, caller_file_path=file_path), 0.5


def _resolve_self_method(
    method_name: str,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
    caller_class_name: str | None = None,
) -> str | None:
    """Find a method with *method_name* in the same file and class.

    When the receiver is ``self`` or ``this`` the target must be a Method
    node defined in the same file.  If *caller_class_name* is provided,
    candidates are further filtered to the same class.
    """
    fallback: str | None = None
    for nid in call_index.get(method_name, []):
        node = graph.get_node(nid)
        if node is not None and node.label == NodeLabel.METHOD and node.file_path == file_path:
            if caller_class_name and node.class_name == caller_class_name:
                return nid
            if fallback is None:
                fallback = nid
    return fallback


def _build_import_cache(
    file_path: str,
    graph: KnowledgeGraph,
) -> dict[str, set[str]]:
    """Build {symbol_name → set of imported file_paths} for a file.

    The special key ``"*"`` contains file paths from wildcard/full-module imports.
    """
    source_file_id = generate_id(NodeLabel.FILE, file_path)
    import_rels = graph.get_outgoing(source_file_id, RelType.IMPORTS)

    cache: dict[str, set[str]] = {}
    for rel in import_rels:
        target_node = graph.get_node(rel.target)
        if target_node is None:
            continue
        symbols_str = rel.properties.get("symbols", "")
        imported_names = {s.strip() for s in symbols_str.split(",") if s.strip()}
        if not imported_names:
            cache.setdefault("*", set()).add(target_node.file_path)
        else:
            for sym_name in imported_names:
                cache.setdefault(sym_name, set()).add(target_node.file_path)
    return cache


def _resolve_via_imports(
    name: str,
    candidate_ids: list[str],
    graph: KnowledgeGraph,
    import_cache: dict[str, set[str]],
) -> str | None:
    """Check if *name* was imported and resolve to the target using cached data.

    Uses the pre-built *import_cache* (from :func:`_build_import_cache`)
    to avoid re-scanning IMPORTS relationships for every call in the same file.
    """
    if not import_cache:
        return None

    imported_file_paths = import_cache.get(name, set()) | import_cache.get("*", set())
    if not imported_file_paths:
        return None

    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is not None and node.file_path in imported_file_paths:
            return nid

    return None


def _common_prefix_len(a: str, b: str) -> int:
    """Return the length of the common directory prefix between two paths."""
    parts_a = a.split("/")
    parts_b = b.split("/")
    common = 0
    for pa, pb in zip(parts_a, parts_b):
        if pa == pb:
            common += 1
        else:
            break
    return common


def _pick_closest(
    candidate_ids: list[str],
    graph: KnowledgeGraph,
    caller_file_path: str = "",
) -> str | None:
    """Pick the candidate sharing the longest directory prefix with the caller.

    Falls back to shortest file path when no caller path is provided.
    Returns ``None`` if no candidates can be resolved to actual nodes.
    """
    best_id: str | None = None
    best_score: tuple[int, int] = (-1, 0)

    for nid in candidate_ids:
        node = graph.get_node(nid)
        if node is None:
            continue
        if caller_file_path:
            prefix = _common_prefix_len(caller_file_path, node.file_path)
            score = (prefix, -len(node.file_path))
        else:
            score = (0, -len(node.file_path))
        if score > best_score:
            best_score = score
            best_id = nid

    return best_id


def _make_edge(
    source_id: str,
    target_id: str,
    confidence: float,
    seen: set[str],
) -> ResolvedEdge | None:
    """Create a deduplicated ResolvedEdge, returning None if already seen."""
    rel_id = f"calls:{source_id}->{target_id}"
    if rel_id in seen:
        return None
    seen.add(rel_id)
    return ResolvedEdge(
        rel_id=rel_id,
        rel_type=RelType.CALLS,
        source=source_id,
        target=target_id,
        properties={"confidence": confidence},
    )


def _resolve_receiver_method(
    receiver: str,
    method_name: str,
    source_id: str,
    file_path: str,
    call_index: dict[str, list[str]],
    graph: KnowledgeGraph,
    type_table: _ScopedTypeTable | None = None,
    caller_line: int = 0,
) -> ResolvedEdge | None:
    """Resolve ``Receiver.method()`` to the METHOD node and return a ResolvedEdge.

    Looks for a METHOD node whose ``name`` matches *method_name* and whose
    ``class_name`` matches *receiver* (or the type-inferred class name).
    Searches same-file first, then globally.
    """
    # Resolve receiver to class name via type table
    resolved_receiver = receiver
    type_inferred = False
    if type_table:
        inferred = _lookup_scoped_type(type_table, file_path, receiver, caller_line)
        if inferred is not None:
            resolved_receiver = inferred
            type_inferred = True

    same_file_match: str | None = None
    global_match: str | None = None

    for nid in call_index.get(method_name, []):
        node = graph.get_node(nid)
        if (
            node is not None
            and node.label == NodeLabel.METHOD
            and node.class_name == resolved_receiver
        ):
            if node.file_path == file_path:
                same_file_match = nid
                break
            elif global_match is None:
                global_match = nid
        if same_file_match is not None:
            break

    target = same_file_match or global_match
    if target is not None:
        confidence = 0.75 if type_inferred else 0.8
        return ResolvedEdge(
            rel_id=f"calls:{source_id}->{target}",
            rel_type=RelType.CALLS,
            source=source_id,
            target=target,
            properties={"confidence": confidence},
        )
    return None


# Scoped type entry: (class_name, scope_start, scope_end).
# scope_start == scope_end == 0 means file-scoped (applies everywhere).
_ScopedTypeEntry = tuple[str, int, int]
_ScopedTypeTable = dict[str, dict[str, list[_ScopedTypeEntry]]]


def _build_type_table(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
) -> _ScopedTypeTable:
    """Build per-file ``{variable_name: [(class_name, scope_start, scope_end)]}``
    from annotations and constructors.

    Variable annotations and constructor assignments are file-scoped.
    Parameter annotations are scoped to their containing symbol's line range
    so that identically-named params in different functions don't collide.
    """
    class_names = {node.name for node in graph.get_nodes_by_label(NodeLabel.CLASS)}
    table: _ScopedTypeTable = {}
    for fpd in parse_data:
        file_entries: dict[str, list[_ScopedTypeEntry]] = {}
        symbols = fpd.parse_result.symbols

        for tref in fpd.parse_result.type_refs:
            if tref.kind == "variable" and tref.variable_name:
                file_entries.setdefault(tref.variable_name, []).append(
                    (tref.name, 0, 0)
                )
            elif tref.kind == "param" and tref.param_name:
                scope_start, scope_end, best_span = 0, 0, float("inf")
                for sym in symbols:
                    span = sym.end_line - sym.start_line
                    if sym.start_line <= tref.line <= sym.end_line and span < best_span:
                        scope_start, scope_end, best_span = sym.start_line, sym.end_line, span
                file_entries.setdefault(tref.param_name, []).append(
                    (tref.name, scope_start, scope_end)
                )

        for call in fpd.parse_result.calls:
            if call.assignment_target and call.name in class_names and not call.receiver:
                file_entries.setdefault(call.assignment_target, []).append(
                    (call.name, 0, 0)
                )
        if file_entries:
            table[fpd.file_path] = file_entries
    return table


def _lookup_scoped_type(
    type_table: _ScopedTypeTable,
    file_path: str,
    var_name: str,
    caller_line: int,
) -> str | None:
    """Look up the inferred type for *var_name* at *caller_line*.

    Returns the class name from the narrowest matching scope, or ``None``.
    """
    file_entries = type_table.get(file_path)
    if not file_entries:
        return None
    entries = file_entries.get(var_name)
    if not entries:
        return None

    best: str | None = None
    best_span = float("inf")
    for class_name, scope_start, scope_end in entries:
        if scope_start == 0 and scope_end == 0:
            # File-scoped: always matches; use as fallback.
            if best is None:
                best = class_name
        elif scope_start <= caller_line <= scope_end:
            span = scope_end - scope_start
            if span < best_span:
                best = class_name
                best_span = span
    return best


def resolve_file_calls(
    fpd: FileParseData,
    call_index: dict[str, list[str]],
    file_sym_index: FileSymbolIndex,
    graph: KnowledgeGraph,
    type_table: _ScopedTypeTable | None = None,
) -> list[ResolvedEdge]:
    """Resolve all call expressions in a single file to ResolvedEdge objects.

    This is a pure-ish function (reads from graph but does not mutate it)
    that can be called in parallel across files.
    """
    edges: list[ResolvedEdge] = []
    seen: set[str] = set()
    import_cache = _build_import_cache(fpd.file_path, graph)

    for call in fpd.parse_result.calls:
        if call.name in _CALL_BLOCKLIST and call.receiver not in ("self", "this"):
            continue

        source_id = find_containing_symbol(call.line, fpd.file_path, file_sym_index)
        if source_id is None:
            # Module-level call: attribute to the File node.
            source_id = generate_id(NodeLabel.FILE, fpd.file_path)
            if graph.get_node(source_id) is None:
                logger.debug(
                    "No containing symbol or file node for call %s at line %d in %s",
                    call.name,
                    call.line,
                    fpd.file_path,
                )
                continue

        caller_class_name: str | None = None
        if call.receiver in ("self", "this"):
            source_node = graph.get_node(source_id)
            if source_node is not None:
                caller_class_name = source_node.class_name

        target_id, confidence = resolve_call(
            call,
            fpd.file_path,
            call_index,
            graph,
            caller_class_name=caller_class_name,
            import_cache=import_cache,
        )
        if target_id is not None:
            edge = _make_edge(source_id, target_id, confidence, seen)
            if edge is not None:
                edges.append(edge)

        for arg_name in call.arguments:
            if arg_name in _CALL_BLOCKLIST:
                continue
            arg_call = CallInfo(name=arg_name, line=call.line)
            arg_id, arg_conf = resolve_call(
                arg_call,
                fpd.file_path,
                call_index,
                graph,
                import_cache=import_cache,
            )
            if arg_id is not None:
                edge = _make_edge(source_id, arg_id, arg_conf * 0.8, seen)
                if edge is not None:
                    edges.append(edge)

        receiver = call.receiver
        if receiver and receiver not in ("self", "this"):
            receiver_call = CallInfo(name=receiver, line=call.line)
            recv_id, recv_conf = resolve_call(
                receiver_call,
                fpd.file_path,
                call_index,
                graph,
                import_cache=import_cache,
            )
            if recv_id is not None:
                edge = _make_edge(source_id, recv_id, recv_conf, seen)
                if edge is not None:
                    edges.append(edge)

            recv_method_edge = _resolve_receiver_method(
                receiver,
                call.name,
                source_id,
                fpd.file_path,
                call_index,
                graph,
                type_table=type_table,
                caller_line=call.line,
            )
            if recv_method_edge is not None and recv_method_edge.rel_id not in seen:
                seen.add(recv_method_edge.rel_id)
                edges.append(recv_method_edge)

    for symbol in fpd.parse_result.symbols:
        if not symbol.decorators:
            continue

        symbol_name = (
            f"{symbol.class_name}.{symbol.name}"
            if symbol.kind == "method" and symbol.class_name
            else symbol.name
        )
        label = _KIND_TO_LABEL.get(symbol.kind)
        if label is None:
            continue
        source_id = generate_id(label, fpd.file_path, symbol_name)

        for dec_name in symbol.decorators:
            base_name = dec_name.rsplit(".", 1)[-1] if "." in dec_name else dec_name
            call_obj = CallInfo(name=base_name, line=symbol.start_line)
            target_id, confidence = resolve_call(
                call_obj,
                fpd.file_path,
                call_index,
                graph,
                import_cache=import_cache,
            )
            if target_id is None and "." in dec_name:
                call_obj = CallInfo(name=dec_name, line=symbol.start_line)
                target_id, confidence = resolve_call(
                    call_obj,
                    fpd.file_path,
                    call_index,
                    graph,
                    import_cache=import_cache,
                )
            if target_id is not None:
                edge = _make_edge(source_id, target_id, confidence, seen)
                if edge is not None:
                    edges.append(edge)

    # First-class function references (e.g. ``handler = my_func``).
    for func_ref in fpd.parse_result.func_refs:
        if func_ref.name in _CALL_BLOCKLIST:
            continue
        source_id = find_containing_symbol(func_ref.line, fpd.file_path, file_sym_index)
        if source_id is None:
            source_id = generate_id(NodeLabel.FILE, fpd.file_path)
            if graph.get_node(source_id) is None:
                continue
        ref_call = CallInfo(name=func_ref.name, line=func_ref.line)
        target_id, confidence = resolve_call(
            ref_call,
            fpd.file_path,
            call_index,
            graph,
            import_cache=import_cache,
        )
        if target_id is not None:
            edge = _make_edge(source_id, target_id, confidence * 0.7, seen)
            if edge is not None:
                edges.append(edge)

    return edges


def process_calls(
    parse_data: list[FileParseData],
    graph: KnowledgeGraph,
    name_index: dict[str, list[str]] | None = None,
    *,
    parallel: bool = False,
    collect: bool = False,
) -> list[ResolvedEdge] | None:
    """Resolve call expressions and create CALLS relationships in the graph.

    For each call expression in the parse data:

    1. Determine which symbol in the file *contains* the call (by line
       number range).
    2. Resolve the call to a target symbol node.
    3. Create a CALLS relationship from the containing symbol to the
       target, with a ``confidence`` property.

    Args:
        parse_data: File parse results from the parser phase.
        graph: The knowledge graph to populate with CALLS relationships.
        name_index: Optional pre-built name index; built automatically if None.
        parallel: When True, resolve files using a thread pool.
        collect: When True, return the list of ResolvedEdge objects instead
            of writing them to the graph.

    Returns:
        A list of ResolvedEdge when *collect* is True, otherwise None.
    """
    call_index = name_index if name_index is not None else build_name_index(graph, _CALLABLE_LABELS)
    file_sym_index = build_file_symbol_index(graph, _CALLABLE_LABELS)
    type_table = _build_type_table(parse_data, graph)

    if parallel and len(parse_data) > 1:
        workers = min(os.cpu_count() or 4, 8, len(parse_data))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(resolve_file_calls, fpd, call_index, file_sym_index, graph, type_table)
                for fpd in parse_data
            ]
            per_file_edges = [f.result() for f in futures]
    else:
        per_file_edges = [
            resolve_file_calls(fpd, call_index, file_sym_index, graph, type_table)
            for fpd in parse_data
        ]

    seen: set[str] = set()
    deduped: list[ResolvedEdge] = []
    for file_edges in per_file_edges:
        for edge in file_edges:
            if edge.rel_id not in seen:
                seen.add(edge.rel_id)
                deduped.append(edge)

    if collect:
        return deduped

    for edge in deduped:
        graph.add_relationship(
            GraphRelationship(
                id=edge.rel_id,
                type=edge.rel_type,
                source=edge.source,
                target=edge.target,
                properties=edge.properties,
            )
        )
    return None
