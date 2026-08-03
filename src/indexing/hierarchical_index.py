from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable, Iterator

from src.models import DocumentNode, NodeType


class HierarchicalIndexError(ValueError):
    """Raised when the document hierarchy is invalid."""


class HierarchicalDocumentIndex:
    """
    In-memory index for hierarchical DocumentNode collections.

    The index supports efficient access by:

    - node ID
    - document ID
    - node type
    - page number
    - parent/child relationships
    - ancestors and descendants

    This class does not perform semantic retrieval. It provides the
    structural index that block-level retrievers will consume later.
    """

    def __init__(
        self,
        nodes: Iterable[DocumentNode] | None = None,
    ) -> None:
        self._nodes_by_id: dict[str, DocumentNode] = {}
        self._node_ids_by_document: dict[str, list[str]] = defaultdict(list)
        self._node_ids_by_type: dict[NodeType, list[str]] = defaultdict(list)
        self._node_ids_by_page: dict[
            tuple[str, int],
            list[str],
        ] = defaultdict(list)

        if nodes is not None:
            self.build(nodes)

    def build(
        self,
        nodes: Iterable[DocumentNode],
    ) -> None:
        """
        Build the index from scratch.

        Args:
            nodes: Parsed DocumentNode objects.

        Raises:
            HierarchicalIndexError: If the hierarchy is invalid.
        """
        node_list = list(nodes)

        self.clear()
        self._validate_unique_node_ids(node_list)

        for node in node_list:
            self._add_to_indexes(node)

        self._validate_parent_references()
        self._validate_child_references()
        self._validate_parent_child_consistency()
        self._validate_no_cycles()

    def add_nodes(
        self,
        nodes: Iterable[DocumentNode],
    ) -> None:
        """
        Add nodes to an existing index.

        The index is rebuilt to ensure the complete hierarchy remains valid.
        """
        combined_nodes = [
            *self._nodes_by_id.values(),
            *list(nodes),
        ]

        self.build(combined_nodes)

    def clear(self) -> None:
        """Remove every indexed node."""
        self._nodes_by_id.clear()
        self._node_ids_by_document.clear()
        self._node_ids_by_type.clear()
        self._node_ids_by_page.clear()

    def __len__(self) -> int:
        return len(self._nodes_by_id)

    def __contains__(self, node_id: object) -> bool:
        return (
            isinstance(node_id, str)
            and node_id in self._nodes_by_id
        )

    def __iter__(self) -> Iterator[DocumentNode]:
        return iter(self._nodes_by_id.values())

    @property
    def node_count(self) -> int:
        return len(self)

    @property
    def document_ids(self) -> list[str]:
        return sorted(self._node_ids_by_document)

    def get_node(
        self,
        node_id: str,
    ) -> DocumentNode:
        """
        Return a node by ID.

        Raises:
            KeyError: If the node does not exist.
        """
        try:
            return self._nodes_by_id[node_id]
        except KeyError as exc:
            raise KeyError(
                f"Node not found in hierarchical index: {node_id}"
            ) from exc

    def get_node_or_none(
        self,
        node_id: str,
    ) -> DocumentNode | None:
        return self._nodes_by_id.get(node_id)

    def get_document_nodes(
        self,
        document_id: str,
    ) -> list[DocumentNode]:
        return self._resolve_node_ids(
            self._node_ids_by_document.get(
                document_id,
                [],
            )
        )

    def get_nodes_by_type(
        self,
        node_type: NodeType,
        document_id: str | None = None,
    ) -> list[DocumentNode]:
        nodes = self._resolve_node_ids(
            self._node_ids_by_type.get(
                node_type,
                [],
            )
        )

        if document_id is None:
            return nodes

        return [
            node
            for node in nodes
            if node.document_id == document_id
        ]

    def get_page_nodes(
        self,
        document_id: str,
        page_number: int,
    ) -> list[DocumentNode]:
        if page_number <= 0:
            raise ValueError(
                "page_number must be greater than zero."
            )

        return self._resolve_node_ids(
            self._node_ids_by_page.get(
                (document_id, page_number),
                [],
            )
        )

    def get_root(
        self,
        document_id: str,
    ) -> DocumentNode:
        roots = [
            node
            for node in self.get_document_nodes(document_id)
            if (
                node.node_type == NodeType.DOCUMENT
                and node.parent_id is None
            )
        ]

        if not roots:
            raise KeyError(
                f"No root document node found for: {document_id}"
            )

        if len(roots) > 1:
            raise HierarchicalIndexError(
                f"Multiple root document nodes found for: {document_id}"
            )

        return roots[0]

    def get_parent(
        self,
        node_id: str,
    ) -> DocumentNode | None:
        node = self.get_node(node_id)

        if node.parent_id is None:
            return None

        return self.get_node(node.parent_id)

    def get_children(
        self,
        node_id: str,
    ) -> list[DocumentNode]:
        node = self.get_node(node_id)
        return self._resolve_node_ids(node.children_ids)

    def get_ancestors(
        self,
        node_id: str,
        include_self: bool = False,
    ) -> list[DocumentNode]:
        """
        Return ancestors from the closest parent to the root.
        """
        current = self.get_node(node_id)
        ancestors: list[DocumentNode] = []

        if include_self:
            ancestors.append(current)

        visited: set[str] = {current.node_id}

        while current.parent_id is not None:
            parent = self.get_node(current.parent_id)

            if parent.node_id in visited:
                raise HierarchicalIndexError(
                    f"Cycle detected while traversing node: {node_id}"
                )

            ancestors.append(parent)
            visited.add(parent.node_id)
            current = parent

        return ancestors

    def get_descendants(
        self,
        node_id: str,
        include_self: bool = False,
        max_depth: int | None = None,
    ) -> list[DocumentNode]:
        """
        Return descendants in breadth-first hierarchical order.

        Args:
            node_id: Starting node.
            include_self: Include the starting node.
            max_depth: Maximum child depth. None means unlimited.
        """
        if max_depth is not None and max_depth < 0:
            raise ValueError(
                "max_depth cannot be negative."
            )

        root = self.get_node(node_id)
        descendants: list[DocumentNode] = []

        if include_self:
            descendants.append(root)

        queue: deque[tuple[DocumentNode, int]] = deque(
            [(root, 0)]
        )
        visited: set[str] = {root.node_id}

        while queue:
            current, depth = queue.popleft()

            if max_depth is not None and depth >= max_depth:
                continue

            for child in self.get_children(current.node_id):
                if child.node_id in visited:
                    raise HierarchicalIndexError(
                        "Cycle detected while traversing descendants "
                        f"from node: {node_id}"
                    )

                visited.add(child.node_id)
                descendants.append(child)
                queue.append((child, depth + 1))

        return descendants

    def get_siblings(
        self,
        node_id: str,
        include_self: bool = False,
    ) -> list[DocumentNode]:
        node = self.get_node(node_id)

        if node.parent_id is None:
            return [node] if include_self else []

        siblings = self.get_children(node.parent_id)

        if include_self:
            return siblings

        return [
            sibling
            for sibling in siblings
            if sibling.node_id != node.node_id
        ]

    def get_content_nodes(
        self,
        document_id: str | None = None,
    ) -> list[DocumentNode]:
        """
        Return nodes that may provide textual or visual evidence.

        Document and page container nodes are excluded.
        """
        nodes = [
            node
            for node in self
            if node.node_type not in {
                NodeType.DOCUMENT,
                NodeType.PAGE,
            }
        ]

        if document_id is None:
            return nodes

        return [
            node
            for node in nodes
            if node.document_id == document_id
        ]

    def to_dict(self) -> dict[str, object]:
        return {
            "node_count": self.node_count,
            "document_ids": self.document_ids,
            "nodes": [
                node.to_dict()
                for node in self
            ],
        }

    def _add_to_indexes(
        self,
        node: DocumentNode,
    ) -> None:
        self._nodes_by_id[node.node_id] = node

        self._node_ids_by_document[
            node.document_id
        ].append(node.node_id)

        self._node_ids_by_type[
            node.node_type
        ].append(node.node_id)

        if node.page_number is not None:
            self._node_ids_by_page[
                (
                    node.document_id,
                    node.page_number,
                )
            ].append(node.node_id)

    def _resolve_node_ids(
        self,
        node_ids: Iterable[str],
    ) -> list[DocumentNode]:
        return [
            self._nodes_by_id[node_id]
            for node_id in node_ids
        ]

    @staticmethod
    def _validate_unique_node_ids(
        nodes: list[DocumentNode],
    ) -> None:
        seen: set[str] = set()

        for node in nodes:
            if node.node_id in seen:
                raise HierarchicalIndexError(
                    f"Duplicate node ID: {node.node_id}"
                )

            seen.add(node.node_id)

    def _validate_parent_references(self) -> None:
        for node in self:
            if node.parent_id is None:
                continue

            if node.parent_id not in self._nodes_by_id:
                raise HierarchicalIndexError(
                    f"Node '{node.node_id}' references missing parent "
                    f"'{node.parent_id}'."
                )

            parent = self._nodes_by_id[node.parent_id]

            if parent.document_id != node.document_id:
                raise HierarchicalIndexError(
                    f"Node '{node.node_id}' and its parent "
                    f"'{parent.node_id}' belong to different documents."
                )

    def _validate_child_references(self) -> None:
        for node in self:
            seen_children: set[str] = set()

            for child_id in node.children_ids:
                if child_id in seen_children:
                    raise HierarchicalIndexError(
                        f"Node '{node.node_id}' contains duplicate child "
                        f"reference '{child_id}'."
                    )

                seen_children.add(child_id)

                if child_id not in self._nodes_by_id:
                    raise HierarchicalIndexError(
                        f"Node '{node.node_id}' references missing child "
                        f"'{child_id}'."
                    )

    def _validate_parent_child_consistency(self) -> None:
        for parent in self:
            for child_id in parent.children_ids:
                child = self._nodes_by_id[child_id]

                if child.parent_id != parent.node_id:
                    raise HierarchicalIndexError(
                        f"Parent-child relationship mismatch: "
                        f"'{parent.node_id}' lists '{child.node_id}', "
                        f"but child parent_id is '{child.parent_id}'."
                    )

        for child in self:
            if child.parent_id is None:
                continue

            parent = self._nodes_by_id[child.parent_id]

            if child.node_id not in parent.children_ids:
                raise HierarchicalIndexError(
                    f"Parent-child relationship mismatch: "
                    f"'{child.node_id}' points to '{parent.node_id}', "
                    "but the parent does not list the child."
                )

    def _validate_no_cycles(self) -> None:
        for node in self:
            visited: set[str] = set()
            current: DocumentNode | None = node

            while current is not None:
                if current.node_id in visited:
                    raise HierarchicalIndexError(
                        f"Cycle detected involving node: "
                        f"{current.node_id}"
                    )

                visited.add(current.node_id)

                if current.parent_id is None:
                    current = None
                else:
                    current = self._nodes_by_id[
                        current.parent_id
                    ]