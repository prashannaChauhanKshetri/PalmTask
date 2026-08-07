"""Chunking strategy module implementing fixed and recursive document chunking.

Both strategies follow the Chunker protocol using tiktoken for exact token counts.
"""

import re
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

import tiktoken

from app.core.exceptions import AppError
from app.core.logging import get_logger

logger = get_logger(__name__)


class ChunkingError(AppError):
    """Raised when document chunking fails or invalid parameters are provided."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"Chunking error: {detail}")


@dataclass(frozen=True)
class ChunkResult:
    """Represents a single extracted chunk with index, text content, and token count."""

    index: int
    text: str
    token_count: int


@runtime_checkable
class Chunker(Protocol):
    """Protocol defining the interface for document chunking strategies."""

    def chunk(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[ChunkResult]:
        """Split input text into chunks of specified maximum token size and overlap.

        Args:
            text: Raw input document text.
            chunk_size: Target max tokens per chunk.
            chunk_overlap: Overlapping tokens between consecutive chunks.

        Returns:
            List of ChunkResult items containing text, token count, and 0-indexed order.
        """
        ...


class FixedWindowChunker:
    """Token-count-based sliding window chunker using tiktoken (cl100k_base).

    Pure sliding window on token stream — deterministic without structural awareness.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)

    def chunk(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[ChunkResult]:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be greater than 0.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be >= 0 and < chunk_size.")

        tokens = self.encoding.encode(text)
        if not tokens:
            return []

        step = chunk_size - chunk_overlap
        chunks: list[ChunkResult] = []
        chunk_idx = 0

        for start_idx in range(0, len(tokens), step):
            end_idx = min(start_idx + chunk_size, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            chunk_text = self.encoding.decode(chunk_tokens).strip()

            if chunk_text:
                chunks.append(
                    ChunkResult(
                        index=chunk_idx,
                        text=chunk_text,
                        token_count=len(chunk_tokens),
                    )
                )
                chunk_idx += 1

            if end_idx >= len(tokens):
                break

        logger.info(
            "Fixed chunking completed",
            extra={"total_tokens": len(tokens), "total_chunks": len(chunks)},
        )
        return chunks


class RecursiveChunker:
    """Structure-aware recursive text chunker.

    Splits hierarchically on paragraphs (`\n\n`), then sentences (`.!?`), then words,
    and recombines small fragments up to chunk_size to maintain semantic coherence.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)
        # Regex splitters: double linebreaks, single linebreaks, sentence boundaries, spaces
        self.separators: list[str] = ["\n\n", "\n", r"(?<=[.!?])\s+", " "]

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _split_text(self, text: str, separator_idx: int) -> list[str]:
        if separator_idx >= len(self.separators):
            # Last resort: fallback to single characters
            return list(text)

        sep = self.separators[separator_idx]
        if sep.startswith("(") and sep.endswith("+"):
            splits = [s for s in re.split(sep, text) if s.strip()]
        elif sep in ("\n\n", "\n", " "):
            splits = [s for s in text.split(sep) if s.strip()]
        else:
            splits = [s for s in re.split(sep, text) if s.strip()]

        return splits if splits else [text]

    def _recursive_split(
        self, text: str, chunk_size: int, separator_idx: int
    ) -> list[str]:
        """Split text until every piece is <= chunk_size tokens or separators exhausted."""
        if self._token_count(text) <= chunk_size:
            return [text]

        splits = self._split_text(text, separator_idx)
        final_pieces: list[str] = []

        for split in splits:
            if self._token_count(split) <= chunk_size:
                final_pieces.append(split)
            else:
                # Recurse with finer separator
                sub_pieces = self._recursive_split(
                    split, chunk_size, separator_idx + 1
                )
                final_pieces.extend(sub_pieces)

        return final_pieces

    def chunk(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[ChunkResult]:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be greater than 0.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be >= 0 and < chunk_size.")

        if not text.strip():
            return []

        # Step 1: Hierarchically split into small semantic pieces
        raw_pieces = self._recursive_split(text, chunk_size, separator_idx=0)

        # Step 2: Combine small adjacent pieces into cohesive chunks up to chunk_size
        combined_chunks: list[str] = []
        current_pieces: list[str] = []
        current_tokens = 0

        for piece in raw_pieces:
            piece_tokens = self._token_count(piece)

            if current_tokens + piece_tokens <= chunk_size:
                current_pieces.append(piece)
                current_tokens += piece_tokens
            else:
                if current_pieces:
                    combined_chunks.append("\n\n".join(current_pieces))

                # Handle overlap: keep ending pieces whose combined tokens <= chunk_overlap
                overlap_pieces: list[str] = []
                overlap_tokens = 0
                for prev_piece in reversed(current_pieces):
                    prev_tok = self._token_count(prev_piece)
                    if overlap_tokens + prev_tok <= chunk_overlap:
                        overlap_pieces.insert(0, prev_piece)
                        overlap_tokens += prev_tok
                    else:
                        break

                current_pieces = overlap_pieces + [piece]
                current_tokens = overlap_tokens + piece_tokens

        if current_pieces:
            combined_chunks.append("\n\n".join(current_pieces))

        # Build output results
        results: list[ChunkResult] = []
        for idx, chunk_text in enumerate(combined_chunks):
            cleaned = chunk_text.strip()
            if cleaned:
                tok_cnt = self._token_count(cleaned)
                results.append(
                    ChunkResult(index=idx, text=cleaned, token_count=tok_cnt)
                )

        logger.info(
            "Recursive chunking completed",
            extra={"pieces": len(raw_pieces), "total_chunks": len(results)},
        )
        return results


class HierarchicalChunker:
    """Structure-aware hierarchical chunker that preserves document section context.

    Detects document structure via markdown-style headers (#, ##, ###) or
    numbered sections, and prepends parent section context to each child chunk
    so the LLM always knows where in the document a chunk originates.

    Produces fundamentally different output than fixed/recursive — each chunk
    carries its section lineage as a prefix for improved RAG retrieval.
    """

    # Regex patterns for detecting section headers (markdown, numbered, underlined)
    HEADER_PATTERNS: list[tuple[re.Pattern[str], int]] = [
        (re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE), 0),         # Markdown headers
        (re.compile(r"^(\d+\.(?:\d+\.)*)\s+(.+)$", re.MULTILINE), 0), # Numbered sections
        (re.compile(r"^([A-Z][A-Z\s]{2,})$", re.MULTILINE), 1),       # ALL-CAPS headings
    ]

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self.encoding = tiktoken.get_encoding(encoding_name)

    def _token_count(self, text: str) -> int:
        return len(self.encoding.encode(text))

    def _detect_sections(self, text: str) -> list[tuple[str, str, int]]:
        """Parse document into (header_text, body_text, depth) sections.

        Returns a list of sections with their header, body content, and nesting depth.
        """
        lines = text.split("\n")
        sections: list[tuple[str, str, int]] = []
        current_header = ""
        current_depth = 0
        current_body_lines: list[str] = []

        for line in lines:
            is_header = False
            header_text = ""
            depth = 0

            # Check markdown headers: # Title, ## Subtitle, etc.
            md_match = re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if md_match:
                is_header = True
                depth = len(md_match.group(1))
                header_text = md_match.group(2).strip()

            # Check numbered sections: 1. Title, 1.2. Subtitle
            if not is_header:
                num_match = re.match(r"^(\d+\.(?:\d+\.)*)\s+(.+)$", line.strip())
                if num_match:
                    is_header = True
                    depth = num_match.group(1).count(".")
                    header_text = f"{num_match.group(1)} {num_match.group(2).strip()}"

            # Check ALL-CAPS headings (depth 1)
            if not is_header:
                caps_match = re.match(r"^([A-Z][A-Z\s]{4,})$", line.strip())
                if caps_match and len(line.strip()) > 3:
                    is_header = True
                    depth = 1
                    header_text = caps_match.group(1).strip()

            if is_header:
                # Flush previous section
                body = "\n".join(current_body_lines).strip()
                if body or current_header:
                    sections.append((current_header, body, current_depth))

                current_header = header_text
                current_depth = depth
                current_body_lines = []
            else:
                current_body_lines.append(line)

        # Flush last section
        body = "\n".join(current_body_lines).strip()
        if body or current_header:
            sections.append((current_header, body, current_depth))

        return sections

    def _build_context_prefix(
        self, header_stack: list[str]
    ) -> str:
        """Build hierarchical breadcrumb prefix from current header ancestry."""
        if not header_stack:
            return ""
        breadcrumb = " > ".join(h for h in header_stack if h)
        return f"[Section: {breadcrumb}]\n" if breadcrumb else ""

    def chunk(
        self, text: str, chunk_size: int = 512, chunk_overlap: int = 50
    ) -> list[ChunkResult]:
        if chunk_size <= 0:
            raise ChunkingError("chunk_size must be greater than 0.")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ChunkingError("chunk_overlap must be >= 0 and < chunk_size.")

        if not text.strip():
            return []

        sections = self._detect_sections(text)

        # If no structure detected, fall back to recursive chunker behavior
        if len(sections) <= 1 and not sections[0][0]:
            logger.info("No document structure detected; falling back to recursive chunking")
            fallback = RecursiveChunker(encoding_name="cl100k_base")
            return fallback.chunk(text, chunk_size, chunk_overlap)

        # Build chunks with section context prefixes
        header_stack: list[str] = []
        results: list[ChunkResult] = []
        chunk_idx = 0

        for header, body, depth in sections:
            # Maintain header stack based on nesting depth
            if depth > 0:
                # Trim stack to current depth and push new header
                header_stack = header_stack[: depth - 1]
                header_stack.append(header)
            elif header:
                header_stack = [header]

            if not body:
                continue

            context_prefix = self._build_context_prefix(header_stack)
            prefix_tokens = self._token_count(context_prefix) if context_prefix else 0
            available_tokens = chunk_size - prefix_tokens

            if available_tokens <= 0:
                available_tokens = chunk_size

            # Split body into paragraphs and combine up to available_tokens
            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            current_parts: list[str] = []
            current_tokens = 0

            for para in paragraphs:
                para_tokens = self._token_count(para)

                if para_tokens > available_tokens:
                    # Paragraph too large — sentence split
                    if current_parts:
                        chunk_text = context_prefix + "\n\n".join(current_parts)
                        tok_cnt = self._token_count(chunk_text)
                        results.append(ChunkResult(index=chunk_idx, text=chunk_text.strip(), token_count=tok_cnt))
                        chunk_idx += 1
                        current_parts = []
                        current_tokens = 0

                    sentences = re.split(r"(?<=[.!?])\s+", para)
                    sent_parts: list[str] = []
                    sent_tokens = 0
                    for sent in sentences:
                        st = self._token_count(sent)
                        if sent_tokens + st <= available_tokens:
                            sent_parts.append(sent)
                            sent_tokens += st
                        else:
                            if sent_parts:
                                chunk_text = context_prefix + " ".join(sent_parts)
                                tok_cnt = self._token_count(chunk_text)
                                results.append(ChunkResult(index=chunk_idx, text=chunk_text.strip(), token_count=tok_cnt))
                                chunk_idx += 1
                            sent_parts = [sent]
                            sent_tokens = st
                    if sent_parts:
                        chunk_text = context_prefix + " ".join(sent_parts)
                        tok_cnt = self._token_count(chunk_text)
                        results.append(ChunkResult(index=chunk_idx, text=chunk_text.strip(), token_count=tok_cnt))
                        chunk_idx += 1

                elif current_tokens + para_tokens <= available_tokens:
                    current_parts.append(para)
                    current_tokens += para_tokens
                else:
                    if current_parts:
                        chunk_text = context_prefix + "\n\n".join(current_parts)
                        tok_cnt = self._token_count(chunk_text)
                        results.append(ChunkResult(index=chunk_idx, text=chunk_text.strip(), token_count=tok_cnt))
                        chunk_idx += 1
                    current_parts = [para]
                    current_tokens = para_tokens

            if current_parts:
                chunk_text = context_prefix + "\n\n".join(current_parts)
                tok_cnt = self._token_count(chunk_text)
                results.append(ChunkResult(index=chunk_idx, text=chunk_text.strip(), token_count=tok_cnt))
                chunk_idx += 1

        logger.info(
            "Hierarchical chunking completed",
            extra={"sections_found": len(sections), "total_chunks": len(results)},
        )
        return results


class ChunkerFactory:
    """Factory dispatch for chunking strategies."""

    @staticmethod
    def get_chunker(strategy: Literal["fixed", "recursive", "hierarchical"]) -> Chunker:
        """Instantiate a chunker by strategy name.

        Raises:
            ChunkingError: If an unrecognized strategy name is provided.
        """
        if strategy == "fixed":
            return FixedWindowChunker()
        elif strategy == "recursive":
            return RecursiveChunker()
        elif strategy == "hierarchical":
            return HierarchicalChunker()
        else:
            raise ChunkingError(
                f"Unknown chunking strategy: '{strategy}'. Use 'fixed', 'recursive', or 'hierarchical'."
            )
