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


class ChunkerFactory:
    """Factory dispatch for chunking strategies."""

    @staticmethod
    def get_chunker(strategy: Literal["fixed", "recursive"]) -> Chunker:
        """Instantiate a chunker by strategy name.

        Raises:
            ChunkingError: If an unrecognized strategy name is provided.
        """
        if strategy == "fixed":
            return FixedWindowChunker()
        elif strategy == "recursive":
            return RecursiveChunker()
        else:
            raise ChunkingError(f"Unknown chunking strategy: '{strategy}'. Use 'fixed' or 'recursive'.")
