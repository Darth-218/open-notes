from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedHeading:
    """Represents a heading parsed from a Markdown document.

    Attributes:
        level: The heading level (1-6, corresponding to # to ######).
        text: The heading text without the leading # characters.
        position: The character position where the heading starts in the document.
    """

    level: int
    text: str
    position: int


class MarkdownParser:
    """Parser for extracting headings from Markdown documents.

    Uses regular expressions to identify headings (# through ######) and
    track their position in the document for building heading paths.
    """

    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def parse(self, content: str) -> tuple[list[ParsedHeading], str]:
        """Parse a Markdown document to extract all headings.

        Args:
            content: The Markdown content to parse.

        Returns:
            A tuple containing:
                - list[ParsedHeading]: List of all headings found in the document.
                - str: The original content (unchanged).
        """
        headings = []
        heading_positions = []

        for match in self.HEADING_RE.finditer(content):
            level = len(match.group(1))
            text = match.group(2).strip()
            position = match.start()
            headings.append(ParsedHeading(level=level, text=text, position=position))
            heading_positions.append(position)

        return headings, content

    def extract_heading_path(
        self, headings: list[ParsedHeading], position: int
    ) -> str:
        """Extract the heading path for a given position in the document.

        Builds a path by joining all headings that appear before the given
        position, separated by "/".

        Args:
            headings: List of parsed headings from the document.
            position: The character position to find the heading path for.

        Returns:
            A string representing the path (e.g., "Introduction/Background/Goals"),
            or an empty string if no headings exist.
        """
        if not headings:
            return ""

        path_parts = []
        for h in headings:
            if h.position <= position:
                path_parts.append(h.text)
            else:
                break

        return "/".join(path_parts)