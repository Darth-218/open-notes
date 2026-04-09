from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass
class ParsedHeading:
    level: int
    text: str
    position: int


class MarkdownParser:
    HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)

    def parse(self, content: str) -> tuple[list[ParsedHeading], str]:
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
        if not headings:
            return ""

        path_parts = []
        for h in headings:
            if h.position <= position:
                path_parts.append(h.text)
            else:
                break

        return "/".join(path_parts)
