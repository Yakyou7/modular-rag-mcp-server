"""Response Builder for constructing MCP-formatted responses.

This module builds structured responses for MCP tools, combining:
- Human-readable Markdown content with citation markers
- Structured citation data for machine consumption
- Multimodal content (text + images) support
- Proper handling of empty results and error cases
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

from mcp import types

from src.core.response.citation_generator import Citation, CitationGenerator
from src.core.response.confidence import ConfidenceAssessment, RERANK_SCORE_MAX
from src.core.types import RetrievalResult


@dataclass
class MCPToolResponse:
    """Structured response for MCP tools.
    
    Attributes:
        content: Human-readable Markdown content with citation markers [1], [2], etc.
        citations: List of structured citations for reference
        metadata: Additional response metadata (query, result_count, etc.)
        is_empty: Whether the search returned no results
        image_contents: List of MCP ImageContent blocks for multimodal responses
    """
    content: str
    citations: List[Citation] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_empty: bool = False
    image_contents: List[types.ImageContent] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for MCP protocol.
        
        Returns:
            Dictionary with 'content' and 'structuredContent' fields.
        """
        return {
            "content": self.content,
            "structuredContent": {
                "citations": [c.to_dict() for c in self.citations],
                "metadata": self.metadata,
                "isEmpty": self.is_empty,
            }
        }
    
    def to_mcp_content(self) -> List[Union[types.TextContent, types.ImageContent]]:
        """Convert to MCP content blocks format.
        
        Returns:
            List of content blocks for MCP CallToolResult.
            Includes TextContent and optionally ImageContent blocks.
        """
        blocks: List[Union[types.TextContent, types.ImageContent]] = [
            types.TextContent(
                type="text",
                text=self.content,
            )
        ]
        
        # Add image blocks if present (multimodal response)
        if self.image_contents:
            blocks.extend(self.image_contents)
        
        # Add structured data as a separate text block (JSON format)
        if self.citations or self.metadata:
            import json
            structured = {
                "citations": [c.to_dict() for c in self.citations],
                "metadata": self.metadata,
                "has_images": len(self.image_contents) > 0,
                "image_count": len(self.image_contents),
            }
            blocks.append(
                types.TextContent(
                    type="text",
                    text=f"\n---\n**References (JSON):**\n```json\n{json.dumps(structured, ensure_ascii=False, indent=2)}\n```",
                )
            )
        
        return blocks
    
    @property
    def has_images(self) -> bool:
        """Check if response contains images.
        
        Returns:
            True if response has image content, False otherwise.
        """
        return len(self.image_contents) > 0


class ResponseBuilder:
    """Builds MCP-formatted responses from retrieval results.
    
    This class transforms retrieval results into structured MCP responses,
    including human-readable Markdown with inline citations and structured
    citation data for machine consumption.
    
    Supports multimodal responses with images when results contain image
    references in their metadata.
    
    Example:
        >>> builder = ResponseBuilder()
        >>> results = [RetrievalResult(chunk_id="doc1_001", score=0.95, ...)]
        >>> response = builder.build(results, "What is Azure OpenAI?")
        >>> print(response.content)  # Markdown with [1], [2] markers
        >>> print(response.citations[0].source)  # "docs/guide.pdf"
        >>> print(response.has_images)  # True if images found
    """
    
    def __init__(
        self,
        citation_generator: Optional[CitationGenerator] = None,
        multimodal_assembler: Optional["MultimodalAssembler"] = None,
        max_results_in_content: int = 5,
        snippet_max_length: int = 300,
        enable_multimodal: bool = True,
    ) -> None:
        """Initialize ResponseBuilder.
        
        Args:
            citation_generator: Optional CitationGenerator instance.
                If None, creates a default one.
            multimodal_assembler: Optional MultimodalAssembler for image handling.
                If None and enable_multimodal=True, creates a default one.
            max_results_in_content: Maximum results to show in Markdown content.
            snippet_max_length: Maximum characters per result snippet in content.
            enable_multimodal: Whether to include images in response (default: True).
        """
        self.citation_generator = citation_generator or CitationGenerator()
        self.max_results_in_content = max_results_in_content
        self.snippet_max_length = snippet_max_length
        self.enable_multimodal = enable_multimodal
        
        # Lazy-load multimodal assembler to avoid circular imports
        self._multimodal_assembler = multimodal_assembler
    
    @property
    def multimodal_assembler(self) -> "MultimodalAssembler":
        """Get or create MultimodalAssembler instance."""
        if self._multimodal_assembler is None:
            from src.core.response.multimodal_assembler import MultimodalAssembler
            self._multimodal_assembler = MultimodalAssembler()
        return self._multimodal_assembler
    
    def build(
        self,
        results: List[RetrievalResult],
        query: str,
        collection: Optional[str] = None,
        include_images: bool = True,
        confidence: Optional[ConfidenceAssessment] = None,
    ) -> MCPToolResponse:
        """Build MCP response from retrieval results.

        Args:
            results: List of RetrievalResult from search.
            query: Original user query.
            collection: Optional collection name.
            include_images: Whether to include images in response (default: True).
            confidence: Optional ConfidenceAssessment from the tool layer.
                When assessable and low, prepends a refusal banner and adds
                confidence fields to metadata.

        Returns:
            MCPToolResponse with formatted content, citations, and optional images.
        """
        # Handle empty results
        if not results:
            return self._build_empty_response(query, collection)

        # Generate citations
        citations = self.citation_generator.generate(results)

        # Build Markdown content
        content = self._build_markdown_content(results, citations, query)

        # Low-confidence banner: honest refusal-style header, but still
        # return the closest materials (never silently output weak matches).
        if confidence is not None and confidence.low:
            content = self._build_low_confidence_banner(confidence) + content

        # Build metadata
        metadata = self._build_metadata(query, collection, len(results))
        if confidence is not None:
            metadata["confidence_method"] = confidence.method
            if confidence.assessable:
                metadata["confidence"] = round(confidence.confidence, 3)
                metadata["confidence_threshold"] = confidence.threshold
                metadata["low_confidence"] = bool(confidence.low)
        
        # Assemble image content if enabled
        image_contents: List[types.ImageContent] = []
        if self.enable_multimodal and include_images:
            image_blocks = self.multimodal_assembler.assemble(results, collection)
            # Filter to only ImageContent blocks
            image_contents = [
                block for block in image_blocks
                if isinstance(block, types.ImageContent)
            ]
            if image_contents:
                metadata["has_images"] = True
                metadata["image_count"] = len(image_contents)
        
        return MCPToolResponse(
            content=content,
            citations=citations,
            metadata=metadata,
            is_empty=False,
            image_contents=image_contents,
        )
    
    def _build_empty_response(
        self,
        query: str,
        collection: Optional[str] = None,
    ) -> MCPToolResponse:
        """Build response for empty results.
        
        Args:
            query: Original user query.
            collection: Optional collection name.
            
        Returns:
            MCPToolResponse indicating no results found.
        """
        content = f"## 未找到相关结果\n\n"
        content += f"查询: **{query}**\n\n"
        
        if collection:
            content += f"在集合 `{collection}` 中未找到与查询相关的文档。\n\n"
        else:
            content += "未找到与查询相关的文档。\n\n"
        
        content += "**建议:**\n"
        content += "- 尝试使用不同的关键词\n"
        content += "- 检查是否已摄取相关文档\n"
        content += "- 扩大搜索范围（如不指定 collection）\n"
        
        metadata = self._build_metadata(query, collection, 0)
        
        return MCPToolResponse(
            content=content,
            citations=[],
            metadata=metadata,
            is_empty=True,
        )
    
    def _build_markdown_content(
        self,
        results: List[RetrievalResult],
        citations: List[Citation],
        query: str,
    ) -> str:
        """Build Markdown content with inline citations.
        
        Args:
            results: List of RetrievalResult.
            citations: List of Citation objects.
            query: Original query string.
            
        Returns:
            Formatted Markdown string.
        """
        lines = []
        
        # Header
        lines.append(f"## 检索结果\n")
        lines.append(f"针对查询 **\"{query}\"** 找到 {len(results)} 条相关结果:\n")
        
        # Results section
        display_count = min(len(results), self.max_results_in_content)
        
        for i, (result, citation) in enumerate(zip(results[:display_count], citations[:display_count])):
            marker = self.citation_generator.format_citation_marker(citation.index)

            # Format single result
            lines.append(f"### {marker} 结果 {citation.index}")
            lines.append(f"**{self._format_relevance(result, citation)}**")
            lines.append(f"**来源:** `{citation.source}`")
            
            if citation.page is not None:
                lines.append(f"**页码:** {citation.page}")
            
            # Content snippet
            snippet = self._truncate_text(result.text, self.snippet_max_length)
            lines.append(f"\n> {snippet}\n")
        
        # Additional results indicator
        if len(results) > display_count:
            remaining = len(results) - display_count
            lines.append(f"\n*...还有 {remaining} 条结果未显示*\n")
        
        # References section
        lines.append("\n---\n")
        lines.append("## 引用来源\n")
        
        for citation in citations:
            source_info = f"`{citation.source}`"
            if citation.page is not None:
                source_info += f" (p.{citation.page})"
            lines.append(f"- [{citation.index}] {source_info}")
        
        return "\n".join(lines)
    
    def _format_relevance(self, result: RetrievalResult, citation: Citation) -> str:
        """Format a scale-aware relevance label for one result.

        The rerank prompt (config/prompts/rerank.txt) scores relevance on a
        0-3 scale, so displaying it as a percentage would show "300.00%".
        RRF fusion scores (~1/(k+rank)) are rank-derived — shown as-is
        rather than disguised as a relevance percentage.
        """
        metadata = result.metadata or {}
        if metadata.get("reranked"):
            raw = max(0.0, min(float(citation.score), RERANK_SCORE_MAX))
            normalized = raw / RERANK_SCORE_MAX
            return f"相关度: {normalized:.0%}（精排分 {raw:g}/3）"
        return f"融合分: {citation.score:.4f}（RRF 排名分）"

    def _build_low_confidence_banner(self, confidence: ConfidenceAssessment) -> str:
        """Build the low-confidence banner prepended to the content.

        Args:
            confidence: A low, assessable ConfidenceAssessment.

        Returns:
            Markdown banner text (ends with a blank line).
        """
        lines = [
            "## ⚠️ 低置信度结果（仅供参考）",
            "",
            (
                f"检索系统对本次查询的把握不足：置信度 "
                f"{confidence.confidence:.0%} 低于阈值 {confidence.threshold:.0%}。"
            ),
            "知识库中可能没有直接回答该问题的资料，以下是最接近的内容，请谨慎采用。",
            "",
        ]
        return "\n".join(lines)

    def _build_metadata(
        self,
        query: str,
        collection: Optional[str],
        result_count: int,
    ) -> Dict[str, Any]:
        """Build response metadata.
        
        Args:
            query: Original query.
            collection: Collection name.
            result_count: Number of results.
            
        Returns:
            Metadata dictionary.
        """
        metadata = {
            "query": query,
            "result_count": result_count,
        }
        if collection:
            metadata["collection"] = collection
        return metadata
    
    def _truncate_text(self, text: str, max_length: int) -> str:
        """Truncate text to maximum length.
        
        Args:
            text: Text to truncate.
            max_length: Maximum characters.
            
        Returns:
            Truncated text with ellipsis if needed.
        """
        if not text:
            return ""
        
        # Clean whitespace
        cleaned = " ".join(text.split())
        
        if len(cleaned) <= max_length:
            return cleaned
        
        # Truncate at word boundary
        truncated = cleaned[:max_length].rsplit(" ", 1)[0]
        return truncated + "..."
