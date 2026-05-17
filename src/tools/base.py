"""
Base tool interface and tool registry for CXR diagnostic agents.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Any, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ToolResult:
    """Output from a tool invocation."""
    tool_name: str
    success: bool
    output: str
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class BaseTool(ABC):
    """Abstract base class for all CXR diagnostic tools."""

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, image_path: str, **kwargs) -> ToolResult:
        """Execute the tool on an image."""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """Return a JSON-like schema describing the tool's interface."""
        return {
            "name": self.name,
            "description": self.description,
        }

    def __repr__(self) -> str:
        return f"{self.name}: {self.description}"


class ToolRegistry:
    """Manages available tools and provides lookup."""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def register_all(self, tools: List[BaseTool]) -> None:
        for tool in tools:
            self.register(tool)

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def list_tools(self) -> List[BaseTool]:
        return list(self._tools.values())

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def get_schema_text(self) -> str:
        """Generate a text description of all available tools for the agent prompt."""
        lines = ["Available Tools:"]
        for tool in self._tools.values():
            lines.append(f"- {tool.name}: {tool.description}")
        return "\n".join(lines)

    def run_tool(self, name: str, image_path: str, **kwargs) -> ToolResult:
        """Execute a tool by name. Returns error result if tool not found."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(
                tool_name=name,
                success=False,
                output=f"Error: Tool '{name}' not found. Available: {', '.join(self.list_names())}"
            )
        try:
            return tool.run(image_path, **kwargs)
        except Exception as e:
            logger.error(f"Tool '{name}' failed: {e}")
            return ToolResult(
                tool_name=name,
                success=False,
                output=f"Error executing {name}: {str(e)}"
            )
