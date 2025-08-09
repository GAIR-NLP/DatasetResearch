"""
CLI utilities for enhanced user experience
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.prompt import Confirm, Prompt
from rich import print as rich_print


class CLIManager:
    """Enhanced CLI manager with rich output formatting"""
    
    def __init__(self):
        self.console = Console()
        self.progress = None
        
    def print_banner(self, title: str, subtitle: str = ""):
        """Print a beautiful banner"""
        banner_text = Text(title, style="bold blue")
        if subtitle:
            banner_text.append(f"\n{subtitle}", style="dim")
        
        panel = Panel(
            banner_text,
            border_style="blue",
            padding=(1, 2)
        )
        self.console.print(panel)
    
    def print_success(self, message: str):
        """Print success message"""
        self.console.print(f"✅ {message}", style="bold green")
    
    def print_error(self, message: str, details: Optional[str] = None):
        """Print error message with optional details"""
        self.console.print(f"❌ {message}", style="bold red")
        if details:
            self.console.print(f"   {details}", style="dim red")
    
    def print_warning(self, message: str):
        """Print warning message"""
        self.console.print(f"⚠️  {message}", style="bold yellow")
    
    def print_info(self, message: str):
        """Print info message"""
        self.console.print(f"ℹ️  {message}", style="bold blue")
    
    def print_step(self, step_num: int, total_steps: int, message: str):
        """Print step indicator"""
        self.console.print(f"[{step_num}/{total_steps}] {message}", style="bold cyan")
    
    def create_progress_bar(self, description: str = "Processing..."):
        """Create a progress bar"""
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=self.console
        )
        return self.progress
    
    def print_table(self, data: list, headers: list, title: str = ""):
        """Print data as a formatted table"""
        table = Table(title=title, show_header=True, header_style="bold magenta")
        
        for header in headers:
            table.add_column(header)
        
        for row in data:
            table.add_row(*[str(cell) for cell in row])
        
        self.console.print(table)
    
    def print_config_summary(self, config: Dict[str, Any]):
        """Print configuration summary"""
        table = Table(title="Configuration Summary", show_header=True)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        
        for key, value in config.items():
            if isinstance(value, (dict, list)):
                value_str = f"{type(value).__name__} ({len(value)} items)"
            else:
                value_str = str(value)
            table.add_row(key, value_str)
        
        self.console.print(table)
    
    def confirm_action(self, message: str, default: bool = False) -> bool:
        """Ask for user confirmation"""
        return Confirm.ask(message, default=default)
    
    def prompt_input(self, message: str, default: Optional[str] = None) -> str:
        """Prompt for user input"""
        return Prompt.ask(message, default=default)
    
    def print_command_info(self, command: str, description: str, running_cmd: str):
        """Print command execution info"""
        info_panel = Panel(
            f"[bold]{command}[/bold]\n{description}\n\n[dim]Running: {running_cmd}[/dim]",
            title="Command Info",
            border_style="cyan"
        )
        self.console.print(info_panel)


# Global CLI manager instance
cli = CLIManager()