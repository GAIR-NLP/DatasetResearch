#!/usr/bin/env python3
"""
Dataset Research CLI Tool
Unified command-line interface for dataset research pipeline.

Commands:
- search: Search for datasets using scripts/method/search_agent.py
- synthesis: Generate synthetic datasets using scripts/method/generation_agent.py  
- run: Execute SFT and inference using evaluation/evaluation_framework.py --step full
- eval: Run evaluation pipeline with option for eval-only or process-only
- metaeval: Run metadata pipeline using scripts/metadata_pipeline.py
- register: Register search datasets to dataset_info.json
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

# Import our enhanced utilities
try:
    from datasetresearch_cli.utils.cli_utils import cli
    from datasetresearch_cli.utils.command_executor import CommandExecutor
    from datasetresearch_cli.utils.config_validator import validate_and_load_config
    ENHANCED_CLI = True
except ImportError:
    # Fallback to basic CLI if dependencies are not installed
    ENHANCED_CLI = False
    import subprocess


class DatasetResearchCLI:
    """Main CLI class for dataset research operations"""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.setup_python_path()
        
        if ENHANCED_CLI:
            self.executor = CommandExecutor(self.project_root)
            # Print welcome banner
            cli.print_banner(
                "Dataset Research CLI",
                "Unified interface for dataset research pipeline"
            )
        else:
            self.executor = None
    
    def setup_python_path(self):
        """Add project root to Python path"""
        if str(self.project_root) not in sys.path:
            sys.path.insert(0, str(self.project_root))
    
    def run_command(self, cmd: List[str], description: str, cwd: Optional[Path] = None, validate_config: bool = True) -> int:
        """Execute command and return exit code with enhanced UI"""
        if cwd is None:
            cwd = self.project_root
        
        # Enhanced execution if available
        if ENHANCED_CLI and self.executor:
            return self.executor.execute_command(
                cmd=cmd,
                description=description,
                show_progress=True,
                cwd=cwd
            )
        else:
            # Fallback to basic execution
            print(f"Running: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, cwd=cwd, check=False)
                return result.returncode
            except Exception as e:
                print(f"Error running command: {e}")
                return 1
    
    def validate_config_if_provided(self, config_path: Optional[str]) -> bool:
        """Validate configuration file if provided"""
        if not config_path or not ENHANCED_CLI:
            return True
        
        config = validate_and_load_config(config_path)
        if config is None and config_path:  # Config was specified but failed to load
            return False
        
        if config and ENHANCED_CLI:
            # Show config summary for user confirmation
            cli.print_config_summary({
                'config_file': config_path,
                'models': len(config.get('models', {})) if 'models' in config else 'N/A',
                'workspace': config.get('workspace', 'Not specified'),
            })
        
        return True


def create_parser():
    """Create argument parser with subcommands"""
    parser = argparse.ArgumentParser(
        description="Dataset Research CLI - Unified interface for dataset research pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  datasetresearch search --config configs/search_config.yaml
  datasetresearch synthesis --config configs/generation_config.yaml
  datasetresearch run --config evaluation/config.yaml --model llama3_8b
  datasetresearch eval --mode eval-only --dataset-set mini
  datasetresearch metaeval --config configs/evaluate_metadata_config.yaml
  datasetresearch register --metadata-file path/to/metadata.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Import command modules
    from datasetresearch_cli.commands.search import add_search_parser
    from datasetresearch_cli.commands.synthesis import add_synthesis_parser
    from datasetresearch_cli.commands.run import add_run_parser
    from datasetresearch_cli.commands.eval import add_eval_parser
    from datasetresearch_cli.commands.metaeval import add_metaeval_parser
    from datasetresearch_cli.commands.register import add_register_parser
    
    # Add subparsers
    add_search_parser(subparsers)
    add_synthesis_parser(subparsers)
    add_run_parser(subparsers)
    add_eval_parser(subparsers)
    add_metaeval_parser(subparsers)
    add_register_parser(subparsers)
    
    return parser


def main():
    """Main CLI entry point"""
    try:
        parser = create_parser()
        args = parser.parse_args()
        
        if not args.command:
            parser.print_help()
            return 1
        
        cli_manager = DatasetResearchCLI()
        
        # Route to appropriate command handler
        if args.command == 'search':
            from datasetresearch_cli.commands.search import handle_search
            return handle_search(cli_manager, args)
        elif args.command == 'synthesis':
            from datasetresearch_cli.commands.synthesis import handle_synthesis
            return handle_synthesis(cli_manager, args)
        elif args.command == 'run':
            from datasetresearch_cli.commands.run import handle_run
            return handle_run(cli_manager, args)
        elif args.command == 'eval':
            from datasetresearch_cli.commands.eval import handle_eval
            return handle_eval(cli_manager, args)
        elif args.command == 'metaeval':
            from datasetresearch_cli.commands.metaeval import handle_metaeval
            return handle_metaeval(cli_manager, args)
        elif args.command == 'register':
            from datasetresearch_cli.commands.register import handle_register
            return handle_register(cli_manager, args)
        else:
            if ENHANCED_CLI:
                cli.print_error(f"Unknown command: {args.command}")
            else:
                print(f"Unknown command: {args.command}")
            return 1
    
    except KeyboardInterrupt:
        if ENHANCED_CLI:
            cli.print_warning("\nOperation interrupted by user")
        else:
            print("\nOperation interrupted by user")
        return 130
    except Exception as e:
        if ENHANCED_CLI:
            cli.print_error(f"Unexpected error: {e}")
            cli.print_info("Please report this issue if it persists")
        else:
            print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())