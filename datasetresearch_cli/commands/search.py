"""
Search command for dataset research CLI
"""

import sys

try:
    from datasetresearch_cli.utils.cli_utils import cli
    ENHANCED_CLI = True
except ImportError:
    ENHANCED_CLI = False


def add_search_parser(subparsers):
    """Add search command parser"""
    search_parser = subparsers.add_parser(
        'search', 
        help='Search for datasets',
        description='Search for relevant datasets using LLM-powered agents'
    )
    search_parser.add_argument('--config', required=True, help='Configuration file path (YAML format)')
    search_parser.add_argument('--dry-run', action='store_true', help='Show what would be executed without running')
    return search_parser


def handle_search(cli_manager, args):
    """Execute dataset search using search_data.py"""
    
    # Build command - search_data.py requires --config parameter
    cmd = [sys.executable, "scripts/method/search_data.py", "--config", args.config]
    
    # Handle dry run
    if hasattr(args, 'dry_run') and args.dry_run:
        if ENHANCED_CLI:
            cli.print_info("Dry run mode - showing command that would be executed:")
            cli.console.print(f"  {' '.join(cmd)}", style="cyan")
            return 0
        else:
            print(f"Would execute: {' '.join(cmd)}")
            return 0
    
    # Show search information
    if ENHANCED_CLI:
        cli.print_step(1, 1, "Starting dataset search")
        cli.print_info(f"Running search with configuration: {args.config}")
        cli.print_warning("This may take a few minutes depending on query complexity...")
        
        # Show info about config
        cli.print_info(f"Using search_data.py with config file: {args.config}")
    
    # Execute command
    result = cli_manager.run_command(
        cmd=cmd, 
        description="Dataset Search"
    )
    
    # Show results
    if ENHANCED_CLI:
        if result == 0:
            cli.print_success("Dataset search completed!")
            cli.print_info("Check the output files for search results")
        else:
            cli.print_error("Dataset search failed!")
            cli.print_info("Check the error output above for details")
    
    return result