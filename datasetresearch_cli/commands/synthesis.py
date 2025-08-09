"""
Synthesis command for dataset research CLI
"""

import sys

try:
    from datasetresearch_cli.utils.cli_utils import cli
    ENHANCED_CLI = True
except ImportError:
    ENHANCED_CLI = False


def add_synthesis_parser(subparsers):
    """Add synthesis command parser"""
    synthesis_parser = subparsers.add_parser(
        'synthesis', 
        help='Generate synthetic datasets',
        description='Generate synthetic training datasets using LLM agents'
    )
    # Based on actual generation_agent.py parameters
    synthesis_parser.add_argument('--config', help='Configuration file path (YAML format)')
    synthesis_parser.add_argument('--agent-query-file', help='Agent query file path')
    synthesis_parser.add_argument('--metadata-file', help='Metadata file path')
    synthesis_parser.add_argument('--w-example-output-dir', help='Output directory for data with examples')
    synthesis_parser.add_argument('--wo-example-output-dir', help='Output directory for data without examples')
    synthesis_parser.add_argument('--results-filename', default='generation_results.json', help='Results filename')
    synthesis_parser.add_argument('--num-data', type=int, help='Number of data items to generate per dataset')
    synthesis_parser.add_argument('--max-workers', type=int, help='Maximum concurrent threads')
    synthesis_parser.add_argument('--skip-existing', action='store_true', help='Skip existing valid files')
    synthesis_parser.add_argument('--dry-run', action='store_true', help='Show what would be executed without running')
    return synthesis_parser


def handle_synthesis(cli_manager, args):
    """Execute dataset synthesis using generation_agent.py"""
    
    # Validate configuration if provided
    if hasattr(args, 'config') and args.config:
        if not cli_manager.validate_config_if_provided(args.config):
            return 1
    
    # Build command
    cmd = [sys.executable, "scripts/method/generation_agent.py"]
    
    # Add actual supported arguments
    if hasattr(args, 'config') and args.config:
        cmd.extend(["--config", args.config])
    if hasattr(args, 'agent_query_file') and args.agent_query_file:
        cmd.extend(["--agent-query-file", args.agent_query_file])
    if hasattr(args, 'metadata_file') and args.metadata_file:
        cmd.extend(["--metadata-file", args.metadata_file])
    if hasattr(args, 'w_example_output_dir') and args.w_example_output_dir:
        cmd.extend(["--w-example-output-dir", args.w_example_output_dir])
    if hasattr(args, 'wo_example_output_dir') and args.wo_example_output_dir:
        cmd.extend(["--wo-example-output-dir", args.wo_example_output_dir])
    if hasattr(args, 'results_filename') and args.results_filename:
        cmd.extend(["--results-filename", args.results_filename])
    if hasattr(args, 'num_data') and args.num_data:
        cmd.extend(["--num-data", str(args.num_data)])
    if hasattr(args, 'max_workers') and args.max_workers:
        cmd.extend(["--max-workers", str(args.max_workers)])
    if hasattr(args, 'skip_existing') and args.skip_existing:
        cmd.extend(["--skip-existing"])
    
    # Handle dry run
    if hasattr(args, 'dry_run') and args.dry_run:
        if ENHANCED_CLI:
            cli.print_info("Dry run mode - showing command that would be executed:")
            cli.console.print(f"  {' '.join(cmd)}", style="cyan")
            return 0
        else:
            print(f"Would execute: {' '.join(cmd)}")
            return 0
    
    # Show synthesis information
    if ENHANCED_CLI:
        cli.print_step(1, 2, "Starting synthetic data generation")
        cli.print_info("This process may take considerable time for large datasets...")
        
        # Show configuration summary
        config_summary = {
            'Config File': args.config if hasattr(args, 'config') and args.config else 'Default',
            'Metadata File': args.metadata_file if hasattr(args, 'metadata_file') and args.metadata_file else 'Default',
            'Num Data': str(args.num_data) if hasattr(args, 'num_data') and args.num_data else 'Default',
            'Max Workers': str(args.max_workers) if hasattr(args, 'max_workers') and args.max_workers else 'Default',
            'Skip Existing': 'Yes' if hasattr(args, 'skip_existing') and args.skip_existing else 'No'
        }
        cli.print_config_summary(config_summary)
    
    # Execute command
    result = cli_manager.run_command(
        cmd=cmd, 
        description="Synthetic Data Generation"
    )
    
    # Show results
    if ENHANCED_CLI:
        if result == 0:
            cli.print_success("Synthetic data generation completed!")
            cli.print_info("Generated data saved to output directories")
        else:
            cli.print_error("Synthetic data generation failed!")
            cli.print_info("Check the error output above for details")
    
    return result