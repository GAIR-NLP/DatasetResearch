"""
Metaeval command for dataset research CLI
"""

import sys

try:
    from datasetresearch_cli.utils.cli_utils import cli
    ENHANCED_CLI = True
except ImportError:
    ENHANCED_CLI = False


def add_metaeval_parser(subparsers):
    """Add metaeval command parser"""
    metaeval_parser = subparsers.add_parser(
        'metaeval', 
        help='Run metadata pipeline',
        description='Run metadata evaluation pipeline to assess dataset quality and characteristics'
    )
    # Based on actual metadata_pipeline.py parameters
    metaeval_parser.add_argument(
        '--config', 
        default='configs/evaluate_metadata_config.yaml',
        help='Configuration file path (default: configs/evaluate_metadata_config.yaml)'
    )
    metaeval_parser.add_argument(
        '--mode', 
        choices=['full', 'generate_only', 'evaluate_only'],
        default='full', 
        help='Pipeline mode: full (default), generate_only, or evaluate_only'
    )
    metaeval_parser.add_argument(
        '--input-file', 
        help='Input file (needed for generation step)'
    )
    metaeval_parser.add_argument(
        '--output-dir', 
        help='Output directory (overrides config file)'
    )
    # Advanced API options
    metaeval_parser.add_argument(
        '--api-key', 
        help='Override API key'
    )
    metaeval_parser.add_argument(
        '--api-base', 
        help='Override API base URL'
    )
    metaeval_parser.add_argument(
        '--api-model', 
        help='Override API model'
    )
    # Skip options  
    metaeval_parser.add_argument(
        '--skip-generation', 
        action='store_true',
        help='Skip generation step'
    )
    metaeval_parser.add_argument(
        '--skip-evaluation', 
        action='store_true',
        help='Skip evaluation step'
    )
    metaeval_parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be executed without running'
    )
    return metaeval_parser


def handle_metaeval(cli_manager, args):
    """Execute metadata pipeline using metadata_pipeline.py"""
    
    # Validate configuration if provided
    if not cli_manager.validate_config_if_provided(args.config):
        return 1
    
    # Build command
    cmd = [sys.executable, "scripts/metadata_pipeline.py"]
    
    # Add configuration arguments based on actual script parameters
    if hasattr(args, 'config') and args.config:
        cmd.extend(["--config", args.config])
    if hasattr(args, 'mode') and args.mode:
        cmd.extend(["--mode", args.mode])
    if hasattr(args, 'input_file') and args.input_file:
        cmd.extend(["--input-file", args.input_file])
    if hasattr(args, 'output_dir') and args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])
    if hasattr(args, 'api_key') and args.api_key:
        cmd.extend(["--api-key", args.api_key])
    if hasattr(args, 'api_base') and args.api_base:
        cmd.extend(["--api-base", args.api_base])
    if hasattr(args, 'api_model') and args.api_model:
        cmd.extend(["--api-model", args.api_model])
    if hasattr(args, 'skip_generation') and args.skip_generation:
        cmd.extend(["--skip-generation"])
    if hasattr(args, 'skip_evaluation') and args.skip_evaluation:
        cmd.extend(["--skip-evaluation"])
    
    # Handle dry run
    if hasattr(args, 'dry_run') and args.dry_run:
        if ENHANCED_CLI:
            cli.print_info("Dry run mode - showing command that would be executed:")
            cli.console.print(f"  {' '.join(cmd)}", style="cyan")
            return 0
        else:
            print(f"Would execute: {' '.join(cmd)}")
            return 0
    
    # Show metaeval information
    if ENHANCED_CLI:
        mode_descriptions = {
            'full': 'Complete metadata pipeline (generate + evaluate)',
            'generate_only': 'Generate metadata only',
            'evaluate_only': 'Evaluate existing metadata only'
        }
        mode_desc = mode_descriptions.get(args.mode, 'full pipeline')
        
        cli.print_step(1, 2, f"Starting metadata evaluation ({args.mode})")
        cli.print_info(f"Running: {mode_desc}")
        
        # Show configuration summary
        config_summary = {
            'Mode': f"{args.mode} - {mode_desc}",
            'Config File': args.config,
            'Input File': args.input_file or 'Default/Not specified',
            'Output Directory': args.output_dir or 'Default/From config',
            'Skip Generation': 'Yes' if args.skip_generation else 'No',
            'Skip Evaluation': 'Yes' if args.skip_evaluation else 'No'
        }
        cli.print_config_summary(config_summary)
        
        # Provide mode-specific information
        if args.mode == 'full':
            cli.print_info("This will generate metadata for datasets and evaluate their quality")
            cli.print_warning("Full pipeline may take considerable time for large datasets")
        elif args.mode == 'generate_only':
            cli.print_info("This will only generate metadata without evaluation")
        elif args.mode == 'evaluate_only':
            cli.print_info("This will evaluate existing metadata files")
        
        # Ask for confirmation for full mode
        if args.mode == 'full' and not args.skip_generation and not args.skip_evaluation:
            if not cli.confirm_action("This will run the complete metadata pipeline. Continue?"):
                cli.print_warning("Operation cancelled by user")
                return 1
    
    # Execute command
    result = cli_manager.run_command(
        cmd=cmd, 
        description=f"Metadata Pipeline ({args.mode})"
    )
    
    # Show results
    if ENHANCED_CLI:
        if result == 0:
            cli.print_success(f"Metadata pipeline ({args.mode}) completed successfully!")
            
            if args.mode in ['full', 'generate_only'] and not args.skip_generation:
                cli.print_info("Metadata files have been generated")
            if args.mode in ['full', 'evaluate_only'] and not args.skip_evaluation:
                cli.print_info("Metadata evaluation results available")
                cli.print_info("Check the output directory for detailed reports")
            
            if args.output_dir:
                cli.print_info(f"Results saved to: {args.output_dir}")
        else:
            cli.print_error(f"Metadata pipeline ({args.mode}) failed!")
            cli.print_info("Check the error output above for details")
    
    return result