"""
Eval command for dataset research CLI
"""

import sys

try:
    from datasetresearch_cli.utils.cli_utils import cli
    ENHANCED_CLI = True
except ImportError:
    ENHANCED_CLI = False


def add_eval_parser(subparsers):
    """Add eval command parser"""
    eval_parser = subparsers.add_parser(
        'eval', 
        help='Run evaluation pipeline',
        description='Run comprehensive evaluation pipeline with optional mode selection'
    )
    # Based on actual integrated_evaluation_pipeline.py parameters
    eval_parser.add_argument(
        '--eval_config', 
        default='evaluation/config.yaml',
        help='Evaluation config file path (default: evaluation/config.yaml)'
    )
    eval_parser.add_argument(
        '--pipeline_config', 
        default='configs/pipeline_settings.yaml',
        help='Pipeline config file path (default: configs/pipeline_settings.yaml)'
    )
    eval_parser.add_argument(
        '--base_dir', 
        help='Project base directory'
    )
    eval_parser.add_argument(
        '--skip_eval', 
        action='store_true',
        help='Skip evaluation step'
    )
    eval_parser.add_argument(
        '--skip_process', 
        action='store_true',
        help='Skip result processing step'
    )
    eval_parser.add_argument(
        '--list_info', 
        action='store_true',
        help='Only show related information'
    )
    eval_parser.add_argument(
        '--set', 
        choices=['full', 'mini'], 
        default='full',
        help='Dataset set to use (default: full)'
    )
    eval_parser.add_argument(
        '--dry-run', 
        action='store_true',
        help='Show what would be executed without running'
    )
    return eval_parser


def handle_eval(cli_manager, args):
    """Execute evaluation pipeline with options"""
    
    # Validate configurations
    configs_to_check = [args.eval_config, args.pipeline_config]
    for config in configs_to_check:
        if config and not cli_manager.validate_config_if_provided(config):
            return 1
    
    # Build command
    cmd = [sys.executable, "scripts/integrated_evaluation_pipeline.py"]
    
    # Add arguments based on actual script parameters
    if hasattr(args, 'eval_config') and args.eval_config:
        cmd.extend(["--eval_config", args.eval_config])
    if hasattr(args, 'pipeline_config') and args.pipeline_config:
        cmd.extend(["--pipeline_config", args.pipeline_config])
    if hasattr(args, 'base_dir') and args.base_dir:
        cmd.extend(["--base_dir", args.base_dir])
    if hasattr(args, 'skip_eval') and args.skip_eval:
        cmd.extend(["--skip_eval"])
    if hasattr(args, 'skip_process') and args.skip_process:
        cmd.extend(["--skip_process"])
    if hasattr(args, 'list_info') and args.list_info:
        cmd.extend(["--list_info"])
    if hasattr(args, 'set') and args.set:
        cmd.extend(["--set", args.set])
    
    # Handle dry run
    if hasattr(args, 'dry_run') and args.dry_run:
        if ENHANCED_CLI:
            cli.print_info("Dry run mode - showing command that would be executed:")
            cli.console.print(f"  {' '.join(cmd)}", style="cyan")
            return 0
        else:
            print(f"Would execute: {' '.join(cmd)}")
            return 0
    
    # Handle list_info mode
    if hasattr(args, 'list_info') and args.list_info:
        if ENHANCED_CLI:
            cli.print_info("Showing evaluation pipeline information only...")
        # Execute command to show info and return
        return cli_manager.run_command(
            cmd=cmd, 
            description="Pipeline Information"
        )
    
    # Show evaluation information
    if ENHANCED_CLI:
        mode_desc = "evaluation pipeline"
        if args.skip_eval:
            mode_desc = "result processing only"
        elif args.skip_process:
            mode_desc = "evaluation only"
        
        cli.print_step(1, 3, f"Starting {mode_desc}")
        cli.print_info(f"Dataset set: {args.set}")
        cli.print_info("This may take significant time for large datasets...")
        
        # Show configuration summary
        config_summary = {
            'Mode': mode_desc,
            'Dataset Set': args.set,
            'Eval Config': args.eval_config,
            'Pipeline Config': args.pipeline_config,
            'Base Directory': args.base_dir or 'Default'
        }
        cli.print_config_summary(config_summary)
        
        # Ask for confirmation for full runs
        if not args.skip_eval and not args.skip_process and args.set == 'full':
            if not cli.confirm_action("This will run the complete evaluation pipeline. Continue?"):
                cli.print_warning("Operation cancelled by user")
                return 1
    
    # Execute command
    result = cli_manager.run_command(
        cmd=cmd, 
        description=f"Evaluation Pipeline ({mode_desc if ENHANCED_CLI else 'eval'})"
    )
    
    # Show results
    if ENHANCED_CLI:
        if result == 0:
            cli.print_success(f"{mode_desc} completed successfully!")
            cli.print_info("Check evaluation/results/ for output files")
            if not args.skip_process:
                cli.print_info("Results summary available in evaluation_summary.csv")
        else:
            cli.print_error(f"{mode_desc} failed!")
            cli.print_info("Check the error output above for details")
    
    return result