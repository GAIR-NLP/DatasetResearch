"""
Run command for dataset research CLI
"""

import sys

try:
    from datasetresearch_cli.utils.cli_utils import cli
    ENHANCED_CLI = True
except ImportError:
    ENHANCED_CLI = False


def add_run_parser(subparsers):
    """Add run command parser"""
    run_parser = subparsers.add_parser(
        'run', 
        help='Execute SFT and inference (full pipeline)',
        description='Execute complete training pipeline including SFT and inference'
    )
    # Based on actual evaluation_framework.py parameters
    run_parser.add_argument('--config', required=True, help='Configuration file path')
    run_parser.add_argument('--dataset_json', help='Dataset metadata JSON file path')
    run_parser.add_argument('--model', default='llama3_8b', help='Model name to use for training')
    run_parser.add_argument('--step', choices=['train', 'inference', 'eval', 'full', 'batch'], 
                           default='full', help='Pipeline step to run')
    run_parser.add_argument('--task_model', default='gpt-4o-search', help='Task model name')
    run_parser.add_argument('--baseline', action='store_true', help='Run baseline evaluation (skip training)')
    run_parser.add_argument('--test', action='store_true', help='Run test set evaluation')
    run_parser.add_argument('--dry-run', action='store_true', help='Show what would be executed without running')
    return run_parser


def handle_run(cli_manager, args):
    """Execute SFT and inference using evaluation_framework.py"""
    
    # Validate configuration - required for evaluation_framework.py
    if not args.config:
        if ENHANCED_CLI:
            cli.print_error("--config is required for the run command")
        else:
            print("Error: --config is required")
        return 1
    
    if not cli_manager.validate_config_if_provided(args.config):
        return 1
    
    # Build command with required config parameter
    cmd = [sys.executable, "evaluation/evaluation_framework.py", "--config", args.config]
    
    # Add other arguments
    if hasattr(args, 'dataset_json') and args.dataset_json:
        cmd.extend(["--dataset_json", args.dataset_json])
    if hasattr(args, 'model') and args.model:
        cmd.extend(["--model", args.model])
    if hasattr(args, 'step') and args.step:
        cmd.extend(["--step", args.step])
    if hasattr(args, 'task_model') and args.task_model:
        cmd.extend(["--task_model", args.task_model])
    if hasattr(args, 'baseline') and args.baseline:
        cmd.extend(["--baseline"])
    if hasattr(args, 'test') and args.test:
        cmd.extend(["--test"])
    
    # Handle dry run
    if hasattr(args, 'dry_run') and args.dry_run:
        if ENHANCED_CLI:
            cli.print_info("Dry run mode - showing command that would be executed:")
            cli.console.print(f"  {' '.join(cmd)}", style="cyan")
            return 0
        else:
            print(f"Would execute: {' '.join(cmd)}")
            return 0
    
    # Show run information
    if ENHANCED_CLI:
        run_type = "Baseline Evaluation" if args.baseline else "Full Training Pipeline"
        if args.test:
            run_type = "Test Set Evaluation"
        
        cli.print_step(1, 3, f"Starting {run_type}")
        cli.print_info("This process includes SFT training and/or inference...")
        if not args.baseline and not args.test:
            cli.print_warning("This may take several hours depending on dataset size and model complexity")
        
        # Show configuration summary
        config_summary = {
            'Run Type': run_type,
            'Config File': args.config,
            'Model': args.model,
            'Step': args.step,
            'Task Model': args.task_model,
            'Dataset JSON': args.dataset_json or 'Not specified'
        }
        cli.print_config_summary(config_summary)
        
        # Ask for confirmation for full training
        if args.step == 'full' and not args.baseline and not args.test:
            if not cli.confirm_action("This will start a full training pipeline. Continue?"):
                cli.print_warning("Operation cancelled by user")
                return 1
    
    # Execute command
    result = cli_manager.run_command(
        cmd=cmd, 
        description=f"Training Pipeline ({args.step})"
    )
    
    # Show results
    if ENHANCED_CLI:
        if result == 0:
            cli.print_success(f"{run_type} completed successfully!")
            cli.print_info("Training logs and model outputs available in the workspace")
            if not args.baseline and not args.test:
                cli.print_info("Trained model saved and ready for inference")
        else:
            cli.print_error(f"{run_type} failed!")
            cli.print_info("Check the error output above for details")
            cli.print_info("You may need to adjust model parameters or check system resources")
    
    return result