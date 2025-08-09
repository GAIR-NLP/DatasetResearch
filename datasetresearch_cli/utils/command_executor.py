"""
Enhanced command execution with progress tracking
"""

import subprocess
import time
import threading
from pathlib import Path
from typing import List, Optional, Callable

from .cli_utils import cli


class CommandExecutor:
    """Enhanced command executor with progress tracking and better error handling"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.current_process = None
        self.output_buffer = []
        self.error_buffer = []
    
    def execute_command(
        self, 
        cmd: List[str], 
        description: str,
        show_progress: bool = True,
        timeout: Optional[int] = None,
        cwd: Optional[Path] = None
    ) -> int:
        """Execute command with enhanced UI and error handling"""
        
        if cwd is None:
            cwd = self.project_root
        
        # Display command info
        cli.print_command_info(
            command=description,
            description=f"Executing {description.lower()}",
            running_cmd=" ".join(cmd)
        )
        
        # Confirm for potentially dangerous operations
        if self._is_potentially_dangerous(cmd):
            if not cli.confirm_action(f"This will execute: {' '.join(cmd)}. Continue?"):
                cli.print_warning("Operation cancelled by user")
                return 1
        
        # Execute with progress tracking
        if show_progress:
            return self._execute_with_progress(cmd, description, cwd, timeout)
        else:
            return self._execute_simple(cmd, cwd, timeout)
    
    def _execute_with_progress(self, cmd: List[str], description: str, cwd: Path, timeout: Optional[int]) -> int:
        """Execute command with progress bar"""
        
        progress = cli.create_progress_bar()
        task_id = progress.add_task(description, total=None)
        
        start_time = time.time()
        
        try:
            with progress:
                # Start the process
                process = subprocess.Popen(
                    cmd,
                    cwd=cwd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                self.current_process = process
                
                # Create threads to read output
                def read_output():
                    for line in iter(process.stdout.readline, ''):
                        self.output_buffer.append(line.strip())
                        if len(self.output_buffer) > 1000:  # Limit buffer size
                            self.output_buffer.pop(0)
                
                def read_error():
                    for line in iter(process.stderr.readline, ''):
                        self.error_buffer.append(line.strip())
                        if len(self.error_buffer) > 1000:  # Limit buffer size
                            self.error_buffer.pop(0)
                
                # Start output reading threads
                output_thread = threading.Thread(target=read_output, daemon=True)
                error_thread = threading.Thread(target=read_error, daemon=True)
                output_thread.start()
                error_thread.start()
                
                # Update progress while process runs
                while process.poll() is None:
                    elapsed = time.time() - start_time
                    progress.update(task_id, description=f"{description} (elapsed: {elapsed:.1f}s)")
                    time.sleep(0.1)
                    
                    # Check timeout
                    if timeout and elapsed > timeout:
                        process.terminate()
                        cli.print_error(f"Command timed out after {timeout} seconds")
                        return 1
                
                # Wait for the process to complete
                return_code = process.wait()
                
                # Wait for output threads to finish
                output_thread.join(timeout=1)
                error_thread.join(timeout=1)
                
        except KeyboardInterrupt:
            cli.print_warning("Command interrupted by user")
            if self.current_process:
                self.current_process.terminate()
            return 130
        except Exception as e:
            cli.print_error(f"Error executing command: {e}")
            return 1
        finally:
            self.current_process = None
        
        # Display results
        if return_code == 0:
            cli.print_success(f"{description} completed successfully")
            # Show last few lines of output if available
            if self.output_buffer:
                recent_output = self.output_buffer[-3:]  # Show last 3 lines
                cli.print_info("Recent output:")
                for line in recent_output:
                    if line.strip():
                        cli.console.print(f"  {line}", style="dim")
        else:
            cli.print_error(f"{description} failed (exit code: {return_code})")
            # Show error output
            if self.error_buffer:
                cli.print_error("Error output:")
                for line in self.error_buffer[-5:]:  # Show last 5 error lines
                    if line.strip():
                        cli.console.print(f"  {line}", style="red")
        
        return return_code
    
    def _execute_simple(self, cmd: List[str], cwd: Path, timeout: Optional[int]) -> int:
        """Execute command without progress tracking"""
        try:
            result = subprocess.run(
                cmd,
                cwd=cwd,
                check=False,
                timeout=timeout,
                capture_output=True,
                text=True
            )
            
            if result.stdout:
                self.output_buffer.extend(result.stdout.split('\n'))
            if result.stderr:
                self.error_buffer.extend(result.stderr.split('\n'))
            
            return result.returncode
            
        except subprocess.TimeoutExpired:
            cli.print_error(f"Command timed out after {timeout} seconds")
            return 1
        except Exception as e:
            cli.print_error(f"Error executing command: {e}")
            return 1
    
    def _is_potentially_dangerous(self, cmd: List[str]) -> bool:
        """Check if command might be dangerous and needs confirmation"""
        dangerous_patterns = [
            'rm', 'del', 'format', 'rmdir',
            '--force', '--delete', '--remove'
        ]
        
        cmd_str = ' '.join(cmd).lower()
        return any(pattern in cmd_str for pattern in dangerous_patterns)
    
    def get_recent_output(self, lines: int = 10) -> List[str]:
        """Get recent output lines"""
        return self.output_buffer[-lines:] if self.output_buffer else []
    
    def get_recent_errors(self, lines: int = 10) -> List[str]:
        """Get recent error lines"""
        return self.error_buffer[-lines:] if self.error_buffer else []