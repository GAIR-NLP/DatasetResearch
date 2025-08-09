#!/bin/bash
set -e

echo "🚀 Deep Dataset Research Setup Script"
echo "===================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_step() {
    echo -e "\n${BLUE}[STEP]${NC} $1"
}

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

print_step "Checking system requirements..."

# Check if we're on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    SYSTEM="macOS"
    print_status "Detected macOS system"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    SYSTEM="Linux"
    print_status "Detected Linux system"
else
    print_error "Unsupported operating system: $OSTYPE"
    exit 1
fi

# Function to install uv
install_uv() {
    print_status "Installing uv package manager..."
    if command -v curl &> /dev/null; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        # Add uv to PATH for current session
        export PATH="$HOME/.cargo/bin:$PATH"
        # Source the shell profile to make uv available
        if [[ -f "$HOME/.bashrc" ]]; then
            source "$HOME/.bashrc"
        fi
        if [[ -f "$HOME/.zshrc" ]]; then
            source "$HOME/.zshrc"
        fi
    else
        print_error "curl is required to install uv. Please install curl first."
        exit 1
    fi
}

# Check if uv is installed
print_step "Checking uv installation..."
if ! command -v uv &> /dev/null; then
    print_warning "uv not found, installing..."
    install_uv
    
    # Check again after installation
    if ! command -v uv &> /dev/null; then
        print_error "Failed to install uv. Please install manually from https://github.com/astral-sh/uv"
        exit 1
    fi
else
    print_status "uv is already installed"
fi

# Navigate to project directory
cd "$PROJECT_ROOT"

# Install dependencies using uv
print_step "Installing Python dependencies with uv..."
print_status "Running: uv sync"

# Handle potential CUDA dependency issues on non-CUDA systems
if uv sync; then
    print_status "Dependencies installed successfully"
else
    print_warning "uv sync failed, trying with --resolution lowest-direct..."
    if uv sync --resolution lowest-direct; then
        print_status "Dependencies installed with resolution strategy"
    else
        print_error "Failed to install dependencies. You may need to manually resolve dependency conflicts."
        print_error "Common issues: CUDA dependencies on non-CUDA systems"
        exit 1
    fi
fi

# Activate the virtual environment created by uv
print_step "Activating virtual environment..."
if [[ -f ".venv/bin/activate" ]]; then
    source .venv/bin/activate
    print_status "Virtual environment activated"
elif [[ -f ".venv/Scripts/activate" ]]; then
    # Windows support
    source .venv/Scripts/activate
    print_status "Virtual environment activated (Windows)"
else
    print_warning "Virtual environment activation script not found"
    print_warning "Continuing without explicit activation (uv should handle this)"
fi

# Wait a moment to let system recover memory
print_status "Waiting for system to stabilize..."
sleep 2


# Install LLaMA-Factory after uv sync is complete
print_step "Setting up LLaMA-Factory..."
LLAMAFACTORY_DIR="$PROJECT_ROOT/LLaMA-Factory"

if [[ -d "$LLAMAFACTORY_DIR" ]]; then
    print_status "LLaMA-Factory directory found"
    cd "$LLAMAFACTORY_DIR"
    
    # Install LLaMA-Factory with torch and metrics using uv
    print_status "Installing LLaMA-Factory dependencies (this may take a while)..."
    
    if uv pip install -e ".[torch,metrics]"; then
        print_status "LLaMA-Factory installed successfully"
    else
        print_error "Failed to install LLaMA-Factory. Please install manually:"
        print_error "  cd LLaMA-Factory && uv pip install -e \".[torch,metrics]\""
        print_warning "Continuing setup without LLaMA-Factory installation..."
    fi
    
    cd "$PROJECT_ROOT"
else
    print_error "LLaMA-Factory directory not found at $LLAMAFACTORY_DIR"
    print_error "Please ensure LLaMA-Factory is properly included in the repository"
    print_warning "Continuing setup without LLaMA-Factory installation..."
fi

# Create necessary directories
print_step "Creating directory structure..."
mkdir -p LLaMA-Factory/data
mkdir -p datasets/results

# Download dataset from Hugging Face
print_step "Downloading dataset from Hugging Face..."
DATASET_URL="https://huggingface.co/datasets/GAIR/DatasetResearch"
DOWNLOAD_DIR="LLaMA-Factory/data"
TARGET_DIR="LLaMA-Factory/data/test_dataset"
TEMP_DIR="LLaMA-Factory/data/DatasetResearch"

if [[ -d "$TARGET_DIR" ]]; then
    print_warning "Dataset directory $TARGET_DIR already exists. Updating..."
    cd "$TARGET_DIR"
    git pull
    cd "$PROJECT_ROOT"
else
    print_status "Cloning dataset from $DATASET_URL..."
    cd "$DOWNLOAD_DIR"
    
    # Clone the repository (will create DatasetResearch folder)
    if git clone "$DATASET_URL"; then
        print_status "Dataset cloned successfully"
        
        # Rename the downloaded folder to test_dataset
        if [[ -d "DatasetResearch" ]]; then
            if [[ -d "test_dataset" ]]; then
                print_warning "Removing existing test_dataset directory..."
                rm -rf "test_dataset"
            fi
            mv "DatasetResearch" "test_dataset"
            print_status "Renamed dataset folder to test_dataset"
        else
            print_error "Downloaded dataset folder 'DatasetResearch' not found"
            cd "$PROJECT_ROOT"
            exit 1
        fi
    else
        print_error "Failed to clone dataset from $DATASET_URL"
        cd "$PROJECT_ROOT"
        exit 1
    fi
    
    cd "$PROJECT_ROOT"
fi

# Check if dataset was downloaded and renamed successfully
if [[ ! -d "$TARGET_DIR" ]]; then
    print_error "Failed to download dataset or rename to test_dataset"
    exit 1
fi

print_status "Dataset downloaded to $TARGET_DIR"

# Copy metadata and result files to appropriate directories
print_step "Copying dataset files..."

# Create evaluation/results directory if it doesn't exist
mkdir -p evaluation/results

# Copy test_metadata.json to datasets/results/
METADATA_SOURCE="$TARGET_DIR/test_metadata.json"
METADATA_TARGET="datasets/results/test_metadata.json"

if [[ -f "$METADATA_SOURCE" ]]; then
    cp "$METADATA_SOURCE" "$METADATA_TARGET"
    print_status "Copied test_metadata.json to datasets/results/"
else
    print_warning "test_metadata.json not found in dataset."
fi

# Copy final_test.json to evaluation/results/
FINAL_TEST_SOURCE="$TARGET_DIR/final_test.json"
FINAL_TEST_TARGET="evaluation/results/final_test.json"

if [[ -f "$FINAL_TEST_SOURCE" ]]; then
    cp "$FINAL_TEST_SOURCE" "$FINAL_TEST_TARGET"
    print_status "Copied final_test.json to evaluation/results/"
else
    print_warning "final_test.json not found in dataset."
fi

# Copy final_baseline.json to evaluation/results/
FINAL_BASELINE_SOURCE="$TARGET_DIR/final_baseline.json"
FINAL_BASELINE_TARGET="evaluation/results/final_baseline.json"

if [[ -f "$FINAL_BASELINE_SOURCE" ]]; then
    cp "$FINAL_BASELINE_SOURCE" "$FINAL_BASELINE_TARGET"
    print_status "Copied final_baseline.json to evaluation/results/"
else
    print_warning "final_baseline.json not found in dataset."
fi

# List available files for debugging if any file was missing
if [[ ! -f "$METADATA_SOURCE" ]] || [[ ! -f "$FINAL_TEST_SOURCE" ]] || [[ ! -f "$FINAL_BASELINE_SOURCE" ]]; then
    print_status "Available files in dataset:"
    ls -la "$TARGET_DIR" | head -10
fi

# Make datasetresearch script executable
print_step "Setting up datasetresearch command..."
DATASETRESEARCH_SCRIPT="$PROJECT_ROOT/datasetresearch"

if [[ -f "$DATASETRESEARCH_SCRIPT" ]]; then
    chmod +x "$DATASETRESEARCH_SCRIPT"
    print_status "Made datasetresearch script executable"
else
    print_error "datasetresearch script not found at expected location"
    exit 1
fi

# Add to PATH for current session and future sessions
print_step "Setting up PATH environment..."

# Add alias for current session
alias datasetresearch="$PROJECT_ROOT/datasetresearch"

# Function to add alias to shell profile
add_to_profile() {
    local profile_file="$1"
    local alias_line="alias datasetresearch='$PROJECT_ROOT/datasetresearch'"
    
    if [[ -f "$profile_file" ]]; then
        # Check if already added
        if ! grep -Fq "alias datasetresearch=" "$profile_file"; then
            echo "" >> "$profile_file"
            echo "# Deep Dataset Research CLI" >> "$profile_file"
            echo "$alias_line" >> "$profile_file"
            print_status "Added alias to $profile_file"
        else
            print_status "Alias already configured in $profile_file"
        fi
    fi
}

# Determine shell and add to appropriate profile
if [[ -n "$ZSH_VERSION" ]] || [[ "$SHELL" == *"zsh"* ]]; then
    add_to_profile "$HOME/.zshrc"
    # Reload zshrc for immediate effect
    if [[ -f "$HOME/.zshrc" ]]; then
        source "$HOME/.zshrc" 2>/dev/null || true
    fi
elif [[ -n "$BASH_VERSION" ]] || [[ "$SHELL" == *"bash"* ]]; then
    add_to_profile "$HOME/.bashrc"
    # Also add to .bash_profile on macOS
    if [[ "$SYSTEM" == "macOS" ]]; then
        add_to_profile "$HOME/.bash_profile"
        # Source .bash_profile on macOS
        if [[ -f "$HOME/.bash_profile" ]]; then
            source "$HOME/.bash_profile" 2>/dev/null || true
        fi
    fi
    # Reload bashrc for immediate effect
    if [[ -f "$HOME/.bashrc" ]]; then
        source "$HOME/.bashrc" 2>/dev/null || true
    fi
else
    print_warning "Unknown shell. Please manually add $PROJECT_ROOT to your PATH"
fi

# Update PATH for current session to ensure immediate availability
export PATH="$PROJECT_ROOT:$PATH"

# Test the installation
print_step "Testing installation..."

# Test the CLI directly first
print_status "Testing CLI functionality..."
if "$DATASETRESEARCH_SCRIPT" --help > /dev/null 2>&1; then
    print_status "CLI is working correctly with direct path"
else
    print_warning "CLI test failed. You may need to check dependencies or python environment."
fi

# Test if datasetresearch command works from PATH
print_status "Testing global command availability..."
if command -v datasetresearch &> /dev/null && datasetresearch --help > /dev/null 2>&1; then
    print_status "✅ datasetresearch command is globally available"
    print_status "You can now use 'datasetresearch' from anywhere!"
else
    print_warning "datasetresearch command not immediately available in new sessions."
    print_status "Current session: you can use ./datasetresearch"
    print_status "New sessions: restart your terminal or run:"
    echo "    export PATH=\"$PROJECT_ROOT:\$PATH\""
fi


print_step "Setup completed successfully! 🎉"
echo ""
print_status "Ready to use! You can now:"

# Check if global command is working and provide appropriate instructions
if command -v datasetresearch &> /dev/null && datasetresearch --help > /dev/null 2>&1; then
    echo "  ✅ datasetresearch --help          # Use from anywhere!"
    echo "  ✅ datasetresearch search          # Start dataset search"  
    echo "  ✅ datasetresearch synthesis       # Generate synthetic data"
    echo "  ✅ datasetresearch register        # Register datasets"
else
    echo "  📋 ./datasetresearch --help        # Use from project directory"
    echo "  📋 ./datasetresearch search        # Start dataset search"
    echo "  📋 ./datasetresearch synthesis     # Generate synthetic data"  
    echo "  📋 ./datasetresearch register      # Register datasets"
    echo ""
    echo "  💡 For global access, try running:"
    echo "     source ~/.bashrc  # or source ~/.zshrc for zsh users"
    echo "     hash -r"
fi

echo ""
print_status "Resources:"
print_status "  📊 Dataset: LLaMA-Factory/data/test_dataset"
print_status "  📋 Metadata: datasets/results/test_metadata.json"
print_status "  🎯 Test Results: evaluation/results/final_test.json"
print_status "  📊 Baseline Results: evaluation/results/final_baseline.json"
print_status "  🔧 Help: datasetresearch --help (or ./datasetresearch --help)"
echo ""
print_status "Happy researching! 🔬"