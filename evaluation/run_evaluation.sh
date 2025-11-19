#!/bin/bash

# =============================================================================
# Automated Evaluation Workflow: Custom Method vs DiffEdit
# =============================================================================
# This script automates the entire evaluation process
#
# Usage:
#   ./run_evaluation.sh /path/to/test_images
#
# Prerequisites:
#   - Test images in specified directory
#   - prompts_custom.json and prompts_diffedit.json in current directory
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default paths
TEST_IMAGES_DIR=${1:-"test_data"}
OUTPUT_BASE="outputs"
CUSTOM_OUTPUT="${OUTPUT_BASE}/custom_method"
DIFFEDIT_OUTPUT="${OUTPUT_BASE}/diffedit"
RESULTS_FILE="comparison_results.json"

# Prompt files
PROMPTS_CUSTOM="prompts_custom.json"
PROMPTS_DIFFEDIT="prompts_diffedit.json"
PROMPTS_CLIP="prompts_for_clip.json"

# Configuration
DEVICE="cuda"  # Change to "cpu" if no GPU available
IMAGE_SIZE="768 768"

# =============================================================================
# Helper Functions
# =============================================================================

print_header() {
    echo -e "\n${BLUE}============================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}============================================${NC}\n"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

check_file() {
    if [ ! -f "$1" ]; then
        print_error "File not found: $1"
        return 1
    fi
    return 0
}

check_dir() {
    if [ ! -d "$1" ]; then
        print_error "Directory not found: $1"
        return 1
    fi
    return 0
}

# =============================================================================
# Pre-flight Checks
# =============================================================================

print_header "Pre-flight Checks"

# Check test images directory
if ! check_dir "$TEST_IMAGES_DIR"; then
    print_error "Test images directory not found: $TEST_IMAGES_DIR"
    print_info "Usage: ./run_evaluation.sh /path/to/test_images"
    exit 1
fi

# Count images
IMAGE_COUNT=$(find "$TEST_IMAGES_DIR" -type f \( -iname "*.jpg" -o -iname "*.jpeg" -o -iname "*.png" \) | wc -l)
print_success "Found $IMAGE_COUNT test images in $TEST_IMAGES_DIR"

# Check for prompt files
if ! check_file "$PROMPTS_CUSTOM"; then
    print_warning "Custom prompts file not found: $PROMPTS_CUSTOM"
    print_info "Creating sample file..."
    python run_custom_batch.py --create_sample_prompts
    print_error "Please edit $PROMPTS_CUSTOM with your actual prompts and re-run"
    exit 1
fi

if ! check_file "$PROMPTS_DIFFEDIT"; then
    print_warning "DiffEdit prompts file not found: $PROMPTS_DIFFEDIT"
    print_info "Creating sample file..."
    python run_diffedit_batch.py --create_sample_prompts
    print_error "Please edit $PROMPTS_DIFFEDIT with your actual prompts and re-run"
    exit 1
fi

print_success "All prompt files found"

# Check Python scripts
REQUIRED_SCRIPTS=("run_custom_batch.py" "run_diffedit_batch.py" "compare_methods.py")
for script in "${REQUIRED_SCRIPTS[@]}"; do
    if ! check_file "$script"; then
        print_error "Required script not found: $script"
        exit 1
    fi
done
print_success "All required scripts found"

# Create output directories
mkdir -p "$CUSTOM_OUTPUT"
mkdir -p "$DIFFEDIT_OUTPUT"
print_success "Output directories created"

# =============================================================================
# Step 1: Run Custom Method
# =============================================================================

print_header "Step 1: Running Custom Method"
print_info "This will process $IMAGE_COUNT images through the 4-stage pipeline"
print_info "Expected time: ~30-60 seconds per image on GPU"

START_TIME=$(date +%s)

python run_custom_batch.py \
    --input_dir "$TEST_IMAGES_DIR" \
    --output_dir "$CUSTOM_OUTPUT" \
    --prompts "$PROMPTS_CUSTOM"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
print_success "Custom method completed in ${ELAPSED}s"

# Check outputs
CUSTOM_FINAL_COUNT=$(find "${CUSTOM_OUTPUT}/final_images" -type f 2>/dev/null | wc -l || echo "0")
print_info "Generated $CUSTOM_FINAL_COUNT final images"

# =============================================================================
# Step 2: Run DiffEdit
# =============================================================================

print_header "Step 2: Running DiffEdit"
print_info "This will process $IMAGE_COUNT images with DiffEdit"
print_info "Expected time: ~60-120 seconds per image on GPU (includes inversion)"

START_TIME=$(date +%s)

python run_diffedit_batch.py \
    --input_dir "$TEST_IMAGES_DIR" \
    --output_dir "$DIFFEDIT_OUTPUT" \
    --prompts "$PROMPTS_DIFFEDIT" \
    --device "$DEVICE" \
    --save_masks \
    --image_size $IMAGE_SIZE

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
print_success "DiffEdit completed in ${ELAPSED}s"

# Check outputs
DIFFEDIT_COUNT=$(find "$DIFFEDIT_OUTPUT" -name "edited_*.jpg" -o -name "edited_*.png" 2>/dev/null | wc -l || echo "0")
print_info "Generated $DIFFEDIT_COUNT edited images"

# =============================================================================
# Step 3: Run Evaluation & Comparison
# =============================================================================

print_header "Step 3: Running Evaluation"
print_info "Computing FID, Inception Score, IoU, LPIPS, and CLIP scores"
print_info "This may take 5-10 minutes depending on dataset size"

START_TIME=$(date +%s)

# Check if CLIP prompts file exists, use DiffEdit target prompts as fallback
if [ ! -f "$PROMPTS_CLIP" ]; then
    print_warning "CLIP prompts file not found: $PROMPTS_CLIP"
    print_info "Using DiffEdit target prompts for CLIP evaluation"
    CLIP_ARG=""
else
    CLIP_ARG="--prompts $PROMPTS_CLIP"
fi

python compare_methods.py \
    --original_images "$TEST_IMAGES_DIR" \
    --custom_outputs "${CUSTOM_OUTPUT}/final_images" \
    --diffedit_outputs "$DIFFEDIT_OUTPUT" \
    --custom_masks "${CUSTOM_OUTPUT}/masks" \
    --diffedit_masks "$DIFFEDIT_OUTPUT" \
    $CLIP_ARG \
    --output_json "$RESULTS_FILE" \
    --device "$DEVICE"

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
print_success "Evaluation completed in ${ELAPSED}s"

# =============================================================================
# Step 4: Display Results
# =============================================================================

print_header "Step 4: Results Summary"

if [ -f "$RESULTS_FILE" ]; then
    print_success "Detailed results saved to: $RESULTS_FILE"
    
    # Extract key metrics using jq if available
    if command -v jq &> /dev/null; then
        echo ""
        print_info "Key Metrics:"
        echo ""
        echo "Custom Method:"
        echo "  FID:              $(jq -r '.custom_method.fid' $RESULTS_FILE)"
        echo "  Inception Score:  $(jq -r '.custom_method.inception_score_mean' $RESULTS_FILE)"
        echo "  LPIPS:            $(jq -r '.custom_method.lpips' $RESULTS_FILE)"
        echo "  CLIP Score:       $(jq -r '.custom_method.clip' $RESULTS_FILE)"
        echo "  IoU:              $(jq -r '.custom_method.iou' $RESULTS_FILE)"
        echo ""
        echo "DiffEdit:"
        echo "  FID:              $(jq -r '.diffedit_method.fid' $RESULTS_FILE)"
        echo "  Inception Score:  $(jq -r '.diffedit_method.inception_score_mean' $RESULTS_FILE)"
        echo "  LPIPS:            $(jq -r '.diffedit_method.lpips' $RESULTS_FILE)"
        echo "  CLIP Score:       $(jq -r '.diffedit_method.clip' $RESULTS_FILE)"
        echo "  IoU:              $(jq -r '.diffedit_method.iou' $RESULTS_FILE)"
        echo ""
    fi
else
    print_error "Results file not found: $RESULTS_FILE"
fi

# =============================================================================
# Cleanup and Final Info
# =============================================================================

print_header "Evaluation Complete!"

print_success "All outputs saved to:"
print_info "  Custom method:   $CUSTOM_OUTPUT"
print_info "  DiffEdit:        $DIFFEDIT_OUTPUT"
print_info "  Results JSON:    $RESULTS_FILE"

print_info "\nTo view detailed results:"
print_info "  cat $RESULTS_FILE | jq ."
print_info "\nTo view processing logs:"
print_info "  cat ${CUSTOM_OUTPUT}/processing_log.json"
print_info "  cat ${DIFFEDIT_OUTPUT}/processing_log.json"

echo ""
print_success "Evaluation workflow completed successfully! 🎉"
echo ""
