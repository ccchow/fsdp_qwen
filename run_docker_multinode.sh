#!/bin/bash
#
# Advanced Multi-Node Multi-GPU Docker Orchestration Script
#
# This script orchestrates the complete setup and execution of multi-node
# multi-GPU distributed training using Docker containers.
#

set -e

# Configuration
IMAGE_NAME="diloco-enhanced"
NETWORK_NAME="diloco-net"
CONTAINER_PREFIX="diloco-node"
WORLD_SIZE=2
GPUS_PER_NODE=2
MASTER_PORT=29500

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to clean up resources
cleanup() {
    print_status "Cleaning up resources..."
    
    # Stop and remove containers
    for ((node=0; node<WORLD_SIZE; node++)); do
        container_name="${CONTAINER_PREFIX}${node}"
        if docker ps -q -f name=$container_name | grep -q .; then
            print_status "Stopping container: $container_name"
            docker stop $container_name >/dev/null 2>&1 || true
        fi
        
        if docker ps -aq -f name=$container_name | grep -q .; then
            print_status "Removing container: $container_name"
            docker rm $container_name >/dev/null 2>&1 || true
        fi
    done
    
    # Remove network
    if docker network ls -q -f name=$NETWORK_NAME | grep -q .; then
        print_status "Removing network: $NETWORK_NAME"
        docker network rm $NETWORK_NAME >/dev/null 2>&1 || true
    fi
}

# Function to build Docker image
build_image() {
    print_status "Building Docker image: $IMAGE_NAME"
    
    if ! docker build -f Dockerfile.diloco -t $IMAGE_NAME .; then
        print_error "Failed to build Docker image"
        exit 1
    fi
    
    print_success "Docker image built successfully"
}

# Function to setup Docker network
setup_network() {
    print_status "Setting up Docker network: $NETWORK_NAME"
    
    # Remove existing network if it exists
    if docker network ls -q -f name=$NETWORK_NAME | grep -q .; then
        docker network rm $NETWORK_NAME >/dev/null 2>&1 || true
    fi
    
    # Create new network
    if ! docker network create $NETWORK_NAME; then
        print_error "Failed to create Docker network"
        exit 1
    fi
    
    print_success "Docker network created successfully"
}

# Function to run multi-node training
run_training() {
    local test_mode=${1:-"outer_optimizer"}
    local steps=${2:-"5"}
    local seq_len=${3:-"32"}
    local batch_size=${4:-"1"}
    local diloco_loops=${5:-"3"}
    local dataset_name=${6:-"HuggingFaceFW/fineweb-edu"}
    local dataset_subset=${7:-"sample-10BT"}
    local use_streaming=${8:-"true"}
    
    print_status "Starting multi-node multi-GPU training"
    print_status "Configuration:"
    print_status "  - Nodes: $WORLD_SIZE"
    print_status "  - GPUs per node: $GPUS_PER_NODE"
    print_status "  - Total ranks: $((WORLD_SIZE * GPUS_PER_NODE))"
    print_status "  - Test mode: $test_mode"
    print_status "  - Steps: $steps"
    print_status "  - Sequence length: $seq_len"
    print_status "  - Batch size: $batch_size"
    print_status "  - DiLoCo loops: $diloco_loops"
    print_status "  - Dataset: $dataset_name ($dataset_subset)"
    print_status "  - Streaming: $use_streaming"
    
    # Create shared results directory
    mkdir -p ./docker_results
    
    # Launch containers
    container_pids=()
    
    for ((node=0; node<WORLD_SIZE; node++)); do
        container_name="${CONTAINER_PREFIX}${node}"
        
        print_status "Launching container: $container_name (Node $node)"
        
        # Calculate rank range for this node
        start_rank=$((node * GPUS_PER_NODE))
        
        # Prepare dataset arguments
        dataset_args=""
        if [ "$use_streaming" = "true" ]; then
            dataset_args="--use_streaming --dataset_name $dataset_name --dataset_subset $dataset_subset"
        else
            dataset_args="--no_streaming"
        fi
        
        docker run \
            --name $container_name \
            --network $NETWORK_NAME \
            --gpus all \
            --rm \
            -v $(pwd)/docker_results:/workspace/results \
            -e WORLD_SIZE=$WORLD_SIZE \
            -e NODE_RANK=$node \
            -e GPUS_PER_NODE=$GPUS_PER_NODE \
            -e MASTER_ADDR=${CONTAINER_PREFIX}0 \
            -e MASTER_PORT=$MASTER_PORT \
            -e RANK=$start_rank \
            $IMAGE_NAME \
            --steps $steps \
            --seq_len $seq_len \
            --batch_size $batch_size \
            --diloco_loops $diloco_loops \
            --mixed_precision fp16 \
            --output_dir /workspace/results \
            $dataset_args &
        
        container_pids+=($!)
        
        # Small delay between container starts
        sleep 2
    done
    
    print_status "All containers launched. Waiting for completion..."
    
    # Wait for all containers to complete
    failed_containers=0
    for pid in "${container_pids[@]}"; do
        if ! wait $pid; then
            ((failed_containers++))
        fi
    done
    
    if [ $failed_containers -eq 0 ]; then
        print_success "All containers completed successfully!"
        return 0
    else
        print_error "$failed_containers containers failed"
        return 1
    fi
}

# Function to show results
show_results() {
    print_status "Multi-Node Training Results"
    echo "=" * 50
    
    if [ -f "./docker_results/multi_node_aggregated_report.md" ]; then
        cat ./docker_results/multi_node_aggregated_report.md
    else
        print_warning "Aggregated report not found"
    fi
    
    print_status "Individual node results:"
    for ((node=0; node<WORLD_SIZE; node++)); do
        result_dir="./docker_results/node_${node}"
        if [ -d "$result_dir" ]; then
            print_status "Node $node results:"
            if [ -f "$result_dir/simulation_report.md" ]; then
                echo "  Status: $(grep -o "✅\|❌" "$result_dir/simulation_report.md" | head -1) $(grep "SUCCESS\|FAILED" "$result_dir/simulation_report.md" | head -1)"
            fi
            if [ -f "$result_dir/stdout.log" ]; then
                echo "  Log size: $(wc -l < "$result_dir/stdout.log") lines"
            fi
        else
            print_warning "No results found for Node $node"
        fi
    done
}

# Function to run quick demo
run_demo() {
    print_status "Running Quick Multi-Node Demo with Streaming Dataset"
    run_training "basic" "5" "32" "1" "2" "HuggingFaceFW/fineweb-edu" "sample-10BT" "true"
}

# Function to run comprehensive test
run_comprehensive() {
    print_status "Running Comprehensive Multi-Node Test with Streaming Dataset"
    run_training "outer_optimizer" "10" "64" "1" "3" "HuggingFaceFW/fineweb-edu" "sample-10BT" "true"
}

# Function to run stress test
run_stress_test() {
    print_status "Running Multi-Node Stress Test with Streaming Dataset"
    run_training "gradient_sync" "20" "128" "1" "5" "wikitext" "wikitext-103-raw-v1" "true"
}

# Function to run simple dataset test (fallback)
run_simple_test() {
    print_status "Running Multi-Node Test with Simple Dataset (fallback)"
    run_training "basic" "10" "32" "1" "2" "" "" "false"
}

# Main execution
main() {
    echo "🐳 Advanced Multi-Node Multi-GPU Docker Training"
    echo "============================================================"
    
    case "${1:-demo}" in
        "build")
            build_image
            ;;
        "setup")
            setup_network
            ;;
        "demo")
            cleanup
            build_image
            setup_network
            run_demo
            show_results
            ;;
        "comprehensive")
            cleanup
            build_image
            setup_network
            run_comprehensive
            show_results
            ;;
        "stress")
            cleanup
            build_image
            setup_network
            run_stress_test
            show_results
            ;;
        "simple")
            cleanup
            build_image
            setup_network
            run_simple_test
            show_results
            ;;
        "clean")
            cleanup
            ;;
        "results")
            show_results
            ;;
        *)
            echo "Usage: $0 {build|setup|demo|comprehensive|stress|simple|clean|results}"
            echo ""
            echo "Commands:"
            echo "  build         - Build Docker image only"
            echo "  setup         - Setup Docker network only"
            echo "  demo          - Run quick demo (5 steps, streaming dataset)"
            echo "  comprehensive - Run full test (10 steps, streaming dataset)"
            echo "  stress        - Run stress test (20 steps, wikitext dataset)"
            echo "  simple        - Run simple test (10 steps, fallback dataset)"
            echo "  clean         - Clean up all resources"
            echo "  results       - Show latest results"
            echo ""
            echo "Examples:"
            echo "  $0 demo                    # Quick demonstration with streaming"
            echo "  $0 comprehensive           # Full validation with streaming"
            echo "  $0 stress                  # Stress testing with wikitext"
            echo "  $0 simple                  # Test with simple fallback dataset"
            exit 1
            ;;
    esac
}

# Trap cleanup on script exit
trap cleanup EXIT

# Run main function
main "$@"
