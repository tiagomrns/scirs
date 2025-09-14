#!/bin/bash

# Dependency Security Audit Script for scirs2-optim
# This script performs comprehensive dependency security checks

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}🔍 Starting Dependency Security Audit for scirs2-optim${NC}"
echo "=================================================="

# Function to print section headers
print_section() {
    echo -e "\n${BLUE}📋 $1${NC}"
    echo "----------------------------------------"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install cargo tools if needed
install_cargo_tools() {
    print_section "Installing Required Tools"
    
    if ! command_exists cargo-audit; then
        echo -e "${YELLOW}Installing cargo-audit...${NC}"
        cargo install cargo-audit
    else
        echo -e "${GREEN}✓ cargo-audit already installed${NC}"
    fi
    
    if ! command_exists cargo-outdated; then
        echo -e "${YELLOW}Installing cargo-outdated...${NC}"
        cargo install cargo-outdated
    else
        echo -e "${GREEN}✓ cargo-outdated already installed${NC}"
    fi
    
    if ! command_exists cargo-deny; then
        echo -e "${YELLOW}Installing cargo-deny...${NC}"
        cargo install cargo-deny
    else
        echo -e "${GREEN}✓ cargo-deny already installed${NC}"
    fi
}

# Function to run security audit
run_security_audit() {
    print_section "Running Security Vulnerability Scan"
    
    cd "$PROJECT_ROOT"
    
    # Generate Cargo.lock if it doesn't exist
    if [ ! -f "Cargo.lock" ]; then
        echo -e "${YELLOW}Generating Cargo.lock...${NC}"
        cargo check --locked 2>/dev/null || cargo check
    fi
    
    # Run cargo audit
    if cargo audit --json > /tmp/audit_results.json 2>/dev/null; then
        echo -e "${GREEN}✓ No security vulnerabilities found${NC}"
    else
        echo -e "${RED}⚠️  Security vulnerabilities detected${NC}"
        cargo audit --color=always
        echo -e "${YELLOW}Check the detailed report above${NC}"
    fi
}

# Function to check for outdated dependencies
check_outdated_dependencies() {
    print_section "Checking for Outdated Dependencies"
    
    cd "$PROJECT_ROOT"
    
    # Create a temporary report file
    OUTDATED_REPORT="/tmp/outdated_report.txt"
    
    if cargo outdated --format=json > /tmp/outdated.json 2>/dev/null; then
        # Parse JSON and create human-readable report
        python3 -c "
import json
import sys

try:
    with open('/tmp/outdated.json', 'r') as f:
        data = json.load(f)
    
    outdated = []
    for dep in data.get('dependencies', []):
        if dep.get('latest', '') != dep.get('project', ''):
            outdated.append({
                'name': dep.get('name', ''),
                'current': dep.get('project', ''),
                'latest': dep.get('latest', ''),
                'kind': dep.get('kind', '')
            })
    
    if outdated:
        print('Outdated dependencies found:')
        for dep in outdated:
            print(f\"  {dep['name']}: {dep['current']} -> {dep['latest']} ({dep['kind']})\")
        sys.exit(1)
    else:
        print('All dependencies are up to date')
        sys.exit(0)
except Exception as e:
    print(f'Error parsing outdated report: {e}')
    sys.exit(1)
" && echo -e "${GREEN}✓ All dependencies are up to date${NC}" || echo -e "${YELLOW}⚠️  Some dependencies are outdated (see above)${NC}"
    else
        echo -e "${YELLOW}Could not check for outdated dependencies (dependency conflicts)${NC}"
        echo "This is likely due to workspace dependency conflicts"
    fi
}

# Function to analyze dependency tree
analyze_dependency_tree() {
    print_section "Analyzing Dependency Tree"
    
    cd "$PROJECT_ROOT"
    
    echo "Generating dependency tree analysis..."
    
    # Get dependency counts
    TOTAL_DEPS=$(cargo tree --format="{p}" | wc -l)
    DIRECT_DEPS=$(cargo metadata --format-version=1 2>/dev/null | python3 -c "
import json, sys
try:
    data = json.load(sys.stdin)
    pkg = next(p for p in data['packages'] if p['name'] == 'scirs2-optim')
    print(len(pkg['dependencies']))
except:
    print('Unknown')
" 2>/dev/null || echo "Unknown")
    
    echo -e "Direct dependencies: ${BLUE}$DIRECT_DEPS${NC}"
    echo -e "Total dependencies: ${BLUE}$TOTAL_DEPS${NC}"
    
    # Check for duplicate versions
    echo -e "\nChecking for duplicate dependency versions..."
    DUPLICATES=$(cargo tree --duplicates 2>/dev/null | grep -v "└──" | wc -l || echo "0")
    
    if [ "$DUPLICATES" -gt 0 ]; then
        echo -e "${YELLOW}⚠️  Found $DUPLICATES potential duplicate dependencies${NC}"
        cargo tree --duplicates 2>/dev/null || true
    else
        echo -e "${GREEN}✓ No duplicate dependencies found${NC}"
    fi
}

# Function to check dependency licenses
check_licenses() {
    print_section "Checking Dependency Licenses"
    
    cd "$PROJECT_ROOT"
    
    # Generate license report
    if command_exists cargo-license; then
        echo "Generating license report..."
        cargo license --json > /tmp/licenses.json 2>/dev/null || echo "Could not generate license report"
    else
        echo -e "${YELLOW}cargo-license not installed, skipping license check${NC}"
        echo "To install: cargo install cargo-license"
    fi
}

# Function to generate security recommendations
generate_recommendations() {
    print_section "Security Recommendations"
    
    cat << 'EOF'
🔐 Security Best Practices:

1. Regular Updates:
   - Run this audit script monthly
   - Subscribe to Rust security advisories
   - Monitor CVE databases for dependency vulnerabilities

2. Dependency Management:
   - Pin critical dependencies in production
   - Use cargo-deny for policy enforcement
   - Review new dependencies before adding

3. Plugin Security:
   - Always verify plugin signatures in production
   - Use restrictive sandbox settings
   - Maintain an approved plugin allowlist
   - Monitor plugin integrity

4. Development Practices:
   - Enable all security lints in CI/CD
   - Use cargo-audit in automated testing
   - Regular security code reviews

5. Runtime Security:
   - Enable comprehensive logging
   - Monitor for unusual plugin behavior
   - Implement rate limiting
   - Use principle of least privilege
EOF
}

# Function to create deny.toml configuration
create_deny_config() {
    if [ ! -f "$PROJECT_ROOT/deny.toml" ]; then
        print_section "Creating cargo-deny Configuration"
        
        cat > "$PROJECT_ROOT/deny.toml" << 'EOF'
# cargo-deny configuration for scirs2-optim
[graph]
targets = [
    { triple = "x86_64-unknown-linux-gnu" },
    { triple = "x86_64-unknown-linux-musl" },
    { triple = "x86_64-apple-darwin" },
    { triple = "x86_64-pc-windows-msvc" },
]

[licenses]
unlicensed = "deny"
copyleft = "allow"
allow = [
    "MIT",
    "Apache-2.0",
    "Apache-2.0 WITH LLVM-exception",
    "BSD-2-Clause",
    "BSD-3-Clause",
    "ISC",
    "Unicode-DFS-2016",
    "CC0-1.0",
]
confidence-threshold = 0.8

[bans]
multiple-versions = "warn"
wildcards = "allow"
highlight = "all"

# Known security issues to deny
deny = [
    # Add specific crates with known security issues here
]

[advisories]
vulnerability = "deny"
unmaintained = "warn"
yanked = "deny"
notice = "warn"
ignore = [
    # Add specific advisories to ignore here if needed
]

[sources]
unknown-registry = "deny"
unknown-git = "deny"
allow-registry = ["https://github.com/rust-lang/crates.io-index"]
EOF
        
        echo -e "${GREEN}✓ Created deny.toml configuration${NC}"
    fi
}

# Function to run comprehensive audit
run_comprehensive_audit() {
    print_section "Running Comprehensive Security Audit"
    
    cd "$PROJECT_ROOT"
    
    if command_exists cargo-deny; then
        echo "Running cargo-deny checks..."
        cargo deny check advisories 2>/dev/null && echo -e "${GREEN}✓ Advisory check passed${NC}" || echo -e "${YELLOW}⚠️  Advisory check failed${NC}"
        cargo deny check licenses 2>/dev/null && echo -e "${GREEN}✓ License check passed${NC}" || echo -e "${YELLOW}⚠️  License check failed${NC}"
        cargo deny check bans 2>/dev/null && echo -e "${GREEN}✓ Ban check passed${NC}" || echo -e "${YELLOW}⚠️  Ban check failed${NC}"
    else
        echo -e "${YELLOW}cargo-deny not available, skipping comprehensive audit${NC}"
    fi
}

# Main execution
main() {
    echo -e "${BLUE}Dependency Security Audit - $(date)${NC}\n"
    
    # Change to project directory
    cd "$PROJECT_ROOT"
    
    # Install required tools
    install_cargo_tools
    
    # Create deny configuration if needed
    create_deny_config
    
    # Run all checks
    run_security_audit
    check_outdated_dependencies
    analyze_dependency_tree
    check_licenses
    run_comprehensive_audit
    
    # Generate final report
    generate_recommendations
    
    echo -e "\n${GREEN}🎉 Security audit completed!${NC}"
    echo -e "Report timestamp: $(date)"
    echo -e "For detailed security features, see: ${BLUE}security_audit_report.md${NC}"
}

# Run the main function
main "$@"