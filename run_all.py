import argparse
import subprocess
import sys
import os

def run_script(script_name, args=None):
    """Runs a python script as a subprocess."""
    print(f"
{'='*60}")
    print(f"🚀 Running: {script_name} {' '.join(args) if args else ''}")
    print(f"{'='*60}
")
    
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
        
    try:
        # We use subprocess.run and check for errors
        result = subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"
❌ Error: {script_name} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run RAG Pipeline Lab Steps")
    parser.add_argument("--step", type=int, choices=[1, 2, 3, 4], help="Run a specific step (1-4)")
    args = parser.parse_args()

    # Define steps
    steps = {
        1: ("01_langsmith_rag_pipeline.py", []),
        2: ("02_prompt_hub_ab_routing.py", []),
        3: ("03_ragas_evaluation.py", []),
        4: ("04_guardrails_validator.py", []) # Note: Task 4 runs both by default now
    }

    if args.step:
        # Run specific step
        script, script_args = steps[args.step]
        run_script(script, script_args)
    else:
        # Run all steps sequentially
        for step_num in sorted(steps.keys()):
            success = run_script(steps[step_num][0], steps[step_num][1])
            if not success:
                print(f"
🛑 Stopping at Step {step_num} due to failure.")
                sys.exit(1)
        
        print("
" + "="*60)
        print("🎉 All lab tasks completed successfully!")
        print("="*60)

if __name__ == "__main__":
    main()
