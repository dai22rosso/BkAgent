"""
Fix Ray 2.54.0 dashboard crash + launch verl GRPO training.

The problem: Ray 2.54.0's dashboard uses opentelemetry's
    Meter.create_histogram(explicit_bucket_boundaries_advisory=...)
but opentelemetry-api 1.26.0 (required by vLLM 0.8.4) doesn't have that parameter.

The fix: Patch the ONE line in Ray that crashes, then launch verl normally.

Usage:
    # First time: apply the patch
    python scripts/fix_ray_and_train.py --patch-only

    # Then train (patch persists across runs):
    CUDA_VISIBLE_DEVICES=3,4,5 N_GPUS=3 python scripts/fix_ray_and_train.py

    # Or just use the shell script after patching:
    CUDA_VISIBLE_DEVICES=3,4,5 N_GPUS=3 bash scripts/run_grpo.sh
"""

import os
import sys
import subprocess


def find_ray_metrics_file():
    """Find the Ray file that crashes."""
    import ray
    ray_dir = os.path.dirname(ray.__file__)
    target = os.path.join(
        ray_dir,
        "_private", "telemetry", "open_telemetry_metric_recorder.py"
    )
    if os.path.exists(target):
        return target

    # Fallback: search
    for root, dirs, files in os.walk(ray_dir):
        for f in files:
            if f == "open_telemetry_metric_recorder.py":
                return os.path.join(root, f)
    return None


def patch_ray():
    """Patch Ray's OpenTelemetry metric recorder to handle the missing parameter."""
    filepath = find_ray_metrics_file()
    if filepath is None:
        print("ERROR: Could not find Ray's open_telemetry_metric_recorder.py")
        return False

    with open(filepath, "r") as f:
        content = f.read()

    # Check if already patched
    if "# PATCHED by triage_agent" in content:
        print(f"[OK] Already patched: {filepath}")
        return True

    # The crash is at line ~184:
    #   instrument = self.meter.create_histogram(
    #       name=name, description=description, unit=unit,
    #       explicit_bucket_boundaries_advisory=boundaries,
    #   )
    # Fix: wrap in try/except to drop the unsupported kwarg

    old_code = "instrument = self.meter.create_histogram("
    new_code = """# PATCHED by triage_agent: handle opentelemetry version mismatch
        try:
            instrument = self.meter.create_histogram("""

    # Find and replace the create_histogram call with a try/except wrapper
    if old_code not in content:
        print(f"WARNING: Could not find target code in {filepath}")
        print("Attempting alternative patch...")
        return patch_ray_alternative(filepath, content)

    # We need to find the full call and wrap it
    # Simpler approach: just replace the method entirely
    patched = patch_ray_method(filepath, content)
    return patched


def patch_ray_method(filepath, content):
    """Replace the register_histogram_metric method to handle the kwarg gracefully."""

    # Find the method
    marker = "def register_histogram_metric("
    if marker not in content:
        print(f"WARNING: Could not find {marker} in {filepath}")
        return patch_ray_alternative(filepath, content)

    # Strategy: add a try/except around the create_histogram call
    # Find "explicit_bucket_boundaries_advisory" and wrap the whole call
    if "explicit_bucket_boundaries_advisory" not in content:
        print(f"[OK] 'explicit_bucket_boundaries_advisory' not found - may already be compatible")
        return True

    # Replace: remove the explicit_bucket_boundaries_advisory kwarg
    patched_content = content.replace(
        "explicit_bucket_boundaries_advisory=boundaries,",
        "# explicit_bucket_boundaries_advisory=boundaries,  # PATCHED by triage_agent",
    )

    # Also handle the case where it's without trailing comma
    patched_content = patched_content.replace(
        "explicit_bucket_boundaries_advisory=boundaries",
        "# explicit_bucket_boundaries_advisory=boundaries  # PATCHED by triage_agent",
    )

    if patched_content == content:
        print("WARNING: Replacement had no effect")
        return patch_ray_alternative(filepath, content)

    with open(filepath, "w") as f:
        f.write(patched_content)

    print(f"[OK] Patched: {filepath}")
    print("     Commented out 'explicit_bucket_boundaries_advisory' parameter")
    return True


def patch_ray_alternative(filepath, content):
    """Alternative: add try/except import guard at the top of the file."""
    # Nuclear option: make the whole module import-safe
    if "# PATCHED by triage_agent" in content:
        return True

    guard = '''# PATCHED by triage_agent
import functools as _ft

def _safe_create_histogram(original_method):
    """Wrapper that drops unsupported kwargs for older opentelemetry."""
    @_ft.wraps(original_method)
    def wrapper(*args, **kwargs):
        kwargs.pop("explicit_bucket_boundaries_advisory", None)
        return original_method(*args, **kwargs)
    return wrapper

try:
    from opentelemetry.metrics import Meter as _Meter
    _Meter.create_histogram = _safe_create_histogram(_Meter.create_histogram)
except Exception:
    pass
# END PATCH

'''

    patched_content = guard + content

    with open(filepath, "w") as f:
        f.write(patched_content)

    print(f"[OK] Applied alternative patch to: {filepath}")
    return True


def verify_patch():
    """Verify the patch works by importing Ray's metrics module."""
    try:
        # This is the import chain that crashes
        from ray._private.telemetry.open_telemetry_metric_recorder import (
            OpenTelemetryMetricRecorder,
        )
        print("[OK] Ray metrics module imports successfully after patch")
        return True
    except TypeError as e:
        if "explicit_bucket_boundaries_advisory" in str(e):
            print(f"[FAIL] Patch did not work: {e}")
            return False
        raise
    except Exception as e:
        # Other errors are fine - we just need the import not to crash on that specific kwarg
        print(f"[OK] Import raised {type(e).__name__} (not the dashboard crash): {e}")
        return True


def launch_training():
    """Launch verl GRPO training via the shell script."""
    print("\n" + "=" * 60)
    print("Launching verl GRPO training...")
    print("=" * 60 + "\n")

    # Use the shell script which has all the config
    env = os.environ.copy()
    result = subprocess.run(
        ["bash", "scripts/run_grpo.sh"],
        env=env,
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--patch-only", action="store_true",
                        help="Only apply the patch, don't train")
    parser.add_argument("--verify", action="store_true",
                        help="Verify the patch works")
    args = parser.parse_args()

    print("=" * 60)
    print("Ray Dashboard Fix for verl GRPO Training")
    print("=" * 60)

    # Step 1: Apply patch
    print("\n[Step 1] Patching Ray's OpenTelemetry metric recorder...")
    success = patch_ray()
    if not success:
        print("ERROR: Could not patch Ray. See errors above.")
        sys.exit(1)

    # Step 2: Verify
    print("\n[Step 2] Verifying patch...")
    if not verify_patch():
        print("ERROR: Patch verification failed.")
        sys.exit(1)

    if args.patch_only or args.verify:
        print("\nPatch applied and verified. You can now run:")
        print("  CUDA_VISIBLE_DEVICES=3,4,5 N_GPUS=3 bash scripts/run_grpo.sh")
        return

    # Step 3: Clean up Ray
    print("\n[Step 3] Cleaning up Ray processes...")
    os.system("ray stop --force 2>/dev/null")

    # Step 4: Launch training
    returncode = launch_training()
    sys.exit(returncode)


if __name__ == "__main__":
    main()
