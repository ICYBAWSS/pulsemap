#!/usr/bin/env python3
"""
PulseMap failed downloads retrier.
Reads training_data/failed_links.txt and retries them one-by-one.
"""
import sys
import time
from pathlib import Path

# Import the collector functions and configuration
from collect import process, tally, load_seen, DATA_DIR, TARGET_PER_CLASS, C_GREEN, C_YEL, C_DIM, C_RST, C_BOLD, print_tally

FAILED_FILE = DATA_DIR / "failed_links.txt"

def main():
    if not FAILED_FILE.exists():
        print(f"{C_YEL}No failed links file found at {FAILED_FILE}{C_RST}")
        return

    failed_links = FAILED_FILE.read_text().splitlines()
    failed_links = [lnk.strip() for lnk in failed_links if lnk.strip()]

    if not failed_links:
        print(f"{C_GREEN}No failed links to retry!{C_RST}")
        return

    print(f"{C_BOLD}Retrying {len(failed_links)} failed links one-by-one...{C_RST}")
    print(f"{C_DIM}Verbose log outputs will still go to training_data/collect.log{C_RST}\n")

    seen = load_seen()
    succeeded = []
    still_failed = []

    for idx, link in enumerate(failed_links, 1):
        print(f"{C_BOLD}[{idx}/{len(failed_links)}] Retrying:{C_RST} {link}")
        
        try:
            # Call process() synchronously
            res = process(link, TARGET_PER_CLASS, seen)
            if res is not None:
                got, added_unsorted, skips = res
                print(f"  {C_GREEN}✓ Success!{C_RST} Added +{got} labeled, +{added_unsorted} unsorted")
                succeeded.append(link)
            else:
                print(f"  {C_YEL}✗ Failed again.{C_RST}")
                still_failed.append(link)
        except Exception as e:
            print(f"  {C_YEL}✗ Error: {e}{C_RST}")
            still_failed.append(link)
            
        print()  # spacer

    # Print summary
    print(f"\n{C_BOLD}=== Retry Summary ==={C_RST}")
    print(f"  {C_GREEN}Total Succeeded:{C_RST} {len(succeeded)}")
    print(f"  {C_YEL}Total Still Failed:{C_RST} {len(still_failed)}")

    # Update failed_links.txt
    if still_failed:
        FAILED_FILE.write_text("\n".join(still_failed) + "\n")
        print(f"  {C_DIM}Updated {FAILED_FILE.name} with remaining failures.{C_RST}")
    else:
        FAILED_FILE.unlink(missing_ok=True)
        print(f"  {C_GREEN}All retried links succeeded! {FAILED_FILE.name} removed.{C_RST}")

    # Print tally at the end
    print_tally(tally(), TARGET_PER_CLASS)

if __name__ == "__main__":
    main()
