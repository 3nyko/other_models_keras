import csv
from pathlib import Path

input_path = Path("data/Car-Hacking Dataset/normal_run_data.txt")
output_path = Path("data/Car-Hacking Dataset/normal_run_data.csv")

if not input_path.exists():
    raise FileNotFoundError(f"Input file not found: {input_path}")

output_path.parent.mkdir(parents=True, exist_ok=True)

converted = 0
skipped = 0

with input_path.open("r", encoding="utf-8", errors="replace") as src, output_path.open(
    "w", encoding="utf-8", newline=""
) as dst:
    writer = csv.writer(dst)

    for line in src:
        parts = line.strip().split()

        # Expected format:
        # Timestamp: <ts> ID: <id> 000 DLC: <dlc> <byte1> ... <byteN>
        if len(parts) < 9:
            skipped += 1
            continue

        timestamp = parts[1]
        can_id = parts[3].lower()

        try:
            # "DLC:" is at index 5, value is index 6
            dlc = int(parts[6])
        except ValueError:
            skipped += 1
            continue

        # Data bytes start right after DLC value
        data_bytes = [b.lower() for b in parts[7 : 7 + dlc]]
        if len(data_bytes) < dlc:
            skipped += 1
            continue

        writer.writerow([timestamp, can_id, str(dlc), *data_bytes, "R"])
        converted += 1

print(f"Done. Converted rows: {converted}, skipped rows: {skipped}")
print(f"Output written to: {output_path}")
