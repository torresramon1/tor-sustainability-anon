import yaml
import pandas as pd
import re
import sys
import os
import argparse
import math
import csv
import statistics

# ==========================================
# CONFIGURATION
# ==========================================
SHADOW_CONFIG_FILE = "shadow.config.yaml"
HOSTS_BASE_DIR     = "shadow.data.template/hosts"
GML_FILE_PATH      = "../networkinfo_staging.gml"
CARBON_DATA_FILE   = "../hourly_country_intensity.csv"
V3BW_INIT_FILE     = "./shadow.data.template/hosts/bwauthority/v3bw.init.consensus"

INCLUDE_KEYWORDS = ["relay", "guard", "middle", "exit"]
EXCLUDE_KEYWORDS = ["client", "server", "authority", "4uthority", "bwauthority"]
DEFAULT_CARBON   = 475
SCALING_FACTOR   = 1244.4

DATA_DIR_NAME     = "bwauthority"
PROCESS_HOST_NAME = "4uthority1"


def parse_args():
    parser = argparse.ArgumentParser(description="Inject Hourly Carbon Priority Signals")
    parser.add_argument("--hours", type=int,   default=24,  help="Number of hours to simulate")
    parser.add_argument("--alpha", type=float, default=0.1, help="Sliding alpha")
    parser.add_argument("--k", type=float, default=-0.05, help="k-value")
    return parser.parse_args()


def parse_gml_simple(filepath):
    node_map        = {}
    current_id      = None
    id_pattern      = re.compile(r'^\s*id\s+(\d+)')
    country_pattern = re.compile(r'^\s*country_code\s+"(\w+)"')
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if m := id_pattern.search(line):
                    current_id = int(m.group(1))
                if m := country_pattern.search(line):
                    if current_id is not None:
                        node_map[current_id] = m.group(1).upper()
    except Exception as e:
        print(f"[!] GML Error: {e}")
        sys.exit(1)
    return node_map


def parse_bandwidth(bw_str):
    if not bw_str:
        return 0
    return int(bw_str.split()[0])


def parse_v3bw_init(filepath: str) -> dict[str, int]:
    """
    Parses v3bw.init.consensus and returns a dict of {nickname: bw_value}.
    Skips header line and authority lines.
    """
    nick_bw = {}
    pattern = re.compile(r'node_id=\S+\s+bw=(\d+)\s+nick=(\S+)')
    try:
        with open(filepath, 'r') as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    bw   = int(m.group(1))
                    nick = m.group(2)
                    nick_bw[nick] = bw
    except FileNotFoundError:
        print(f"[!] Error: {filepath} not found.")
        sys.exit(1)
    print(f"[*] Loaded {len(nick_bw)} relay bandwidths from {filepath}")
    return nick_bw


def get_fingerprint(hostname):
    fp_path = os.path.join(HOSTS_BASE_DIR, hostname, "fingerprint")
    if not os.path.exists(fp_path):
        return None
    try:
        with open(fp_path, 'r') as f:
            content = f.read().strip()
            parts   = content.split()
            return parts[1] if len(parts) >= 2 else content
    except:
        return None


def generate_hourly_v3bw(relays, args, carbon_series, h):
    # ── Pre-compute constants once outside the relay loop ─────────────────
    max_bw = max(r['old_bw'] for r in relays)
    min_bw = min(r['old_bw'] for r in relays)

    # Carbon stats for sigmoid
    carbon_values = list(carbon_series.values())
    x0 = statistics.quantiles(carbon_values, n=4)[0]  # Q1
    #x0 = statistics.mean(carbon_values)# avg 

    # Max green score for normalisation
    max_green = max(
        (1.0 / carbon_series.get(r['country'], DEFAULT_CARBON)
         for r in relays
         if carbon_series.get(r['country'], DEFAULT_CARBON) > 0),
        default=1.0
    )

    L = 1.0
    #k = -0.05
    k = args.k

    # Standard V3BW header
    lines = ["946684801\n"]
    lines.append("node_id=$F7010DD8329D3E3411C4D66818BC039E975E8CFB\tbw=20\tnick=4uthority1\n")
    lines.append("node_id=$B42E38B678F581B655B6090D35EACEB81BB6090F\tbw=20\tnick=4uthority2\n")
    lines.append("node_id=$FDFBA4600CE861B0C4BD2B12316FB9EC0652A841\tbw=20\tnick=4uthority3\n")

    with open(f"./new_relay_bw_{h}.csv", "w", newline="") as csv_file:
        fieldnames = ["Hostname", "Country", "old_bw", "new_bw", "CI"]
        writer     = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()

        for r in relays:
            if not r.get('fingerprint'):
                continue

            c_val = carbon_series.get(r['country'], DEFAULT_CARBON)
            if isinstance(c_val, pd.Series):
                c_val = c_val.iloc[0]

            # Sigmoid logistic function
            x_x0            = c_val - x0
            lf               = L / (1.0 + math.exp(k * x_x0))
            old_bw           = r['old_bw']
            new_bw_kilobits  = int( ((1 - args.alpha) * old_bw) + (args.alpha * ((1.0 - lf) * (max_bw - min_bw))) )

            writer.writerow({
                "Hostname": r['hostname'],
                "Country":  r['country'],
                "old_bw":   old_bw,
                "new_bw":   new_bw_kilobits,
                "CI":       c_val,
            })

            lines.append(
                f"node_id=${r['fingerprint']}\tbw={new_bw_kilobits}\tnick={r['hostname']}\n"
            )

    return lines


def main():
    args = parse_args()
    print(f"Using alpha = {args.alpha}")
    print(f"Using k = {args.k}")

    data_dir = os.path.join(HOSTS_BASE_DIR, DATA_DIR_NAME)
    if not os.path.exists(data_dir):
        print(f"[!] Error: Data Directory {data_dir} not found.")
        sys.exit(1)

    print(f"[*] Loading Configuration...")
    with open(SHADOW_CONFIG_FILE, 'r') as f:
        config = yaml.safe_load(f)
    node_map = parse_gml_simple(GML_FILE_PATH)

    try:
        df = pd.read_csv(CARBON_DATA_FILE)
        df.set_index('Country', inplace=True)
    except Exception as e:
        print(f"[!] Error reading Carbon CSV: {e}")
        sys.exit(1)

    # ── Gather relay data and fingerprints ────────────────────────────────
    relays  = []
    hosts   = config.get('hosts', {})
    nick_bw = parse_v3bw_init(V3BW_INIT_FILE)

    print(f"[*] Gathering fingerprints for relays...")
    for hostname, host_data in hosts.items():
        h_lower = hostname.lower()
        if not any(k in h_lower for k in INCLUDE_KEYWORDS):
            continue
        if any(k in h_lower for k in EXCLUDE_KEYWORDS):
            continue

        node_id = host_data.get('network_node_id')
        country = node_map.get(node_id, '??')
        fp      = get_fingerprint(hostname)

        if not fp:
            print(f"[!] Warning: No fingerprint file found for {hostname}. Skipping.")
            continue

        bw_val = nick_bw.get(hostname)
        if bw_val is None:
            print(f"[!] Warning: No bw entry found for {hostname} "
                  f"in v3bw.init.consensus. Skipping.")
            continue

        relays.append({
            'hostname':    hostname,
            'country':     country,
            'old_bw':      bw_val,
            'fingerprint': fp,
        })

    print(f"[*] {len(relays)} relays loaded.")

    # ── Schedule updates ──────────────────────────────────────────────────
    print(f"[*] Scheduling updates on host: {PROCESS_HOST_NAME} "
          f"targeting folder: {DATA_DIR_NAME}")

    auth_host_entry = config['hosts'].get(PROCESS_HOST_NAME)
    if not auth_host_entry:
        print(f"[!] Error: Host {PROCESS_HOST_NAME} missing from shadow.config.yaml.")
        sys.exit(1)

    if 'processes' not in auth_host_entry:
        auth_host_entry['processes'] = []

    for h in range(args.hours):
        col_idx = h % len(df.columns)
        lines   = generate_hourly_v3bw(relays, args, df.iloc[:, col_idx].to_dict(), h)

        filename     = f"v3bw.{h}"
        filepath_abs = os.path.abspath(os.path.join(data_dir, filename))
        target_abs   = os.path.abspath(os.path.join(data_dir, "v3bw"))

        with open(filepath_abs, 'w') as f:
            f.writelines(lines)

        print(f"  Written: {filepath_abs}")

        # Schedule cp command to replace active v3bw at sim time h
        update_cmd = {
            "path":       "/usr/bin/cp",
            "args":       f"{filepath_abs} {target_abs}",
            "start_time": h,
        }

        # Deduplicate — compare as int
        is_dupe = any(
            p.get('start_time') == h and "/usr/bin/cp" in p.get('path', '')
            for p in auth_host_entry['processes']
        )

        if not is_dupe:
            auth_host_entry['processes'].append(update_cmd)

    # ── Save updated config ───────────────────────────────────────────────
    with open(SHADOW_CONFIG_FILE, 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    print(f"\n[+] Success! Files created in {DATA_DIR_NAME} with correct node_ids.")


if __name__ == "__main__":
    main()
