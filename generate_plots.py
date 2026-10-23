import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Paths
current_dir = os.path.dirname(os.path.abspath(__file__))
BENCHMARK_FILE = os.path.join(current_dir, "benchmark_data.json")
OUTPUT_IMAGE = os.path.join(current_dir, "benchmark_results.png")


def generate_benchmark_plots():
    if not os.path.exists(BENCHMARK_FILE):
        print(f"Error: {BENCHMARK_FILE} does not exist. Please run queries first.")
        return

    with open(BENCHMARK_FILE, "r", encoding="utf-8") as f:
        records = json.load(f)

    if not records:
        print("No benchmark records found in benchmark_data.json.")
        return

    queries = [f"Q{r['id']}: " + (r['query'][:28] + "..." if len(r['query']) > 28 else r['query']) for r in records]
    genai_latencies = [r["genai"]["latency_ms"] for r in records]
    laya_latencies = [r["laya"]["latency_ms"] for r in records]
    genai_tokens = [r["genai"]["tokens_generated"] for r in records]
    laya_tokens = [r["laya"]["tokens_generated"] for r in records]
    agreements = [r["agreement"] for r in records]

    # Create figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('#0f172a')  # Dark sleek background
    for ax in (ax1, ax2):
        ax.set_facecolor('#1e293b')
        ax.tick_params(colors='#e2e8f0', labelsize=10)
        for spine in ax.spines.values():
            spine.set_color('#334155')

    y_pos = np.arange(len(queries))
    bar_height = 0.35

    # --- Plot 1: Latency Comparison (ms) ---
    b1 = ax1.barh(y_pos - bar_height/2, genai_latencies, bar_height, label='GenAI (Qwen 2.5)', color='#f43f5e', alpha=0.9)
    b2 = ax1.barh(y_pos + bar_height/2, laya_latencies, bar_height, label='Decision Model (Laya)', color='#10b981', alpha=0.9)

    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(queries, color='#f8fafc', fontweight='medium')
    ax1.invert_yaxis()
    ax1.set_xlabel('Latency (Milliseconds) - Lower is Faster', color='#94a3b8', fontweight='bold', fontsize=11)
    ax1.set_title('Decision Latency: GenAI vs. Laya', color='#f8fafc', fontsize=14, fontweight='bold', pad=12)
    ax1.legend(loc='lower right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')

    # Add value annotations
    for bar in b1:
        w = bar.get_width()
        ax1.text(w + 50, bar.get_y() + bar.get_height()/2, f"{w:.0f}ms", va='center', ha='left', color='#f43f5e', fontsize=9, fontweight='bold')
    for bar in b2:
        w = bar.get_width()
        ax1.text(w + 50, bar.get_y() + bar.get_height()/2, f"{w:.0f}ms", va='center', ha='left', color='#10b981', fontsize=9, fontweight='bold')

    # --- Plot 2: Token Consumption ---
    b3 = ax2.barh(y_pos - bar_height/2, genai_tokens, bar_height, label='GenAI Tokens', color='#38bdf8', alpha=0.9)
    b4 = ax2.barh(y_pos + bar_height/2, laya_tokens, bar_height, label='Laya Tokens (Zero Tokens)', color='#a855f7', alpha=0.9)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([f"Agreement: {'YES' if a else 'NO'}" for a in agreements], color='#34d399' if all(agreements) else '#f8fafc')
    ax2.invert_yaxis()
    ax2.set_xlabel('Tokens Generated - Lower is Cheaper', color='#94a3b8', fontweight='bold', fontsize=11)
    ax2.set_title('Token Cost & Tool Agreement', color='#f8fafc', fontsize=14, fontweight='bold', pad=12)
    ax2.legend(loc='lower right', facecolor='#0f172a', edgecolor='#334155', labelcolor='#e2e8f0')

    # Add token annotations
    for bar in b3:
        w = bar.get_width()
        ax2.text(w + 2, bar.get_y() + bar.get_height()/2, f"{int(w)} toks", va='center', ha='left', color='#38bdf8', fontsize=9, fontweight='bold')
    for bar in b4:
        w = bar.get_width()
        ax2.text(w + 2, bar.get_y() + bar.get_height()/2, "0 toks", va='center', ha='left', color='#a855f7', fontsize=9, fontweight='bold')

    plt.subplots_adjust(left=0.22, right=0.95, top=0.88, bottom=0.15, wspace=0.35)
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight', facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n[Plot Generated] Saved benchmark plot to: {OUTPUT_IMAGE}")


if __name__ == "__main__":
    generate_benchmark_plots()
