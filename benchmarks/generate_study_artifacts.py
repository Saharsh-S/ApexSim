"""Generate graphs and the machine-readable final benchmark table."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def generate(data, out):
    out=Path(out)
    sim=[x for x in data['simulation'] if x.get('status')=='measured']
    if sim:
        labels=[x['implementation'] for x in sim]; vals=[x['throughput_sims_s'] for x in sim]
        plt.figure(figsize=(8,5)); plt.bar(labels,vals); plt.ylabel('simulations / second'); plt.title('Measured simulation throughput'); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(out/'simulation_throughput.png',dpi=160); plt.close()
        if len(vals)>=2:
            speedups={labels[i]: vals[i]/vals[0] for i in range(len(vals))}
            (out/'simulation_speedups.json').write_text(json.dumps(speedups,indent=2)+'\n')
    inf=[x for x in data['inference'] if x.get('status')=='measured' and x.get('batch_size')==1 and 'latency_ms' in x]
    if inf:
        plt.figure(figsize=(8,5)); plt.bar([x['implementation'] for x in inf],[x['latency_ms'] for x in inf]); plt.ylabel('latency (ms)'); plt.title('Measured batch-1 inference latency'); plt.xticks(rotation=20); plt.tight_layout(); plt.savefig(out/'inference_latency.png',dpi=160); plt.close()
    batch=[x for x in data['inference'] if x.get('implementation')=='pytorch_batch_fp32' and x.get('status')=='measured']
    if batch:
        plt.figure(figsize=(8,5)); plt.plot([x['batch_size'] for x in batch],[x['throughput_samples_s'] for x in batch],marker='o'); plt.xscale('log',base=2); plt.xlabel('batch size'); plt.ylabel('samples / second'); plt.title('PyTorch FP32 batching'); plt.tight_layout(); plt.savefig(out/'batching_throughput.png',dpi=160); plt.close()
    ab=data.get('historical_optimization_ablation',[])
    noise=next((x for x in ab if x.get('optimization')=='conditional stochastic work'),None)
    if noise:
        plt.figure(figsize=(7,5)); plt.bar(['before','after'],[noise['baseline_s']*1000,noise['optimized_s']*1000]); plt.ylabel('kernel time (ms)'); plt.title('Measured CUDA-simulator optimization: conditional stochastic work'); plt.tight_layout(); plt.savefig(out/'optimization_contribution_cudasim.png',dpi=160); plt.close()
    transfer=next((x for x in ab if x.get('optimization')=='persistent device configuration'),None)
    if transfer:
        plt.figure(figsize=(7,5)); plt.bar(['before','after'],[transfer['baseline_bytes'],transfer['optimized_bytes']]); plt.ylabel('host-to-device bytes'); plt.title('Measured CUDA-simulator optimization: persistent configuration'); plt.tight_layout(); plt.savefig(out/'optimization_transfer_bytes.png',dpi=160); plt.close()
    precision=[x for x in data['inference'] if x.get('batch_size')==1 and x.get('implementation') in ('pytorch_fp32','pytorch_fp16','pytorch_bf16') and x.get('status')=='measured']
    if precision:
        plt.figure(figsize=(7,5)); plt.bar([x['implementation'] for x in precision],[x['latency_ms'] for x in precision]); plt.ylabel('latency (ms)'); plt.title('Measured precision ablation on CPU'); plt.tight_layout(); plt.savefig(out/'optimization_precision.png',dpi=160); plt.close()
    # Final normalized table.
    table=[]
    for x in data['simulation']:
        table.append({k:x.get(k) for k in ('implementation','status','runtime_s','throughput_sims_s','latency_ms_per_sim','cpu_util_percent','gpu_util_percent','gpu_memory_mb','energy_j')})
    for x in data['inference']:
        if x.get('batch_size',1)==1:
            table.append({"implementation":x.get('implementation'),"status":x.get('status'),"runtime_s":x.get('runtime_s'),"throughput_sims_s":x.get('throughput_samples_s'),"latency_ms_per_sim":x.get('latency_ms'),"cpu_util_percent":x.get('cpu_util_percent'),"gpu_util_percent":x.get('gpu_util_percent'),"gpu_memory_mb":x.get('gpu_memory_mb'),"energy_j":x.get('energy_j')})
    (out/'final_benchmark_table.json').write_text(json.dumps(table,indent=2)+'\n')
    lines=['# Final Benchmark Table','', '| Implementation | Status | Runtime (s) | Throughput | Latency (ms) | CPU util. | GPU util. | GPU memory (MB) | Energy (J) |','|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in table:
        def fmt(v): return '-' if v is None else f'{v:.4f}' if isinstance(v,float) else str(v)
        lines.append('| {implementation} | {status} | {runtime_s} | {throughput_sims_s} | {latency_ms_per_sim} | {cpu_util_percent} | {gpu_util_percent} | {gpu_memory_mb} | {energy_j} |'.format(**{k:fmt(r.get(k)) for k in ('implementation','status','runtime_s','throughput_sims_s','latency_ms_per_sim','cpu_util_percent','gpu_util_percent','gpu_memory_mb','energy_j')}))
    (out/'final_benchmark_table.md').write_text('\n'.join(lines)+'\n')
