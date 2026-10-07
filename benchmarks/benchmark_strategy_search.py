"""Benchmark automatic strategy search across CPU and CUDA backends."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))
from f1sim import StrategySearchConfig, generate_search_candidates, search_cpu, search_gpu, summarize_top_strategies
from basic_race import build_config
from f1sim.gpu import cuda_available, cuda_backend_status

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--candidates', type=int, default=1000)
    p.add_argument('--simulations', type=int, default=100)
    p.add_argument('--backend', choices=['python','numpy','cuda','all'], default='all')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--output', default='benchmarks/results/phase6_search.json')
    a=p.parse_args()
    config=build_config()
    sc=StrategySearchConfig(candidate_count=a.candidates, simulations_per_strategy=a.simulations, seed=a.seed)
    candidates=generate_search_candidates(config, sc)
    records={'candidates':len(candidates),'simulations_per_strategy':a.simulations,'total_simulations':len(candidates)*a.simulations,'cuda_status':cuda_backend_status()}
    if a.backend in {'python','all'}:
        r=search_cpu(config,candidates,a.simulations,a.seed,engine='python')
        records['python']={'runtime_seconds':r.runtime_seconds,'strategies_per_second':r.strategies_per_second,'simulations_per_second':r.simulations_per_second,'top10':[x.__dict__ for x in summarize_top_strategies(r)]}
    if a.backend in {'numpy','all'}:
        r=search_cpu(config,candidates,a.simulations,a.seed,engine='numpy')
        records['numpy']={'runtime_seconds':r.runtime_seconds,'strategies_per_second':r.strategies_per_second,'simulations_per_second':r.simulations_per_second,'top10':[x.__dict__ for x in summarize_top_strategies(r)]}
    if a.backend in {'cuda','all'}:
        if cuda_available():
            r=search_gpu(config,candidates,a.simulations,a.seed)
            records['cuda']={'runtime_seconds':r.runtime_seconds,'strategies_per_second':r.strategies_per_second,'simulations_per_second':r.simulations_per_second,'top10':[x.__dict__ for x in summarize_top_strategies(r)]}
        else:
            records['cuda']={'status':'not_run','reason':cuda_backend_status()}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(records,indent=2))
    print(json.dumps(records,indent=2))
if __name__=='__main__': main()
