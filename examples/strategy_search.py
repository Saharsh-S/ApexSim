"""Small automatic strategy-search example."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from f1sim import StrategySearchConfig, generate_search_candidates, search_cpu
from basic_race import build_config

def main():
    config=build_config()
    search=StrategySearchConfig(candidate_count=100, simulations_per_strategy=50, seed=42)
    candidates=generate_search_candidates(config, search)
    result=search_cpu(config,candidates,search.simulations_per_strategy,search.seed,engine='numpy')
    print('Candidates:', result.candidate_count)
    print('Simulations:', result.total_simulations)
    print('Runtime:', f'{result.runtime_seconds:.4f}s')
    for rank, i in enumerate(result.top_indices,1):
        c=result.candidates[i]; s=result.results[i].statistics
        print(f'{rank:2d}. {c.strategy_id} win={s.win_probability:.3f} podium={s.podium_probability:.3f} time={s.mean_race_time:.2f}s risk={c.risk_level:.3f}')
if __name__=='__main__': main()
