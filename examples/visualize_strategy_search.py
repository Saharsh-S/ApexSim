"""Generate Phase 6 strategy-search plots."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from basic_race import build_config
from f1sim import StrategySearchConfig, generate_search_candidates, search_cpu
from f1sim.search_visualization import plot_strategy_space, plot_expected_performance, plot_search_risk_reward, plot_top_strategies

def main():
    out=Path('benchmarks/results'); out.mkdir(parents=True,exist_ok=True)
    cfg=StrategySearchConfig(candidate_count=100,simulations_per_strategy=50,seed=42)
    r=search_cpu(build_config(),generate_search_candidates(build_config(),cfg),50,42,engine='numpy')
    plot_strategy_space(r,out/'phase6_strategy_space.png').savefig(out/'phase6_strategy_space.png')
    plot_expected_performance(r,out/'phase6_expected_performance.png').savefig(out/'phase6_expected_performance.png')
    plot_search_risk_reward(r,out/'phase6_risk_reward.png').savefig(out/'phase6_risk_reward.png')
    plot_top_strategies(r,out/'phase6_top_strategies.png').savefig(out/'phase6_top_strategies.png')
    print('Generated Phase 6 plots in benchmarks/results/')
if __name__=='__main__': main()
