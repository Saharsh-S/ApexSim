"""Matplotlib visualizations for strategy-search results."""
from __future__ import annotations
from pathlib import Path
from typing import Sequence
import matplotlib.pyplot as plt
import numpy as np
from .strategy_search import SearchCandidate, StrategySearchResult

def plot_strategy_space(result: StrategySearchResult, path: str|None=None):
    fig, ax=plt.subplots(figsize=(9,5))
    x=[c.target_pace for c in result.candidates]; y=[c.risk_level for c in result.candidates]
    score=[r.statistics.win_probability for r in result.results]
    ax.scatter(x,y,c=score,s=18)
    ax.set_xlabel('Target pace factor'); ax.set_ylabel('Risk level'); ax.set_title('Generated Strategy Space')
    fig.tight_layout()
    if path: fig.savefig(path,dpi=150)
    return fig

def plot_expected_performance(result: StrategySearchResult, path: str|None=None):
    fig, ax=plt.subplots(figsize=(9,5))
    x=np.arange(len(result.results)); y=[r.statistics.mean_race_time for r in result.results]
    ax.plot(x,y,linewidth=0.8); ax.set_xlabel('Candidate index'); ax.set_ylabel('Expected race time (s)'); ax.set_title('Expected Strategy Performance')
    fig.tight_layout()
    if path: fig.savefig(path,dpi=150)
    return fig

def plot_search_risk_reward(result: StrategySearchResult, path: str|None=None):
    fig, ax=plt.subplots(figsize=(9,5))
    risk=[c.risk_level for c in result.candidates]; win=[r.statistics.win_probability for r in result.results]
    ax.scatter(risk,win,s=18); ax.set_xlabel('Risk level'); ax.set_ylabel('Win probability'); ax.set_title('Strategy Search Risk / Reward')
    fig.tight_layout()
    if path: fig.savefig(path,dpi=150)
    return fig

def plot_top_strategies(result: StrategySearchResult, path: str|None=None):
    top=result.top_results; labels=[r.strategy_id for r in top]; vals=[r.statistics.win_probability for r in top]
    fig, ax=plt.subplots(figsize=(10,6)); ax.barh(labels[::-1],vals[::-1]); ax.set_xlabel('Win probability'); ax.set_title('Top Strategies')
    fig.tight_layout()
    if path: fig.savefig(path,dpi=150)
    return fig
