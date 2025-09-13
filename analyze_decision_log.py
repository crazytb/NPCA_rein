import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def analyze_decision_log(filepath="./semi_mdp_results/decision_log.csv"):
    """
    Analyzes the decision log to understand model behavior.

    Args:
        filepath (str): Path to the decision_log.csv file.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {filepath}")
        print("먼저 main_semi_mdp_training.py를 실행하여 로그 파일을 생성해주세요.")
        return

    print(f"✅ 총 {len(df)}개의 결정 로그를 읽었습니다.")

    # Filter for DRL agent decisions if strategy column exists
    if 'strategy' in df.columns:
        drl_df = df[df['strategy'] != 'fixed'].copy()
        print(f"🤖 DRL 에이전트의 결정 {len(drl_df)}개를 분석합니다.")
    else:
        # If no strategy is specified, assume all are DRL decisions
        drl_df = df.copy()

    # Define bins for OBSS remaining time
    bins = [0, 10, 20, 30, 40, 50, 75, 100, 150, 200, np.inf]
    labels = ["0-10", "11-20", "21-30", "31-40", "41-50", "51-75", "76-100", "101-150", "151-200", "200+"]
    
    drl_df['obss_bin'] = pd.cut(drl_df['primary_channel_obss_occupied_remained'], bins=bins, labels=labels, right=False)

    # Calculate the switch probability for each bin
    action_distribution = drl_df.groupby('obss_bin')['action'].value_counts(normalize=True).unstack().fillna(0)
    
    if 1 not in action_distribution.columns:
        action_distribution[1] = 0.0
    
    switch_prob = action_distribution[1]

    print("\n📈 OBSS 잔여 시간에 따른 NPCA 전환 확률:")
    print("="*50)
    for idx, val in switch_prob.items():
        print(f"  - OBSS {str(idx):>7s} slots: Switch Prob = {val:.2%}")
    print("="*50)

    # Visualization
    fig, ax = plt.subplots(figsize=(12, 7))
    switch_prob.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    
    ax.set_title('NPCA Switch Probability vs. OBSS Duration', fontsize=16, fontweight='bold')
    ax.set_xlabel('OBSS Remaining Time (slots)', fontsize=12)
    ax.set_ylabel('Probability of Switching to NPCA', fontsize=12)
    ax.set_xticklabels(switch_prob.index, rotation=45, ha='right')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add text labels on bars
    for i, v in enumerate(switch_prob):
        ax.text(i, v + 0.02, f"{v:.1%}", ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.tight_layout()
    
    output_path = "./semi_mdp_results/decision_analysis.png"
    plt.savefig(output_path, dpi=300)
    
    print(f"\n📊 분석 그래프가 다음 경로에 저장되었습니다: {output_path}")
    print("\n💡 해석: 이 그래프는 OBSS 잔여 시간이 길어질수록 모델이 NPCA로 전환하는 것을 더 선호하는지, 아니면 다른 요인에 의해 결정을 내리는지를 보여줍니다.")

if __name__ == "__main__":
    analyze_decision_log()