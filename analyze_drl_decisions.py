#!/usr/bin/env python3
"""
DRL 정책의 액션 선택 요인 분석
상태별 액션 선택 패턴과 Q값 분석을 통한 의사결정 해석
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # seaborn 없이 구현
import pandas as pd
import os
# sklearn 의존성 제거
from drl_framework.random_access import STA, Channel, Simulator
from drl_framework.network import DQN

def generate_state_samples(num_samples=1000):
    """다양한 상태 샘플 생성"""
    states = []
    
    # 상태 공간 정의
    # [obss_remain, radio_transition_time, tx_duration, cw_index]
    
    for _ in range(num_samples):
        obss_remain = np.random.randint(0, 100)  # 0~99 슬롯
        radio_transition_time = 1  # 고정값
        tx_duration = np.random.choice([33, 165])  # short/long frame
        cw_index = np.random.randint(0, 7)  # CW stage 0~6
        
        state = [obss_remain, radio_transition_time, tx_duration, cw_index]
        states.append(state)
    
    return np.array(states)

def analyze_q_values(model, states, device):
    """상태별 Q값 분석"""
    model.eval()
    
    q_values_list = []
    actions_list = []
    
    with torch.no_grad():
        for state in states:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            q_values = model(state_tensor).cpu().numpy()[0]
            action = np.argmax(q_values)
            
            q_values_list.append(q_values)
            actions_list.append(action)
    
    return np.array(q_values_list), np.array(actions_list)

def analyze_state_action_correlation(states, actions, q_values):
    """상태와 액션 선택 간의 상관관계 분석"""
    # 상태 특성 이름
    feature_names = [
        'OBSS Remaining (slots)',
        'Radio Transition Time', 
        'TX Duration (slots)',
        'CW Index'
    ]
    
    # DataFrame 생성
    df = pd.DataFrame(states, columns=feature_names)
    df['Action'] = actions
    df['Action_Name'] = df['Action'].map({0: 'Stay Primary', 1: 'Go NPCA'})
    df['Q_Primary'] = q_values[:, 0]
    df['Q_NPCA'] = q_values[:, 1]
    df['Q_Diff'] = q_values[:, 1] - q_values[:, 0]  # NPCA - Primary
    
    return df

def plot_decision_analysis(df, save_dir="./decision_analysis"):
    """의사결정 분석 시각화"""
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. 액션 선택 분포
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1-1. OBSS Remaining vs Action
    ax = axes[0, 0]
    for action, name in [(0, 'Stay Primary'), (1, 'Go NPCA')]:
        data = df[df['Action'] == action]['OBSS Remaining (slots)']
        ax.hist(data, alpha=0.7, label=name, bins=20)
    ax.set_xlabel('OBSS Remaining (slots)')
    ax.set_ylabel('Frequency')
    ax.set_title('Action Selection vs OBSS Remaining')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 1-2. TX Duration vs Action
    ax = axes[0, 1]
    tx_action_counts = df.groupby(['TX Duration (slots)', 'Action_Name']).size().unstack()
    tx_action_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('TX Duration (slots)')
    ax.set_ylabel('Count')
    ax.set_title('Action Selection vs TX Duration')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, alpha=0.3)
    
    # 1-3. CW Index vs Action
    ax = axes[0, 2]
    cw_action_counts = df.groupby(['CW Index', 'Action_Name']).size().unstack()
    cw_action_counts.plot(kind='bar', ax=ax)
    ax.set_xlabel('CW Index')
    ax.set_ylabel('Count')
    ax.set_title('Action Selection vs CW Index')
    ax.grid(True, alpha=0.3)
    
    # 2-1. Q값 차이 분포
    ax = axes[1, 0]
    ax.hist(df['Q_Diff'], bins=30, alpha=0.7, color='purple')
    ax.axvline(0, color='red', linestyle='--', label='Q_NPCA = Q_Primary')
    ax.set_xlabel('Q_NPCA - Q_Primary')
    ax.set_ylabel('Frequency')
    ax.set_title('Q-Value Difference Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2-2. OBSS Remaining vs Q값 차이
    ax = axes[1, 1]
    scatter = ax.scatter(df['OBSS Remaining (slots)'], df['Q_Diff'], 
                        c=df['Action'], cmap='RdYlBu', alpha=0.6)
    ax.axhline(0, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('OBSS Remaining (slots)')
    ax.set_ylabel('Q_NPCA - Q_Primary')
    ax.set_title('OBSS Remaining vs Q-Value Difference')
    plt.colorbar(scatter, ax=ax, label='Action (0=Primary, 1=NPCA)')
    ax.grid(True, alpha=0.3)
    
    # 2-3. 액션별 Q값 분포
    ax = axes[1, 2]
    ax.boxplot([df[df['Action'] == 0]['Q_Primary'], df[df['Action'] == 0]['Q_NPCA'],
                df[df['Action'] == 1]['Q_Primary'], df[df['Action'] == 1]['Q_NPCA']], 
               labels=['Primary\n(Stay)', 'NPCA\n(Stay)', 'Primary\n(Go)', 'NPCA\n(Go)'])
    ax.set_ylabel('Q-Value')
    ax.set_title('Q-Values by Action Selection')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/decision_analysis.png", dpi=300, bbox_inches='tight')
    plt.show()

def analyze_decision_rules(df):
    """의사결정 규칙 분석"""
    print("=" * 70)
    print("🧠 DRL 정책 의사결정 규칙 분석")
    print("=" * 70)
    
    # 1. 전체 액션 분포
    action_dist = df['Action_Name'].value_counts()
    print("📊 전체 액션 분포:")
    for action, count in action_dist.items():
        percentage = count / len(df) * 100
        print(f"  {action}: {count} ({percentage:.1f}%)")
    
    # 2. OBSS Remaining별 액션 선택 경향
    print("\n🔍 OBSS Remaining별 액션 선택 경향:")
    obss_bins = [0, 10, 30, 50, 100]
    obss_labels = ['Very Low (0-10)', 'Low (11-30)', 'Medium (31-50)', 'High (51-100)']
    
    df['OBSS_Category'] = pd.cut(df['OBSS Remaining (slots)'], 
                                bins=obss_bins, labels=obss_labels, include_lowest=True)
    
    obss_action_pct = df.groupby('OBSS_Category')['Action'].agg(['mean', 'count'])
    for category, row in obss_action_pct.iterrows():
        npca_pct = row['mean'] * 100
        print(f"  {category}: {npca_pct:.1f}% Go NPCA (n={row['count']})")
    
    # 3. TX Duration별 액션 선택
    print("\n📏 TX Duration별 액션 선택:")
    tx_action_pct = df.groupby('TX Duration (slots)')['Action'].agg(['mean', 'count'])
    for duration, row in tx_action_pct.iterrows():
        npca_pct = row['mean'] * 100
        frame_type = "Short Frame" if duration == 33 else "Long Frame"
        print(f"  {frame_type} ({duration} slots): {npca_pct:.1f}% Go NPCA (n={row['count']})")
    
    # 4. CW Index별 액션 선택
    print("\n⏱️ Contention Window별 액션 선택:")
    cw_action_pct = df.groupby('CW Index')['Action'].agg(['mean', 'count'])
    for cw, row in cw_action_pct.iterrows():
        npca_pct = row['mean'] * 100
        print(f"  CW Stage {cw}: {npca_pct:.1f}% Go NPCA (n={row['count']})")
    
    # 5. 핵심 의사결정 패턴
    print("\n🎯 핵심 의사결정 패턴:")
    
    # OBSS가 많이 남았을 때
    high_obss = df[df['OBSS Remaining (slots)'] > 50]
    if len(high_obss) > 0:
        high_obss_npca = high_obss['Action'].mean() * 100
        print(f"  - OBSS가 많이 남은 경우 (50+ slots): {high_obss_npca:.1f}% Go NPCA")
    
    # OBSS가 적게 남았을 때  
    low_obss = df[df['OBSS Remaining (slots)'] <= 10]
    if len(low_obss) > 0:
        low_obss_npca = low_obss['Action'].mean() * 100
        print(f"  - OBSS가 적게 남은 경우 (≤10 slots): {low_obss_npca:.1f}% Go NPCA")
    
    # 긴 프레임일 때
    long_frame = df[df['TX Duration (slots)'] == 165]
    if len(long_frame) > 0:
        long_frame_npca = long_frame['Action'].mean() * 100
        print(f"  - 긴 프레임 전송 시: {long_frame_npca:.1f}% Go NPCA")
    
    # 짧은 프레임일 때
    short_frame = df[df['TX Duration (slots)'] == 33]
    if len(short_frame) > 0:
        short_frame_npca = short_frame['Action'].mean() * 100
        print(f"  - 짧은 프레임 전송 시: {short_frame_npca:.1f}% Go NPCA")

def plot_heatmap_analysis(df, save_dir="./decision_analysis"):
    """히트맵을 통한 상세 분석 (matplotlib만 사용)"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. OBSS vs TX Duration 히트맵
    obss_bins = np.arange(0, 101, 10)
    obss_labels = [f"{i}-{i+9}" for i in range(0, 100, 10)]
    df['OBSS_Binned'] = pd.cut(df['OBSS Remaining (slots)'], bins=obss_bins, labels=obss_labels)
    
    heatmap_data = df.groupby(['OBSS_Binned', 'TX Duration (slots)'])['Action'].mean().unstack()
    
    ax = axes[0]
    im = ax.imshow(heatmap_data.values, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(len(heatmap_data.columns)))
    ax.set_xticklabels(heatmap_data.columns)
    ax.set_yticks(range(len(heatmap_data.index)))
    ax.set_yticklabels(heatmap_data.index)
    ax.set_title('Action Selection: OBSS Remaining vs TX Duration')
    ax.set_xlabel('TX Duration (slots)')
    ax.set_ylabel('OBSS Remaining (slots)')
    
    # 값 표시
    for i in range(len(heatmap_data.index)):
        for j in range(len(heatmap_data.columns)):
            value = heatmap_data.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center')
    
    cbar1 = plt.colorbar(im, ax=ax, label='Probability of Go NPCA')
    
    # 2. OBSS vs CW Index 히트맵  
    heatmap_data2 = df.groupby(['OBSS_Binned', 'CW Index'])['Action'].mean().unstack()
    
    ax = axes[1]
    im2 = ax.imshow(heatmap_data2.values, cmap='RdYlBu_r', aspect='auto')
    ax.set_xticks(range(len(heatmap_data2.columns)))
    ax.set_xticklabels(heatmap_data2.columns)
    ax.set_yticks(range(len(heatmap_data2.index)))
    ax.set_yticklabels(heatmap_data2.index)
    ax.set_title('Action Selection: OBSS Remaining vs CW Index')
    ax.set_xlabel('CW Index')
    ax.set_ylabel('OBSS Remaining (slots)')
    
    # 값 표시
    for i in range(len(heatmap_data2.index)):
        for j in range(len(heatmap_data2.columns)):
            value = heatmap_data2.iloc[i, j]
            if not np.isnan(value):
                ax.text(j, i, f'{value:.2f}', ha='center', va='center')
    
    cbar2 = plt.colorbar(im2, ax=ax, label='Probability of Go NPCA')
    
    plt.tight_layout()
    plt.savefig(f"{save_dir}/decision_heatmaps.png", dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """메인 함수"""
    print("=" * 70)
    print("🔍 DRL 정책 액션 선택 요인 분석")
    print("=" * 70)
    
    model_path = "./semi_mdp_results/semi_mdp_model.pth"
    save_dir = "./decision_analysis"
    
    if not os.path.exists(model_path):
        print(f"❌ 모델 파일이 없습니다: {model_path}")
        return
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else 
                         "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🖥️ 사용 디바이스: {device}")
    
    # 모델 로드
    print("🤖 DRL 모델 로딩 중...")
    model = DQN(n_observations=4, n_actions=2).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['policy_net_state_dict'])
    model.eval()
    
    # 상태 샘플 생성
    print("📊 상태 샘플 생성 중...")
    num_samples = 2000
    states = generate_state_samples(num_samples)
    print(f"   생성된 샘플 수: {num_samples}")
    
    # Q값 분석
    print("🧮 Q값 및 액션 분석 중...")
    q_values, actions = analyze_q_values(model, states, device)
    
    # 상관관계 분석
    print("📈 상태-액션 상관관계 분석 중...")
    df = analyze_state_action_correlation(states, actions, q_values)
    
    # 의사결정 규칙 분석
    analyze_decision_rules(df)
    
    # 시각화
    print(f"\n📊 결과 시각화 중...")
    plot_decision_analysis(df, save_dir)
    plot_heatmap_analysis(df, save_dir)
    
    # 데이터 저장
    df.to_csv(f"{save_dir}/decision_analysis_data.csv", index=False)
    
    print(f"\n✅ 분석 완료!")
    print(f"📁 결과 저장 위치: {save_dir}/")
    print("   - decision_analysis.png: 의사결정 패턴 분석")
    print("   - decision_heatmaps.png: 상태별 액션 선택 히트맵")
    print("   - decision_analysis_data.csv: 분석 데이터")

if __name__ == "__main__":
    main()