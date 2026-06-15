# -*- coding: utf-8 -*-
"""
OpenClaw調度調度中心 - 依據 §3 & §9 規範
封死跨層污染，實現單向控制流，持久化Parquet事實庫，支持心跳發射。
"""
import pandas as pd
import numpy as np
import time
from datetime import datetime
from config import SystemConfig
from gatekeeper import L1DataGatekeeper
from factor_engine import FactorEngineV7
from signal_engine import SignalEngineV7

class OpenClawOrchestrator:
    def __init__(self, workspace_dir: str):
        self.workspace_dir = workspace_dir
        self.l0_parquet_path = f"{workspace_dir}/L0_bedrock_store.parquet"

    def execute_live_pipeline(self, raw_market_snapshot: pd.DataFrame, 
                              market_type: str, 
                              source_tag: str, 
                              m_minutes: int):
        """
        執行盤中實時管線 (15分鐘定時任務核心)
        """
        # 1. 啟動環境權威性硬性自檢 (§11.1.1)
        assert pd.__version__ == "3.0.3", f"[運維熔斷] Pandas版本衝突: {pd.__version__}"
        
        # 2. L1事實層洗滌與單位對齊
        live_facts = L1DataGatekeeper.align_and_verify(raw_market_snapshot, source_tag)
        live_facts['date'] = datetime.now().strftime('%Y-%m-%d')
        
        # 3. 讀取 L0 歷史存儲 Parquet 增量合流
        try:
            history_df = pd.read_parquet(self.l0_parquet_path)
            # 排除可能重複的今日歷史行，確保冪等性
            history_df = history_df[history_df['date'] < live_facts['date'].iloc[0]]
            full_matrix = pd.concat([history_df, live_facts], ignore_index=True)
        except FileNotFoundError:
            # P1 生死閘門警示：無歷史緩存則自動退化降級 (§3.2.4)
            print("⚠️ [數據級降級] L0 Parquet基石庫不存在，僅輸出報價，拒絕噴射信號。")
            return live_facts
        
        # 4. 注入 L2 因子引擎 (純函數)
        t_total = SystemConfig.MARKET_LIMITS[market_type]["t_total"]
        factored_matrix = FactorEngineV7.generate_factors(
            full_matrix, is_intraday=True, m_minutes=m_minutes, t_total=t_total
        )
        
        # 5. 注入 L3 信號引擎 (純函數)
        signal_matrix = SignalEngineV7.evaluate_signals(
            factored_matrix, SystemConfig.DEFAULT_REGIME, is_intraday=True, m_minutes=m_minutes
        )
        
        # 6. 🛠️ §9.1.2 運維落盤：更新定時任務心跳戳，防止靜默死亡
        with open(f"{self.workspace_dir}/monitor_heartbeat.ts", "w") as f:
            f.write(f"last_run_ts: {int(time.time())}")
            
        # 提取最新一天的實時判定結果交付給 L4 渲染層
        latest_mask = signal_matrix.groupby('ticker')['date'].transform('max')
        latest_signals = signal_matrix[signal_matrix['date'] == latest_mask]
        
        print(f"✅ [管線執行完畢] 截面信號已收斂。當前追蹤標的數: {len(latest_signals)}")
        return latest_signals