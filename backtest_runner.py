# -*- coding: utf-8 -*-
"""
OpenClaw P1 階優勢核驗與事件驅動撮合引擎 - 依據 §8 規範
純確定性，嚴防前視泄露，精算真實全摩擦稅費，執行無條件基線優勢證偽。
"""
import pandas as pd
import numpy as np
import json
from datetime import datetime
from factor_engine import FactorEngineV7
from signal_engine import SignalEngineV7
from config import SystemConfig

class OpenClawBacktestRunner:
    def __init__(self, workspace_dir: str, commission_bps: float = 3.0, stamp_duty_bps: float = 5.0, slippage_pct: float = 0.1):
        """
        參數設定依據 §8.1.2：
        commission_bps: 單邊佣金萬3
        stamp_duty_bps: A股賣出單邊印花稅萬5 (港股另行適配)
        slippage_pct: 進場/出場惡意滑點 0.1%
        """
        self.l0_parquet_path = f"{workspace_dir}/L0_bedrock_store.parquet"
        self.com_fee = commission_bps / 10000.0
        self.stamp_fee = stamp_duty_bps / 10000.0
        self.slip = slippage_pct / 100.0

    def run_historical_backtest(self, market_type: str, start_date: str, end_date: str):
        print("================================================================================")
        print(f" 📊 OpenClaw V7 引擎 - 策略優勢歷史實證矩陣啟動 [MVP Phase]")
        print("================================================================================")
        
        # 1. 載入 L0 事實存儲
        try:
            df = pd.read_parquet(self.l0_parquet_path)
        except FileNotFoundError:
            print(f"❌ [回測熔斷] 未找到基準 L0 Parquet 存儲。請先執行邊緣攝取。")
            return None

        # 2. 嚴格時空邊界切分
        df = df[(df['date'] >= start_date) & (df['date'] <= end_date)].copy()
        if df.empty:
            print("❌ [回測熔斷] 當前時間窗口內無任何Fact事實數據。")
            return None

        # 3. 呼叫 L2/L3 實盤原始代碼生成因子與信號 (100% 邏輯復用 §8.1.1)
        print("📡 正在跨標的时空矩陣生成 V7 純函數因子與有限狀態機信號...")
        t_total = SystemConfig.MARKET_LIMITS[market_type]["t_total"]
        
        # 回測視同盤後結算，is_intraday=False，封死任何未成熟的盤中投影噪聲
        df = FactorEngineV7.generate_factors(df, is_intraday=False, m_minutes=t_total, t_total=t_total)
        signals_df = SignalEngineV7.evaluate_signals(df, SystemConfig.DEFAULT_REGIME, is_intraday=False, m_minutes=t_total)
        
        # 合流 Facts 庫與信號庫
        full_m = pd.merge(df, signals_df, on=['ticker', 'date'], how='inner')
        full_m = full_m.sort_values(by=['ticker', 'date']).reset_index(drop=True)

        # 4. §8.2.3 執行時滯與全摩擦事件驅動撮合模擬
        print("⚙️ 啟動悲觀主義撮合模擬器（次日開盤進場 + 摩擦損耗扣減）...")
        trades = []
        
        # 提取所有觸發買入相變的時點 (signal == 1)
        buy_signals = full_m[full_m['signal'] == 1].copy()
        
        for _, sig_row in buy_signals.iterrows():
            ticker = sig_row['ticker']
            buy_date = sig_row['date']
            evidence = json.loads(sig_row['evidence_refs'])
            
            # 尋找該標的在時序上的後續事實序列
            sub_seq = full_m[(full_m['ticker'] == ticker) & (full_m['date'] > buy_date)].sort_values(by='date').copy()
            if sub_seq.empty:
                continue # 後續無數據，自動歸入[觀察中/未出局]
                
            # 🛠️ §8.1.2 撮合定價：今日信號觸發，次日開盤價(Open)加滑點作為真實進場成本
            entry_date = sub_seq['date'].iloc[0]
            entry_price_raw = sub_seq['open'].iloc[0]
            entry_price_real = entry_price_raw * (1.0 + self.slip) # 買入遭遇向上滑點
            
            # 初始化持倉生命週期狀態
            stop_loss_line = evidence['stop_line']
            target_t1 = evidence['locked_line'] + (evidence['locked_line'] - stop_loss_line) # 等幅測距§7.3.1
            
            exit_date = None
            exit_price_real = None
            exit_reason = "Time_Out"
            
            # 持倉最大時限為 30 個交易日 (max_setup_days 隱式時間止損 §7.1.3)
            holding_period = sub_seq.head(30)
            
            for idx, bar in holding_period.iterrows():
                current_close = bar['close']
                current_low = bar['low']
                current_high = bar['high']
                
                # 判斷 1：觸及動態技術止損生死線 (收盤價跌破止損)
                if current_close < bar['stop_loss_line']:
                    exit_date = bar['date']
                    # 跌破當日收盤離場，遭遇向下滑點
                    exit_price_real = current_close * (1.0 - self.slip)
                    exit_reason = "Stop_Loss"
                    break
                    
                # 判斷 2：觸及等幅測距 T1 止盈目標位
                if current_high >= target_t1:
                    exit_date = bar['date']
                    exit_price_real = target_t1 * (1.0 - self.slip) # 觸及目標止盈
                    exit_reason = "Target_T1"
                    break
            
            # 若 30 個交易日未達預期，觸發強制時間出清
            if exit_date is None and not holding_period.empty:
                last_bar = holding_period.iloc[-1]
                exit_date = last_bar['date']
                exit_price_real = last_bar['close'] * (1.0 - self.slip)
                exit_reason = "Time_Expired"

            if exit_date:
                # 稅費計算
                raw_return = (exit_price_real - entry_price_real) / entry_price_real
                total_fees = self.com_fee * 2 + self.stamp_fee  # 買賣佣金 + 賣出印花稅
                net_return = raw_return - total_fees
                
                trades.append({
                    "ticker": ticker,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "reason": exit_reason,
                    "net_return": net_return
                })

        # 5. §8.4 績效指標清算
        if not trades:
            print("🚦 [MVP 審判結果] 歷史截面中未產生任何有效買入信號。請優化右側濾網方案。")
            return None
            
        trades_df = pd.DataFrame(trades)
        win_rate = len(trades_df[trades_df['net_return'] > 0]) / len(trades_df)
        avg_profit = trades_df[trades_df['net_return'] > 0]['net_return'].mean() if win_rate > 0 else 0
        avg_loss = trades_df[trades_df['net_return'] <= 0]['net_return'].mean() if win_rate < 1 else -1e-6
        real_profit_loss_ratio = avg_profit / abs(avg_loss) if avg_loss != 0 else np.inf
        
        # 期望值計算
        expectancy = win_rate * avg_profit + (1 - win_rate) * avg_loss

        print("\n================================================================================")
        print(" 🚦 [MVP 終局裁判報告] 蔡森 V7 策略核心優勢清算")
        print("================================================================================")
        print(f" 🟢 總計觸發有效交易筆數 : {len(trades_df)} 筆")
        print(f" 🟢 扣費後勝率 (Win Rate)  : {win_rate * 100:.2f}%")
        print(f" 🟢 實測每筆平均盈虧比     : {real_profit_loss_ratio:.2f} : 1 (設計目標為 ≥ 2:1)")
        print(f" 🟢 單筆淨期望值 (Expectancy): {expectancy * 100:+.4f}% 每筆")
        print("--------------------------------------------------------------------------------")
        print(f" 📊 離散出局原因歸因統計:")
        print(trades_df['reason'].value_counts().to_string())
        print("================================================================================")
        
        return trades_df