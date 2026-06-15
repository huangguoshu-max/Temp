# -*- coding: utf-8 -*-
"""
L3信號判定引擎 - 依據 §6 規範
純函數，時序因果箭頭驗證，開盤異常脈衝時間懲罰，修正波段去重 Bug 1。
"""
import pandas as pd
import numpy as np
import json

class SignalEngineV7:
    @classmethod
    def evaluate_signals(cls, factor_df: pd.DataFrame, 
                         regime_config: dict, 
                         is_intraday: bool, 
                         m_minutes: int) -> pd.DataFrame:
        """
        輸入L2因子矩陣，動態收斂判定，輸出離散信號與幾何證據引用鏈
        """
        df = factor_df.copy()
        
        theta_panic = regime_config.get("theta_panic", 1.2)
        theta_hollow = regime_config.get("theta_hollow", 0.3333)
        base_theta_buy = regime_config.get("theta_buy", 1.5)
        
        # 1. §6.4 開盤前30分鐘時間衰減懲罰算子
        if is_intraday and m_minutes <= 30:
            theta_buy = base_theta_buy * (1.0 + 0.3 * ((30.0 - m_minutes) / 30.0))
        else:
            theta_buy = base_theta_buy

        # 2. 因果狀態機脈衝事件鎖定
        df['panic_triggered'] = np.where((df['inside_window'] == 1) & (df['low'] < df['locked_base_line']) & (df['rvr'] >= theta_panic), 1, 0)
        df['hollow_triggered'] = np.where((df['inside_window'] == 1) & (df['close'] < df['locked_base_line']) & (df['rvr'] <= theta_hollow), 1, 0)

        # 跨浪清洗：把早於本次破位起點的舊指針物理置空 (§2.2.3)
        df['raw_panic_idx'] = df['global_idx'].where(df['panic_triggered'] == 1)
        df['last_panic_idx'] = df.groupby('ticker')['raw_panic_idx'].ffill()
        df['last_panic_idx'] = np.where(df['last_panic_idx'] < df['last_break_idx'], np.nan, df['last_panic_idx'])

        df['raw_hollow_idx'] = df['global_idx'].where(df['hollow_triggered'] == 1)
        df['last_hollow_idx'] = df.groupby('ticker')['raw_hollow_idx'].ffill()
        df['last_hollow_idx'] = np.where(df['last_hollow_idx'] < df['last_break_idx'], np.nan, df['last_hollow_idx'])

        # 3. 驗證時間箭頭不可逆性：Panic 必須早於或等於 Hollow
        df['sequence_valid'] = np.where(
            (df['inside_window'] == 1) & 
            (df['last_panic_idx'].notna()) & 
            (df['last_hollow_idx'].notna()) & 
            (df['last_hollow_idx'] >= df['last_panic_idx']), 1, 0
        )

        # 4. §6.1.1 結構安全空間限制 (合併報告版上限墊)
        df['structure_safe'] = np.where((df['close'] > df['locked_base_line']) & (df['close'] < df['locked_base_line'] * 1.25), 1, 0)

        df['close_shift1'] = df.groupby('ticker')['close'].shift(1)
        df['locked_base_line_shift1'] = df.groupby('ticker')['locked_base_line'].shift(1)
        df['sequence_valid_shift1'] = df.groupby('ticker')['sequence_valid'].shift(1)

        # 原始突破判定條件串聯与門
        df['buy_trigger_raw'] = np.where(
            (df['structure_safe'] == 1) &
            (df['close'] > df['locked_base_line']) &
            (df['close_shift1'] <= df['locked_base_line_shift1']) &
            (df['rvr'] > theta_buy) &
            ((df['close'] - df['vwap']) / df['vwap'] > 0) & # 站穩分時均線
            (df['sequence_valid_shift1'] == 1) &
            (df['is_right_side'] == 1), 1, 0
        )

        # 🛠️ §6.2.4 修正首發去重 Bug 1：採取 標的*當前破位浪雙鍵分組
        df['is_buy_idx'] = df['global_idx'].where(df['buy_trigger_raw'] == 1)
        df['first_buy_idx_in_wave'] = df.groupby(['ticker', 'last_break_idx'])['is_buy_idx'].transform('min')
        df['buy_trigger'] = np.where((df['global_idx'] == df['first_buy_idx_in_wave']) & (df['first_buy_idx_in_wave'].notna()), 1, 0)

        # 5. §6.3.2 狀態式下穿止損線監控
        df['stop_triggered'] = np.where(df['close'] < df['stop_loss_line'], -1, 0)

        # 收束離散決策狀態碼
        df['signal'] = np.where(df['buy_trigger'] == 1, 1, np.where(df['stop_triggered'] == -1, -1, 0))
        
        # 6. 生成可追溯證據 JSON 引用鏈
        df['evidence_refs'] = df.apply(lambda r: json.dumps({
            "close": float(r['close']), "vwap": float(r['vwap']), "rvr": float(r['rvr']),
            "locked_line": float(r['locked_base_line']) if pd.notna(r['locked_base_line']) else 0.0,
            "stop_line": float(r['stop_loss_line'])
        }), axis=1)

        # 剔除臨時變量，交付標準長表契約
        return df[['ticker', 'date', 'signal', 'evidence_refs']].reset_index(drop=True)