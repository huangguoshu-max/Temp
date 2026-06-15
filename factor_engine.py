# -*- coding: utf-8 -*-
"""
L2因子計算引擎 - 依據 §5 規範
純函數，無I/O，隔離 Live Bar 投影，解決 RVR 係數對消 Bug 3。
"""
import pandas as pd
import numpy as np

class FactorEngineV7:
    @classmethod
    def generate_factors(cls, aligned_df: pd.DataFrame, 
                         is_intraday: bool, 
                         m_minutes: int, 
                         t_total: int = 240) -> pd.DataFrame:
        """
        輸入歸一化 facts 矩陣，向量化輸出幾何坐標與因子
        """
        df = aligned_df.copy()
        # 強制全局時空索排序，防範未來函數 (§4.4.3)
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        df['global_idx'] = df.index

        # 1. 🛠️ §5.1.1 修正量能投影 Bug 3：只對當前最後一行（Live Bar）單點執行投影
        df['working_volume'] = df['volume'].astype(float)
        if is_intraday:
            projection_factor = float(t_total) / max(m_minutes, 1)
            # 定位每隻股票時間序列的最後一個行索引
            last_bar_mask = df.groupby('ticker')['global_idx'].transform('max')
            df['is_live_bar'] = np.where(df['global_idx'] == last_bar_mask, 1, 0)
            
            df['working_volume'] = np.where(
                df['is_live_bar'] == 1, 
                df['volume'] * projection_factor, 
                df['working_volume']
            )
            df.drop(columns=['is_live_bar'], inplace=True)

        # 精算基於歷史實際全天量的 20日均量基準与相對量比 (RVR)
        df['vol_ma20'] = df.groupby('ticker')['working_volume'].transform(lambda x: x.rolling(20).mean())
        df['rvr'] = df['working_volume'] / df['vol_ma20']

        # 2. §5.2.1 幾何底線與焊死頸線
        df['ll_base'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(60).min().shift(1))
        df['low_shift1'] = df.groupby('ticker')['low'].shift(1)
        df['ll_base_shift1'] = df.groupby('ticker')['ll_base'].shift(1)
        
        # 首次擊穿脈衝標記
        df['is_initial_break'] = np.where(
            (df['low'] < df['ll_base']) & (df['low_shift1'] >= df['ll_base_shift1']), 1, 0
        )
        
        # 頸線焊死在前向填充中，拒絕隨價格單邊陰跌下移
        df['locked_base_line'] = df['ll_base'].where(df['is_initial_break'] == 1)
        df['locked_base_line'] = df.groupby('ticker')['locked_base_line'].ffill()
        
        # 記錄本次破位浪起點指針
        df['last_break_idx'] = df['global_idx'].where(df['is_initial_break'] == 1)
        df['last_break_idx'] = df.groupby('ticker')['last_break_idx'].ffill()
        df['days_since_break'] = df['global_idx'] - df['last_break_idx']
        df['inside_window'] = np.where((df['days_since_break'] <= 30) & (df['last_break_idx'].notna()), 1, 0)

        # 3. 🛠️ §5.5 修正幾何互斥 Bug 2：採取方案 A 均線降維與斜率解耦
        df['ma20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())
        df['ma60'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(60).mean())
        df['ma60_shift3'] = df.groupby('ticker')['ma60'].shift(3)
        
        df['ma60_is_turning'] = np.where(df['ma60'] >= df['ma60_shift3'], 1, 0)
        df['is_right_side'] = np.where((df['close'] > df['ma20']) & (df['ma60_is_turning'] == 1), 1, 0)

        # 4. §5.6 蔡森極限動態平滑技術防守線
        df['low_min_20'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(20).min())
        df['stop_loss_line'] = df.groupby('ticker')['low_min_20'].transform(lambda x: x.rolling(5).mean())

        # 清理歷史清洗衍生中間列
        df.drop(columns=['low_shift1', 'll_base_shift1', 'ma60_shift3', 'low_min_20'], inplace=True)
        return df