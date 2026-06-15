# -*- coding: utf-8 -*-
"""
L1事實層准入網關 - 依據 §4.3 & §4.4 規範
100% 攔截分母單位錯算，利用數學不變性自檢，防止髒數據污染 L0 存儲。
"""
import pandas as pd
import numpy as np

class L1DataGatekeeper:
    REQUIRED_SCHEMA = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover']

    @classmethod
    def align_and_verify(cls, raw_df: pd.DataFrame, source_tag: str) -> pd.DataFrame:
        """
        強制 Schema 審計與單位歸一化
        A股體量統一換算為: 股 / 元
        """
        # 1. 鐵血 Schema 校驗
        for col in cls.REQUIRED_SCHEMA:
            if col not in raw_df.columns:
                raise ValueError(f"[L1網關熔斷] 缺少核心 facts 欄位: '{col}'")

        df = raw_df[cls.REQUIRED_SCHEMA].copy()
        source_tag = source_tag.lower()

        # 2. §4.3.3 跨源單位對齊
        if source_tag == "tencent_a":
            df['volume'] = df['volume'] * 100          # 手 -> 股
            df['turnover'] = df['turnover'] * 10000.0   # 萬元 -> 元
        elif source_tag == "akshare_a":
            df['volume'] = df['volume'] * 100          # 手 -> 股
        elif source_tag == "tencent_hk":
            pass # 港股原始口徑即為 股 / 元
        else:
            raise NotImplementedError(f"[L1網關熔斷] 未註冊數據源: {source_tag}")

        # 3. 派生當前分時真 VWAP
        df['volume_safe'] = df['volume'].replace(0, np.nan)
        df['vwap'] = df['turnover'] / df['volume_safe']
        df['vwap'] = df['vwap'].fillna(df['close'])

        # 4. §4.3.1 物理幾何不變性硬核審計 (防範隱性髒數據)
        # 容許前復權帶來的 0.5% 浮動誤差
        vwap_min = df['low'] * 0.995
        vwap_max = df['high'] * 1.005
        
        out_of_bounds = (df['vwap'] < vwap_min) | (df['vwap'] > vwap_max)
        if out_of_bounds.any():
            corrupt_tickers = df.loc[out_of_bounds, 'ticker'].unique()
            raise AssertionError(f"[L1不變性崩塌] 標的 {corrupt_tickers} 算出 VWAP 嚴重越界，觸發單位錯算斷言！")

        df.drop(columns=['volume_safe'], inplace=True)
        return df