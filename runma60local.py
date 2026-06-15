# -*- coding: utf-8 -*-
"""
OpenClaw 系統 - 本地住宅 IP 邊緣採集與 V7 純淨內核精算矩陣
執行命令: python runma60local.py
"""
import os
import sys
import time
import json
import urllib.request
from datetime import datetime
import pandas as pd
import numpy as np

# 鎖定大綱指定的硬核版本環境，防止 API 漂移 (§11.1.1)
try:
    import akshare as ak
except ImportError:
    print("❌ [環境熔斷] 本地未安裝 akshare，請先執行: pip install akshare pandas numpy pyarrow")
    sys.exit(1)

# ==============================================================================
# 1. 配置與標的池矩陣 (§9.2)
# ==============================================================================
TICKERS_A = ["688041.SH", "603986.SH", "002185.SZ", "601138.SH", "688981.SH"]
WORKSPACE_DIR = os.path.join(os.path.expanduser("~"), ".openclaw", "workspace")
L0_PARQUET_PATH = os.path.join(WORKSPACE_DIR, "L0_bedrock_store.parquet")

os.makedirs(WORKSPACE_DIR, exist_ok=True)

# ==============================================================================
# 2. L1 事實層：數據准入與幾何不變性自檢網關 (§4.3 & §4.4)
# ==============================================================================
class L1DataGatekeeper:
    REQUIRED_COLUMNS = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover']

    @classmethod
    def clean_and_align_akshare(cls, raw_df: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """清洗本地 AkShare 歷史事實，換算單位為 股/元"""
        df = raw_df.copy()
        df['ticker'] = ticker
        df = df.rename(columns={
            "日期": "date", "开盘": "open", "最高": "high", 
            "最低": "low", "收盘": "close", "成交量": "volume", "成交额": "turnover"
        })
        # AkShare A股成交量單位為「手」，強制對齊契約轉為「股」 (§1.4.3)
        df['volume'] = df['volume'] * 100
        df['turnover'] = df['turnover'].astype(float)
        
        df = df[cls.REQUIRED_COLUMNS]
        
        # 計算日級真 VWAP 並執行物理邊界斷言
        df['vwap'] = df['turnover'] / df['volume'].replace(0, np.nan)
        df['vwap'] = df['vwap'].fillna(df['close'])
        
        # 剛性幾何不變性自檢 (容許復權 0.5% 誤差)
        out_of_bounds = (df['vwap'] < df['low'] * 0.995) | (df['vwap'] > df['high'] * 1.005)
        if out_of_bounds.any():
            raise AssertionError(f"[L1網關熔斷] 標的 {ticker} 歷史數據爆發 VWAP 幾何越界，拒絕入庫！")
            
        return df

    @classmethod
    def fetch_tencent_live_snapshot(cls, tickers: list) -> pd.DataFrame:
        """從騰訊 API 採集零延時盤中快照，利用不變性防範魔術索引錯位 (§4.1.1 & §4.3.1)"""
        live_rows = []
        # 轉為騰訊格式: sh688041, sz002185
        api_symbols = [f"{t.split('.')[1].lower()}{t.split('.')[0]}" for t in tickers]
        url = f"http://qt.gtimg.cn/q={','.join(api_symbols)}"
        
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req, timeout=5) as response:
                content = response.read().decode('gbk') # 騰訊接口硬性指定 gbk 解碼 (§4.3.4)
        except Exception as e:
            print(f"⚠️ [網路抖動] 騰訊實時快照獲取失敗: {e}，管線將降級使用全歷史數據。")
            return pd.DataFrame()

        for line in content.split('\n'):
            if not line.strip() or '=' not in line: continue
            parts = line.split('~')
            if len(parts) < 40: continue
            
            # 逆向解析並與標準長表字段映射 (§4.3.2)
            raw_symbol = line.split('=')[0].split('_')[-1]
            ticker = f"{raw_symbol[2:]}.{'SH' if raw_symbol[:2]=='sh' else 'SZ'}"
            
            close = float(parts[3])
            high = float(parts[33])
            low = float(parts[34])
            volume_hand = float(parts[36])      # 騰訊 A股盤中原始口徑為手
            turnover_wan = float(parts[37])     # 騰訊 A股盤中原始口徑為萬元
            
            # 依據憲法字典強制標準化對齊為 股/元
            volume_share = volume_hand * 100
            turnover_yuan = turnover_wan * 10000.0
            
            # 執行不變性自檢，防止騰訊接口多版本字段順序錯位
            vwap = turnover_yuan / volume_share if volume_share > 0 else close
            if not (low * 0.99 <= vwap <= high * 1.01) and volume_share > 0:
                print(f"⚠️ [不變性攔截] 標的 {ticker} 盤中快照數據校驗未通過，標髒剔除。")
                continue
                
            live_rows.append({
                'ticker': ticker, 'date': datetime.now().strftime('%Y-%m-%d'),
                'open': float(parts[5]), 'high': high, 'low': low, 'close': close,
                'volume': volume_share, 'turnover': turnover_yuan, 'vwap': vwap
            })
            
        return pd.DataFrame(live_rows)

# ==============================================================================
# 3. L2 因子計算層：純函數矩陣變換矩陣 (§5)
# ==============================================================================
class FactorEngineV7:
    @classmethod
    def generate_factors(cls, full_df: pd.DataFrame, is_intraday: bool, m_minutes: int) -> pd.DataFrame:
        df = full_df.copy()
        df = df.sort_values(by=['ticker', 'date']).reset_index(drop=True)
        df['global_idx'] = df.index

        # --- 🛠️ 修正 Bug 3：Live Bar 隔離投影算法 (§5.1.1) ---
        df['working_volume'] = df['volume'].astype(float)
        if is_intraday and not df.empty:
            # 歷史行算分母時排除今日半截量
            df['vol_ma20_hist'] = df.groupby('ticker')['working_volume'].transform(lambda x: x.shift(1).rolling(20).mean())
            
            # 定位 Live Bar 行索引
            last_idx_mask = df.groupby('ticker')['global_idx'].transform('max')
            df['is_live_bar'] = np.where(df['global_idx'] == last_idx_mask, 1, 0)
            
            # 僅對 Live Bar 行分子執行量能非均勻線性投影 (240分鐘)
            projected_factor = 240.0 / max(m_minutes, 1)
            df['projected_vol'] = df['volume'] * projected_factor
            
            df['rvr'] = np.where(
                df['is_live_bar'] == 1,
                df['projected_vol'] / df['vol_ma20_hist'],
                df['working_volume'] / df.groupby('ticker')['working_volume'].transform(lambda x: x.rolling(20).mean())
            )
            df.drop(columns=['vol_ma20_hist', 'is_live_bar', 'projected_vol'], inplace=True)
        else:
            df['vol_ma20'] = df.groupby('ticker')['working_volume'].transform(lambda x: x.rolling(20).mean())
            df['rvr'] = df['working_volume'] / df['vol_ma20']
            df.drop(columns=['vol_ma20'], inplace=True)

        # --- 🛠️ 幾何底線與焊死基準線 (§5.2.1) ---
        df['ll_base'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(60).min().shift(1))
        df['low_shift1'] = df.groupby('ticker')['low'].shift(1)
        df['ll_base_shift1'] = df.groupby('ticker')['ll_base'].shift(1)
        
        # 首次擊穿事件標記 (去重)
        df['is_initial_break'] = np.where((df['low'] < df['ll_base']) & (df['low_shift1'] >= df['ll_base_shift1']), 1, 0)
        
        # 坐標焊死，拒絕隨價格陰跌下移
        df['locked_base_line'] = df['ll_base'].where(df['is_initial_break'] == 1)
        df['locked_base_line'] = df.groupby('ticker')['locked_base_line'].ffill()
        
        df['last_break_idx'] = df['global_idx'].where(df['is_initial_break'] == 1)
        df['last_break_idx'] = df.groupby('ticker')['last_break_idx'].ffill()
        df['inside_window'] = np.where((df['global_idx'] - df['last_break_idx'] <= 30) & (df['last_break_idx'].notna()), 1, 0)

        # --- 🛠️ 修正 Bug 2：方案 A 均線降維與斜率解耦 (§5.5) ---
        df['ma20'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(20).mean())
        df['ma60'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(60).mean())
        df['ma60_shift3'] = df.groupby('ticker')['ma60'].shift(3)
        
        df['ma60_is_turning'] = np.where(df['ma60'] >= df['ma60_shift3'], 1, 0)
        df['is_right_side'] = np.where((df['close'] > df['ma20']) & (df['ma60_is_turning'] == 1), 1, 0)

        # 蔡森技術防守線平滑
        df['low_min_20'] = df.groupby('ticker')['low'].transform(lambda x: x.rolling(20).min())
        df['stop_loss_line'] = df.groupby('ticker')['low_min_20'].transform(lambda x: x.rolling(5).mean())
        
        # 衍生計算當前價 vs MA60 空間偏離度
        df['space_bias_ma60'] = (df['close'] - df['ma60']) / df['ma60'] * 100.0

        df.drop(columns=['low_shift1', 'll_base_shift1', 'ma60_shift3', 'low_min_20'], inplace=True)
        return df

# ==============================================================================
# 4. L3 信號判定層：狀態機與波段去重合流 (§6)
# ==============================================================================
class SignalEngineV7:
    @classmethod
    def evaluate_signals(cls, factor_df: pd.DataFrame) -> pd.DataFrame:
        df = factor_df.copy()
        
        # 狀態機時序事件標記
        df['panic_triggered'] = np.where((df['inside_window'] == 1) & (df['low'] < df['locked_base_line']) & (df['rvr'] >= 1.2), 1, 0)
        df['hollow_triggered'] = np.where((df['inside_window'] == 1) & (df['close'] < df['locked_base_line']) & (df['rvr'] <= 0.3333), 1, 0)

        df['raw_panic_idx'] = df['global_idx'].where(df['panic_triggered'] == 1)
        df['last_panic_idx'] = df.groupby('ticker')['raw_panic_idx'].ffill()
        df['last_panic_idx'] = np.where(df['last_panic_idx'] < df['last_break_idx'], np.nan, df['last_panic_idx'])

        df['raw_hollow_idx'] = df['global_idx'].where(df['hollow_triggered'] == 1)
        df['last_hollow_idx'] = df.groupby('ticker')['raw_hollow_idx'].ffill()
        df['last_hollow_idx'] = np.where(df['last_hollow_idx'] < df['last_break_idx'], np.nan, df['last_hollow_idx'])

        # 驗證時間不可逆因果鏈
        df['sequence_valid'] = np.where(
            (df['inside_window'] == 1) & (df['last_panic_idx'].notna()) & (df['last_hollow_idx'].notna()) & (df['last_hollow_idx'] >= df['last_panic_idx']), 1, 0
        )

        df['structure_safe'] = np.where((df['close'] > df['locked_base_line']) & (df['close'] < df['locked_base_line'] * 1.25), 1, 0)
        df['close_shift1'] = df.groupby('ticker')['close'].shift(1)
        df['locked_base_line_shift1'] = df.groupby('ticker')['locked_base_line'].shift(1)
        df['sequence_valid_shift1'] = df.groupby('ticker')['sequence_valid'].shift(1)

        # 原始信號與門判定
        df['buy_trigger_raw'] = np.where(
            (df['structure_safe'] == 1) & (df['close'] > df['locked_base_line']) &
            (df['close_shift1'] <= df['locked_base_line_shift1']) & (df['rvr'] > 1.5) &
            (df['close'] >= df['vwap']) & (df['sequence_valid_shift1'] == 1) & (df['is_right_side'] == 1), 1, 0
        )

        # --- 🛠️ 修正 Bug 1：波段首發雙鍵去重過濾 (§6.2.4) ---
        df['is_buy_idx'] = df['global_idx'].where(df['buy_trigger_raw'] == 1)
        df['first_buy_idx_in_wave'] = df.groupby(['ticker', 'last_break_idx'])['is_buy_idx'].transform('min')
        df['buy_trigger'] = np.where((df['global_idx'] == df['first_buy_idx_in_wave']) & (df['first_buy_idx_in_wave'].notna()), 1, 0)

        return df

# ==============================================================================
# 5. 邊緣主控制流：增量拉取、精算與物理渲染 (§3 & §10)
# ==============================================================================
def run_edge_pipeline():
    print("================================================================================")
    print(f" 📊 蔡森 V7 引擎 - 本地歷史數據邊緣採集矩陣 (PC本地運行結果)")
    print(f" 📅 執行時間：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} GMT+8")
    print("================================================================================")

    # 1. 增量獲取或初始化本地歷史存儲庫 (§4.5)
    all_history_blocks = []
    
    for ticker in TICKERS_A:
        pure_symbol = ticker.split('.')[0]
        print(f"📡 正在獲取 {ticker} (AkShare 歷史 K 線數據)...")
        try:
            # 本地住宅IP調用，前復權鎖死，預熱長度設為180天以確保MA60精準度(§5.3.1)
            raw_hist = ak.stock_zh_a_hist(symbol=pure_symbol, period="daily", adjust="qfq")
            clean_hist = L1DataGatekeeper.clean_and_align_akshare(raw_hist, ticker)
            all_history_blocks.append(clean_hist)
            time.sleep(0.5) # 自我限速，優雅應對頻率限制 (§11.2.2)
        except Exception as e:
            print(f"  ❌ {ticker} 採集失敗: {e}")
            
    if not all_history_blocks:
        print("[熔斷] 本地無任何歷史 facts 可用，退出管線。")
        return

    master_history = pd.concat(all_history_blocks, ignore_index=True)
    
    # 2. 獲取盤中實時快照流
    print("--------------------------------------------------------------------------------")
    print("📡 正在獲取盤中實時五檔流快照 (qt.gtimg.cn)...")
    live_snapshot = L1DataGatekeeper.fetch_tencent_live_snapshot(TICKERS_A)
    
    is_intraday = not live_snapshot.empty
    m_minutes = 240 # 默認盤後全天
    
    if is_intraday:
        # 計算當前已交易分鐘數，扣除午休 9:30-11:30 (120分鐘) (§2.4.1)
        now = datetime.now()
        curr_min = now.hour * 60 + now.minute
        if curr_min <= 690: # 11:30前
            m_minutes = max(0, curr_min - 570)
        elif curr_min >= 780: # 13:00後
            m_minutes = min(240, 120 + (curr_min - 780))
        else:
            m_minutes = 120
            
        # 歷史事實與最新快照動態合流 (Join)
        # 排除歷史庫中與今日日期重疊的行，保障截面乾淨
        today_str = datetime.now().strftime('%Y-%m-%d')
        master_history = master_history[master_history['date'] < today_str]
        full_matrix = pd.concat([master_history, live_snapshot], ignore_index=True)
    else:
        print("💡 [時序通知] 未獲取到盤中快照或已收盤，管線自動切換為【日級全量歷史模式】。")
        full_matrix = master_history

    # 3. 驅動解耦後的 L2 / L3 內核引擎
    factored_matrix = FactorEngineV7.generate_factors(full_matrix, is_intraday=is_intraday, m_minutes=m_minutes)
    final_matrix = SignalEngineV7.evaluate_signals(factored_matrix)

    # 持久化落盤至 L0 Parquet 存儲庫，供 MClaw 容器一鍵讀取 (§3.4.2)
    # 僅保存純淨的事實字段，嚴禁分析層污染事實層 (§1.3.2)
    fact_columns = ['ticker', 'date', 'open', 'high', 'low', 'close', 'volume', 'turnover']
    final_matrix[fact_columns].to_parquet(L0_PARQUET_PATH, index=False)
    print(f"🚀 數據合流成功！已落盤為本地基石緩存文件：{L0_PARQUET_PATH}")
    print("================================================================================")

    # 4. ⚙️ 像素級別渲染：滿足您指定格式的要求 (§10.1.1)
    latest_indices = final_matrix.groupby('ticker')['global_idx'].transform('max')
    latest_rows = final_matrix[final_matrix['global_idx'] == latest_indices]

    for _, row in latest_rows.iterrows():
        # 獲取昨日的 MA60 坐標
        ticker = row['ticker']
        ticker_series = final_matrix[final_matrix['ticker'] == ticker].sort_values(by='date')
        
        # 提取昨日的均線事實
        if len(ticker_series) >= 2:
            prev_row = ticker_series.iloc[-2]
            prev_ma60 = prev_row['ma60']
            prev_close = prev_row['close']
        else:
            prev_ma60 = row['ma60']
            prev_close = row['close']

        turning_flag = "True" if row['ma60_is_turning'] == 1 else "False"
        bias_str = f"{row['space_bias_ma60']:+.2f}%"
        
        # 嚴格輸出您指定的格式字符串
        print(f"  ✅ {ticker} -> 昨日 MA60: {prev_ma60:.2f} | 趨勢右側翻揚: {turning_flag} | 60日大底: {row['locked_base_line']:.2f}")
        print(f"     昨收: {prev_close:.2f} | 當前價: {row['close']:.2f} | 空間偏離 MA60: {bias_str}")
        print("------------------------------------------------------------")

if __name__ == "__main__":
    run_edge_pipeline()