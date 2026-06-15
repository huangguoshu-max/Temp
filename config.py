# -*- coding: utf-8 -*-
"""
OpenClaw 系統配置中心 - 依據 §7 & §11.5 規範
鎖定全局魔術數字，禁止硬編碼，從環境變量安全注入憑證。
"""
import os

class SystemConfig:
    # --- 交易市場時鐘與限制 ---
    MARKET_LIMITS = {
        "A_SHARE": {"t_total": 240, "tail_lock_minute": 1420},  # 14:30後形態鎖定
        "HK_SHARE": {"t_total": 330, "tail_lock_minute": 1545}
    }
    
    # --- §6.3 蔡森狀態機核心閾值參數 (支持動態注入) ---
    DEFAULT_REGIME = {
        "theta_panic": 1.2,        # 恐慌放量門檻
        "theta_hollow": 0.3333,    # 洗盤極度縮量門檻
        "theta_buy": 1.5,          # 買入進攻量門檻
        "max_setup_days": 30,      # 破位後黃金恢復窗口期
        "base_risk_budget": 0.01,  # §7.2.1 單筆風險賬戶比例 (1%)
        "max_position_cap": 0.25   # 單標的絕對持倉上限 (25%)
    }

    @staticmethod
    def get_secure_secret(key: str, default: str = "") -> str:
        """安全憑證獲取，防範Token烘進鏡像 (§11.5)"""
        return os.environ.get(f"OPENCLAW_{key.upper()}", default)