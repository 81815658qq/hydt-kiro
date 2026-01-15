"""祝福语生成模块"""
import hashlib
from typing import List
from src.logger import get_logger

logger = get_logger(__name__)


class BlessingGenerator:
    """祝福语生成器，为访客分配吉祥祝福语"""
    
    # 四字吉祥祝福语列表（至少20个）
    BLESSINGS: List[str] = [
        "鸿运当头",
        "好运常在",
        "福星高照",
        "吉祥如意",
        "万事如意",
        "心想事成",
        "步步高升",
        "财源广进",
        "喜气洋洋",
        "笑口常开",
        "福寿安康",
        "事业有成",
        "前程似锦",
        "大吉大利",
        "五福临门",
        "学业进步",
        "得偿所愿",
        "鲲鹏之志",
        "一举夺魁",
        "平安喜乐"
    ]
    
    def __init__(self):
        """初始化祝福语生成器"""
        logger.info(f"BlessingGenerator initialized with {len(self.BLESSINGS)} blessings")
    
    def get_blessing_for_visitor(self, visitor_id: str) -> str:
        """
        为访客ID分配一个固定的祝福语
        
        使用哈希函数确保相同的访客ID总是得到相同的祝福语
        
        Args:
            visitor_id: 访客的唯一ID
            
        Returns:
            四字吉祥祝福语
        """
        # 使用MD5哈希将visitor_id映射到祝福语列表的索引
        hash_object = hashlib.md5(visitor_id.encode())
        hash_int = int(hash_object.hexdigest(), 16)
        index = hash_int % len(self.BLESSINGS)
        
        blessing = self.BLESSINGS[index]
        logger.debug(f"Assigned blessing '{blessing}' to visitor {visitor_id}")
        
        return blessing
    
    def get_random_blessing(self) -> str:
        """
        获取随机祝福语
        
        注意：此方法返回的祝福语不保证一致性，
        主要用于测试或临时场景
        
        Returns:
            随机的四字吉祥祝福语
        """
        import random
        blessing = random.choice(self.BLESSINGS)
        logger.debug(f"Generated random blessing: {blessing}")
        return blessing
    
    def get_all_blessings(self) -> List[str]:
        """
        获取所有祝福语列表
        
        Returns:
            所有祝福语的列表
        """
        return self.BLESSINGS.copy()
    
    def get_blessing_count(self) -> int:
        """
        获取祝福语总数
        
        Returns:
            祝福语的数量
        """
        return len(self.BLESSINGS)
