from typing import List, Dict, Any
from dataclasses import dataclass
from config import settings


@dataclass
class Message:
    """对话消息"""
    role: str  # user 或 assistant
    content: str
    timestamp: float


class ShortTermMemory:
    """短期记忆管理"""
    
    def __init__(self):
        self.messages: List[Message] = []
        self.max_size = settings.MEMORY_SIZE
    
    def add_message(self, role: str, content: str, timestamp: float):
        """添加消息到记忆"""
        message = Message(role=role, content=content, timestamp=timestamp)
        self.messages.append(message)
        
        # 保持记忆大小限制
        if len(self.messages) > self.max_size:
            self.messages = self.messages[-self.max_size:]
    
    def get_recent_messages(self, window: int = None) -> List[Dict[str, str]]:
        """获取最近的消息"""
        if window is None:
            window = settings.MEMORY_WINDOW
        
        recent_messages = self.messages[-window:]
        return [
            {"role": msg.role, "content": msg.content}
            for msg in recent_messages
        ]
    
    def clear(self):
        """清空记忆"""
        self.messages.clear()
    
    def get_size(self) -> int:
        """获取当前记忆大小"""
        return len(self.messages)
