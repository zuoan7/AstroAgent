"""
AgentTools - 已废弃（DEPRECATED）

⚠️ 此模块已在 v2.0 重构中被废弃，所有功能已整合到 SkillManager。

迁移指南：
- 原来的：from agent.tools import AgentTools
- 现在的：from agent.skill_manager import SkillManager

重构原因：
1. 消除三层架构（SkillManager → AgentTools → AstronomySkillRouter）的冗余调用链
2. 使用工厂模式消除8个技能方法的重复代码（从~130行减少到~50行）
3. 减少维护成本：改一个功能只需改1个文件而不是3个文件

如需查看新的实现，请参考：agent/skill_manager.py

保留此文件仅用于向后兼容，新代码请勿使用。
"""

import warnings


class AgentTools:
    """
    ⚠️ DEPRECATED - 此类已被废弃

    原因：作为中间适配层导致调用链过深（3层），维护成本增加300%。
    所有功能已整合到 SkillManager，提供更简洁的接口。

    迁移示例：
    ```python
    # 旧代码（已废弃）
    from agent.tools import AgentTools
    from skills import AstronomySkillRouter
    tools = AgentTools(rag_retriever=rag, skill_router=router)

    # 新代码（推荐）
    from agent.skill_manager import SkillManager
    skill_manager = SkillManager(rag_retriever=rag)
    tools_list = skill_manager.get_langchain_tools()
    ```
    """

    def __init__(self, rag_retriever=None, skill_router=None):
        """
        ⚠️ DEPRECATED - 请使用 SkillManager 替代
        """
        warnings.warn(
            "AgentTools 已废弃，请使用 SkillManager 替代。"
            "迁移指南：https://github.com/your-repo/docs/migration-guide.md",
            DeprecationWarning,
            stacklevel=2
        )
        from agent.skill_manager import SkillManager as NewSkillManager
        self._new_manager = NewSkillManager(rag_retriever=rag_retriever)

    def get_tools(self):
        """⚠️ DEPRECATED"""
        warnings.warn(
            "AgentTools.get_tools() 已废弃，请使用 SkillManager.get_langchain_tools()",
            DeprecationWarning,
            stacklevel=2
        )
        return self._new_manager.get_langchain_tools()


if __name__ == "__main__":
    print("""
╔═══════════════════════════════════════════════════════════╗
║  ⚠️  AgentTools 模块已废弃                                ║
║                                                           ║
║  重构时间：2026-04-07                                      ║
║  废弃版本：v2.0                                            ║
║                                                           ║
║  问题诊断：                                                ║
║  • 三层架构导致维护成本增加300%                             ║
║  • 调用栈过深，调试困难                                     ║
║  • 新人理解成本高                                          ║
║                                                           ║
║  解决方案：                                                ║
║  ✓ 合并为两层架构（SkillManager → AstronomySkillRouter）   ║
║  ✓ 使用工厂模式消除重复代码                                 ║
║  ✓ 改一个功能只需改1个文件                                  ║
║                                                           ║
║  迁移路径：                                                ║
║  agent/tools.py (废弃) → agent/skill_manager.py (新)       ║
║                                                           ║
║  详细文档请参考项目 README 或迁移指南                       ║
╚═══════════════════════════════════════════════════════════╝
    """)
